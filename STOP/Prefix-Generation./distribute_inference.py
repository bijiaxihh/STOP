import os
import sys
import signal
import threading
import pandas as pd
import numpy as np
import subprocess
import re
import shutil
import time
import py_compile
import glob
from concurrent.futures import ThreadPoolExecutor
import nbformat
from nbconvert import PythonExporter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def repo_path(*parts):
    return os.path.join(REPO_ROOT, *parts)

# =================配置区域=================
# 包含答案的题目文件路径
# LOCAL_REFERENCE_FILE = './artifacts/data/reference.csv'
# LOCAL_REFERENCE_FILE = './artifacts/data/eval_large.csv'
DEFAULT_LOCAL_REFERENCE_FILE = repo_path('artifacts', 'data', 'reference.csv')

# Notebook路径
# NOTEBOOK_FILENAME = './Prefix-Generation./saving-trace-fix.ipynb'
DEFAULT_NOTEBOOK_FILENAME = repo_path('Prefix-Generation.', 'saving-trace-fix.ipynb')
DEFAULT_MODEL_PATH = repo_path('artifacts', 'models', 'gpt-oss-120b')

TOTAL_GPUS = int(os.environ.get('TOTAL_GPUS', '8'))
GPUS_PER_WORKER = 1      # gpt-oss: 1, Qwen3.5-397B: 4, Kimi-K2.5: 8
NUM_WORKERS = TOTAL_GPUS // GPUS_PER_WORKER

LOCAL_REFERENCE_FILE = os.environ.get('LOCAL_REFERENCE_FILE') or DEFAULT_LOCAL_REFERENCE_FILE
NOTEBOOK_FILENAME = os.environ.get('NOTEBOOK_FILENAME') or DEFAULT_NOTEBOOK_FILENAME
MODEL_PATH = os.environ.get('MODEL_PATH') or DEFAULT_MODEL_PATH
OUTPUT_DIR = (os.environ.get('OUTPUT_DIR') or '').strip()
TIKTOKEN_ENCODINGS_BASE = (
    os.environ.get('TIKTOKEN_ENCODINGS_BASE')
    or repo_path('artifacts', 'tiktoken_encodings')
)

VLLM_PORT_BASE = 18000
VLLM_PORT_STRIDE = 10
MASTER_PORT_BASE = 29500
WORKER_PYTHON = os.environ.get('WORKER_PYTHON') or sys.executable
# ===========================================

# =====================================================================
#                         内存看门狗
# =====================================================================
KERNEL_RSS_LIMIT_GB = 30  # 单个 ipykernel 允许的最大 RSS (GB)

def _read_cgroup_file(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None

def _get_memory_info():
    lines = []
    current = _read_cgroup_file('/sys/fs/cgroup/memory.current')
    limit = _read_cgroup_file('/sys/fs/cgroup/memory.max')
    if current:
        cur_gb = int(current) / (1024**3)
        lim_str = f"{int(limit) / (1024**3):.1f}G" if limit and limit != 'max' else 'unlimited'
        lines.append(f"  cgroup memory: {cur_gb:.1f}G / {lim_str}")

    stat = _read_cgroup_file('/sys/fs/cgroup/memory.stat')
    if stat:
        stat_dict = {}
        for line in stat.split('\n'):
            parts = line.split()
            if len(parts) == 2:
                stat_dict[parts[0]] = int(parts[1])
        anon = stat_dict.get('anon', 0) / (1024**3)
        file_cache = stat_dict.get('file', 0) / (1024**3)
        slab = stat_dict.get('slab', 0) / (1024**3)
        shmem = stat_dict.get('shmem', 0) / (1024**3)
        lines.append(f"  breakdown: anon={anon:.1f}G  file/pagecache={file_cache:.1f}G  slab={slab:.1f}G  shmem={shmem:.1f}G")

    procs = []
    try:
        for pid_dir in os.listdir('/proc'):
            if not pid_dir.isdigit():
                continue
            try:
                with open(f'/proc/{pid_dir}/status') as f:
                    status = f.read()
                rss_kb = 0
                name = pid_dir
                for l in status.split('\n'):
                    if l.startswith('VmRSS:'):
                        rss_kb = int(l.split()[1])
                    elif l.startswith('Name:'):
                        name = l.split('\t', 1)[1].strip()
                if rss_kb > 100_000:
                    with open(f'/proc/{pid_dir}/cmdline') as f:
                        cmdline = f.read().replace('\x00', ' ')[:120]
                    procs.append((rss_kb, pid_dir, name, cmdline))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
    except Exception:
        pass

    if procs:
        procs.sort(reverse=True)
        total_rss = sum(p[0] for p in procs)
        lines.append(f"  process RSS total (>100MB): {total_rss / (1024**2):.1f}G  ({len(procs)} procs)")
        for rss_kb, pid, name, cmdline in procs[:15]:
            lines.append(f"    [{pid:>7}] {rss_kb / (1024**2):6.2f}G  {name:<15} {cmdline[:80]}")
    return '\n'.join(lines)

def start_memory_monitor(log_path, interval=30, stop_event=None):
    """后台线程：每 interval 秒记录内存快照 + 杀掉超限的 ipykernel 进程。"""
    def _kill_oversized_kernels(f, ts, elapsed_min):
        limit_kb = KERNEL_RSS_LIMIT_GB * 1024 * 1024
        try:
            for pid_dir in os.listdir('/proc'):
                if not pid_dir.isdigit():
                    continue
                try:
                    with open(f'/proc/{pid_dir}/cmdline') as cf:
                        cmdline = cf.read()
                    if 'ipykernel_launcher' not in cmdline:
                        continue
                    with open(f'/proc/{pid_dir}/status') as sf:
                        for line in sf:
                            if line.startswith('VmRSS:'):
                                rss_kb = int(line.split()[1])
                                if rss_kb > limit_kb:
                                    rss_gb = rss_kb / (1024 * 1024)
                                    msg = (f"⚠️ [{ts}] +{elapsed_min:.0f}min "
                                           f"KILLING ipykernel PID {pid_dir} "
                                           f"(RSS={rss_gb:.1f}G > {KERNEL_RSS_LIMIT_GB}G limit)\n")
                                    f.write(msg)
                                    f.flush()
                                    print(msg, end='')
                                    os.kill(int(pid_dir), signal.SIGKILL)
                            break
                except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
                    continue
        except Exception:
            pass

    def _monitor():
        with open(log_path, 'w') as f:
            f.write(f"=== Memory Monitor started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"=== Kernel RSS limit: {KERNEL_RSS_LIMIT_GB} GB ===\n")
            f.flush()
            while not (stop_event and stop_event.is_set()):
                ts = time.strftime('%H:%M:%S')
                elapsed_min = (time.time() - t0) / 60
                info = _get_memory_info()
                f.write(f"\n[{ts}] +{elapsed_min:.0f}min\n{info}\n")
                f.flush()
                _kill_oversized_kernels(f, ts, elapsed_min)
                for _ in range(interval):
                    if stop_event and stop_event.is_set():
                        break
                    time.sleep(1)
            f.write(f"\n=== Memory Monitor stopped at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    t0 = time.time()
    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    return t

# =====================================================================

def setup_workspace(num_chunks):
    """清理并创建工作目录，切分数据。work_dir 用绝对路径，避免受 cwd 影响。"""
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    _beijing = _tz(_td(hours=8))
    if OUTPUT_DIR:
        work_dir = os.path.abspath(OUTPUT_DIR)
    else:
        work_dir = os.path.join(_script_dir, f'distributed_work_{_dt.now(_beijing).strftime("%m%d%H%M")}')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    
    if not os.path.exists(LOCAL_REFERENCE_FILE):
        print(f"❌ Error: {LOCAL_REFERENCE_FILE} not found.")
        sys.exit(1)
        
    df = pd.read_csv(LOCAL_REFERENCE_FILE)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    chunks = np.array_split(df, num_chunks)
    chunk_paths = []
    
    for i, chunk in enumerate(chunks):
        worker_dir = os.path.join(work_dir, f'worker_{i}')
        os.makedirs(worker_dir, exist_ok=True)
        chunk_path = os.path.join(worker_dir, 'reference_chunk.csv')
        chunk.to_csv(chunk_path, index=False)
        chunk_paths.append(os.path.abspath(chunk_path))
        print(f"   - Worker {i}: {len(chunk)} questions")
        
    return work_dir, chunk_paths

def patch_notebook_script(notebook_path, work_dir):
    """将Notebook转为Python并施加'手术'（Monkey Patch 劫持 pd.read_csv，不依赖正则）"""
    print(f"🔄 Converting {notebook_path} to Python...")

    cleaned_notebook = os.path.join(work_dir, 'notebook_cleaned.ipynb')
    with open(notebook_path, 'rb') as f:
        raw = f.read().rstrip(b'\x00')
    with open(cleaned_notebook, 'wb') as f:
        f.write(raw)

    output_basename = 'runner_nbconverted'
    subprocess.run(
        [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'python', cleaned_notebook,
         '--output', output_basename, '--output-dir', work_dir],
        check=True
    )
    script_name = os.path.join(work_dir, f'{output_basename}.py')
    
    with open(script_name, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🔧 Patching the script using Monkey Patch technique...")
    
    # === 核心：Monkey Patch 劫持 pd.read_csv，运行时重定向 reference 路径到 TARGET_CSV ===
    monkey_patch_code = """
# ================= DISTRIBUTED INFERENCE PATCH START =================
import os
import pandas as pd
import sys

# 劫持 display：在 stdout.log 里打印表格/对象
def _display_to_stdout(obj):
    try:
        import pandas as _pd
    except Exception:
        _pd = None
    try:
        import polars as _pl
    except Exception:
        _pl = None

    if _pd is not None and isinstance(obj, _pd.DataFrame):
        with _pd.option_context(
            'display.max_rows', 20,
            'display.max_columns', 50,
            'display.width', 120,
            'display.max_colwidth', 120
        ):
            print(obj.to_string(index=False))
        return True
    if _pd is not None and isinstance(obj, _pd.Series):
        with _pd.option_context(
            'display.max_rows', 20,
            'display.max_columns', 50,
            'display.width', 120,
            'display.max_colwidth', 120
        ):
            print(obj.to_string())
        return True
    if _pl is not None and isinstance(obj, _pl.DataFrame):
        print(obj)
        return True
    return False

def display(*args, **kwargs):
    for obj in args:
        if not _display_to_stdout(obj):
            print(obj)

# 劫持 pd.read_csv
_original_read_csv = pd.read_csv

def _patched_read_csv(*args, **kwargs):
    # 获取第一个参数（路径），可能是位置参数 args[0] 或关键字参数 'filepath_or_buffer'
    path = args[0] if args else kwargs.get('filepath_or_buffer')
    
    # 只要是 CSV 文件就尝试重定向（更宽松的条件）
    if isinstance(path, str) and path.endswith('.csv'):
        target_csv = os.environ.get('TARGET_CSV')
        if target_csv:
            print(f"🔀 [Distributed Patch] Redirecting read_csv from '{path}' to '{target_csv}'")
            new_args = list(args)
            if new_args:
                new_args[0] = target_csv
            else:
                kwargs['filepath_or_buffer'] = target_csv
            return _original_read_csv(*new_args, **kwargs)
            
    return _original_read_csv(*args, **kwargs)

pd.read_csv = _patched_read_csv

# 轻量重试：用于 Jupyter kernel 初始化的偶发端口/就绪失败
def _create_sandbox_with_retry(
    timeout,
    max_retries=5,
    base_wait=1.5,
    max_wait=20.0,
    initial_jitter_max=0.6,
):
    import time as _time
    import random as _random
    import traceback as _traceback

    # 先做一次小抖动，打散同一时刻大量并发建核
    if initial_jitter_max > 0:
        _time.sleep(_random.uniform(0.0, initial_jitter_max))

    transient_markers = (
        "address already in use",
        "kernel didn't respond",
        "kernel died before replying to kernel_info",
        "kernel died",
        "zmqerror",
        "connection refused",
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return AIMO3Sandbox(timeout=timeout)
        except Exception as e:
            last_error = e
            err_text = f"{type(e).__name__}: {e}\\n{_traceback.format_exc()}".lower()
            is_transient = any(marker in err_text for marker in transient_markers)

            if is_transient and attempt < max_retries:
                exp_backoff = min(max_wait, base_wait * (2 ** (attempt - 1)))
                jitter = _random.uniform(0.0, 0.8)
                wait_s = exp_backoff + jitter
                print(
                    f"⚠️ [Distributed Patch] Sandbox init failed ({e}); "
                    f"retry {attempt}/{max_retries} in {wait_s:.2f}s"
                )
                _time.sleep(wait_s)
                continue
            raise

    raise last_error

# 隔离 Kaggle Evaluation gRPC 端口，避免多个 worker 误连到同一服务端
def _patch_grpc_ports():
    try:
        worker_id = int(os.environ.get('WORKER_ID', '0'))
    except Exception:
        worker_id = 0

    # 为所有 worker（包括 0）统一使用高位端口，避免与系统默认的 50051 冲突
    base = 60050 + worker_id * 3
    ports = [base, base + 1, base + 2]

    try:
        import kaggle_evaluation.core.relay as _relay
        _relay.GRPC_PORTS = ports
        _relay._RETRY_SLEEP_SECONDS = 1 / len(_relay.GRPC_PORTS)
        print(f"🔧 [Distributed Patch] Using gRPC ports {ports} for worker {worker_id}")
    except Exception as e:
        print(f"⚠️ [Distributed Patch] Failed to patch gRPC ports: {e}")

_patch_grpc_ports()

# 退出时生成 submission.csv，供主进程合并
import atexit

def _write_submission_at_exit():
    try:
        preds = globals().get('predictions')
        acc = globals().get('attempt_accuracy', {})
        if isinstance(preds, dict) and preds:
            import pandas as _pd
            df = _pd.DataFrame({'id': list(preds.keys()), 'answer': list(preds.values())})
            df['id'] = df['id'].astype(str)
            if acc:
                df['accuracy'] = df['id'].map(acc).fillna('')
            df.sort_values('id').to_csv('submission.csv', index=False)
            print(f"💾 [Distributed Patch] Saved submission.csv ({len(df)} rows)")
    except Exception as e:
        print(f"⚠️ [Distributed Patch] Failed to save submission.csv: {e}")

atexit.register(_write_submission_at_exit)

# 记录模型路径，待 CFG 定义后覆盖
_PATCH_MODEL_PATH = os.environ.get('TARGET_MODEL_PATH', '')
_TRACE_STAGE1_BUDGET = int(os.environ.get('TRACE_STAGE1_BUDGET', '0') or '0')
# ================= DISTRIBUTED INFERENCE PATCH END =================
"""

    # 在第一个 import pandas 所在行的下一行插入补丁（保证 pd 已存在）
    match = re.search(r"^(import pandas[^\n]*)$", content, re.MULTILINE)
    if match:
        content = content[:match.end()] + "\n" + monkey_patch_code + "\n" + content[match.end():]
    else:
        content = monkey_patch_code + "\n" + content

    # === Patch: 在 CFG class 定义后覆盖 model_path ===
    content = re.sub(
        r"(class CFG:.*?min_p\s*=\s*[\d.]+)",
        r"""\1

# [Distributed Patch] Override model_path from environment
if _PATCH_MODEL_PATH:
    CFG.model_path = _PATCH_MODEL_PATH
    print(f"🔀 [Distributed Patch] CFG.model_path -> {_PATCH_MODEL_PATH}")

# [Distributed Patch] Use a more conservative GPU memory target for single-GPU GPT-OSS loading
CFG.gpu_memory_utilization = 0.95
print(f"🔧 [Distributed Patch] CFG.gpu_memory_utilization -> {CFG.gpu_memory_utilization}")

# [Distributed Patch] Reduce server concurrency to lower startup memory pressure
CFG.batch_size = 16
print(f"🔧 [Distributed Patch] CFG.batch_size -> {CFG.batch_size}")

# [Distributed Patch] Align sampling temperature with local generation config
CFG.temperature = 1.0
print(f"🔧 [Distributed Patch] CFG.temperature -> {CFG.temperature}")
""",
        content,
        flags=re.DOTALL
    )

    # === 其他必要替换 (端口、CUDA、Log、gateway) ===
    content = content.replace(
        "./artifacts/tiktoken_encodings",
        TIKTOKEN_ENCODINGS_BASE
    )
    content = content.replace(
        "import kaggle_evaluation.aimo_3_inference_server",
        "import kaggle_evaluation.aimo_2_inference_server as aimo_3_inference_server"
    )
    content = content.replace(
        "kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer",
        "aimo_3_inference_server.AIMO2InferenceServer"
    )
    content = content.replace(
        "    Conversation\n)",
        "    Conversation,\n    RenderConversationConfig\n)"
    )
    content = re.sub(r"os\.environ\['CUDA_VISIBLE_DEVICES'\]\s*=\s*['\"][0-9,]+['\"]", "pass", content)
    content = re.sub(r"solver\s*=\s*AIMO3Solver\(CFG\)", 
                     "solver = AIMO3Solver(CFG, port=int(os.environ.get('VLLM_PORT', 8000)))", content)
    content = content.replace(
        """            for _ in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)

                if max_tokens < self.cfg.buffer_tokens:
                    break
""",
        """            stage1_budget = max(0, _TRACE_STAGE1_BUDGET)
            stage1_complete = stage1_budget == 0

            for _ in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)

                if max_tokens < self.cfg.buffer_tokens:
                    break

                if not stage1_complete:
                    remaining_stage1_tokens = stage1_budget - total_tokens
                    if remaining_stage1_tokens <= 0:
                        stage1_complete = True
                    else:
                        max_tokens = min(max_tokens, remaining_stage1_tokens)
"""
    )
    content = content.replace(
        """                new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]
""",
        """                new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]

                if (not stage1_complete) and total_tokens >= stage1_budget:
                    stage1_complete = True
"""
    )

    content = content.replace(
        "        logprobs_buffer = []\n        all_token_ids = []\n        last_token_buffer = []\n        last_text_chunks = []\n",
        "        logprobs_buffer = []\n        all_token_ids = []\n        last_token_buffer = []\n        last_text_chunks = []\n        all_completion_token_ids = []\n        all_completion_text_parts = []\n        full_render_cfg = RenderConversationConfig(auto_drop_analysis=False)\n        saved_harmony_token_ids = []\n",
    )

    content = content.replace(
        "            conversation = Conversation.from_messages(messages)\n\n            stage1_budget = max(0, _TRACE_STAGE1_BUDGET)\n",
        "            conversation = Conversation.from_messages(messages)\n            saved_harmony_token_ids = list(encoding.render_conversation_for_completion(conversation, Role.ASSISTANT, full_render_cfg))\n\n            stage1_budget = max(0, _TRACE_STAGE1_BUDGET)\n",
    )

    content = content.replace(
        "                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)\n",
        "                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT, full_render_cfg)\n                if saved_harmony_token_ids != list(prompt_ids):\n                    prompt_ids_list = list(prompt_ids)\n                    if prompt_ids_list[:len(saved_harmony_token_ids)] == saved_harmony_token_ids:\n                        saved_harmony_token_ids.extend(prompt_ids_list[len(saved_harmony_token_ids):])\n                    else:\n                        saved_harmony_token_ids = prompt_ids_list.copy()\n",
    )

    content = content.replace(
        "                last_token_buffer = list(token_buffer)\n                last_text_chunks = list(text_chunks)\n",
        "                all_completion_token_ids.extend(token_buffer)\n                all_completion_text_parts.extend(text_chunks)\n                saved_harmony_token_ids.extend(token_buffer)\n\n                last_token_buffer = list(token_buffer)\n                last_text_chunks = list(text_chunks)\n",
    )

    content = content.replace(
        "                    conversation.messages.extend(tool_responses)\n",
        "                    conversation.messages.extend(tool_responses)\n                    for tool_response in tool_responses:\n                        try:\n                            saved_harmony_token_ids.extend(encoding.render(tool_response))\n                        except Exception as render_tool_exc:\n                            print(f'Error rendering tool response for {question_id} attempt {attempt_index}: {render_tool_exc}')\n",
    )

    content = content.replace(
        "            # After all iterations, render the complete conversation to get all token IDs\n"
        "            # This includes initial prompt, all generated tokens, and all tool responses\n"
        "            # Moved to finally block to ensure saving even when exceptions occur\n"
        "            if save_tokens and question_id and conversation is not None:\n"
        "                try:\n"
        "                    all_token_ids = list(encoding.render_conversation_for_completion(conversation, Role.ASSISTANT))\n"
        "                except Exception as render_exc:\n"
        "                    print(f'Error rendering conversation for {question_id} attempt {attempt_index}: {render_exc}')\n",
        "            # Use the token-level accumulator so saved traces preserve the full Harmony path.\n"
        "            if save_tokens and question_id and saved_harmony_token_ids:\n"
        "                all_token_ids = list(saved_harmony_token_ids)\n",
    )

    content = content.replace(
        "                    trailing_text = ''\n"
        "                    if final_answer is not None and last_text_chunks:\n"
        "                        trailing_text = ''.join(last_text_chunks)\n"
        "\n"
        "                    with open(f'{base_path}-text.txt', 'w') as f:\n"
        "                        if trailing_text:\n"
        "                            render_suffix = '<|start|>assistant'\n"
        "                            if detokenized_text.endswith(render_suffix):\n"
        "                                f.write(detokenized_text[:-len(render_suffix)])\n"
        "                            else:\n"
        "                                f.write(detokenized_text)\n"
        "                            f.write(trailing_text)\n"
        "                        else:\n"
        "                            f.write(detokenized_text)\n",
        "                    with open(f'{base_path}-text.txt', 'w') as f:\n"
        "                        f.write(detokenized_text)\n",
    )
    content = re.sub(
        r"(?m)^(\s*)return\s+AIMO3Sandbox\(timeout=self\.cfg\.jupyter_timeout\)\s*$",
        r"\1return _create_sandbox_with_retry(self.cfg.jupyter_timeout)",
        content
    )
    content = re.sub(r"_next_port\s*=\s*50000", 
                     "_next_port = 20000 + int(os.environ.get('WORKER_ID', 0)) * 500", content)

    content = content.replace(
        """    def close(self):

        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()

        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

            with contextlib.suppress(Exception):
                self._km.cleanup_resources()
""",
        """    def close(self):

        suppress = getattr(contextlib, 'suppress', None)
        if suppress is None:
            return

        with suppress(Exception):
            if self._client:
                self._client.stop_channels()

        if self._owns_kernel and self._km is not None:
            with suppress(Exception):
                self._km.shutdown_kernel(now=True)

            with suppress(Exception):
                self._km.cleanup_resources()
"""
    )

    # 替换 _get_next_ports：检查端口可用性，跳过被占用的端口（TIME_WAIT / 其他进程）
    content = re.sub(
        r"def _get_next_ports\(cls, count: int = 5\).*?return ports",
        """def _get_next_ports(cls, count: int = 5) -> list[int]:
        import socket as _socket
        with cls._port_lock:
            ports = []
            while len(ports) < count:
                port = cls._next_port
                cls._next_port += 1
                try:
                    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                        _s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 0)
                        _s.bind(('127.0.0.1', port))
                    ports.append(port)
                except OSError:
                    continue
            return ports""",
        content,
        flags=re.DOTALL
    )
    content = re.sub(
        r"(?m)^(\s*)self\._km\.start_kernel\(env=env,\s*extra_arguments=\['--Application\.log_level=CRITICAL'\]\)\s*$",
        r"""\1print(f"🔍 [Sandbox] Requested ports: {ports}")
\1print(f"🔍 [Sandbox] Connection file (pre-start): {self._km.connection_file}")
\1self._km.cache_ports = False
\1print(f"🔍 [Sandbox] cache_ports forced to: {self._km.cache_ports}")
\1self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])
\1try:
\1    conn_file = self._km.connection_file
\1    conn_dir = getattr(self._km, "connection_dir", None)
\1    conn_path = conn_file
\1    if conn_file and conn_dir and not os.path.isabs(conn_file):
\1        conn_path = os.path.join(conn_dir, conn_file)
\1    print(f"🔍 [Sandbox] Connection file (post-start): {conn_file}")
\1    if conn_path and os.path.exists(conn_path):
\1        with open(conn_path, 'r') as _cf:
\1            print(f"🔍 [Sandbox] Connection file contents: {_cf.read().strip()}")
\1    else:
\1        print("🔍 [Sandbox] Connection file not found on disk")
\1except Exception as _e:
\1    print(f"⚠️ [Sandbox] Failed to read connection file: {_e}")""",
        content
    )
    content = re.sub(
        r"open\('vllm_server\.log',\s*'([rw])'\)", 
        r'''open(f'vllm_server_{os.environ.get("WORKER_ID")}.log', '\1')''', 
        content
    )
    
    # run_local_gateway 固定读本地 test.csv（每个 worker 的 df 已被劫持为 chunk，to_csv 生成的就是 chunk）
    def gateway_replacer(match):
        prefix = match.group(1)
        return f"{prefix}('test.csv',)"
    pattern_gateway = r"(inference_server\.run_local_gateway\s*\(\s*(?:#.*?\n\s*)*)(\(\s*['\"].*?['\"]\s*,?\s*\))"
    content = re.sub(pattern_gateway, gateway_replacer, content, flags=re.DOTALL)

    # 本地分布式运行时绕过 kaggle_evaluation gateway，直接遍历 test.csv 调 predict()
    content = re.sub(
        r"""if os\.getenv\('KAGGLE_IS_COMPETITION_RERUN'\):\s*
    inference_server\.serve\(\)\s*

else:\s*
    inference_server\.run_local_gateway\(\s*
        # \('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference\.csv',\)\s*
        \('test\.csv',\)\s*
    \)""",
        """if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()

else:
    local_test_df = pd.read_csv('test.csv')
    print(f"🧪 [Local Loop] Loaded {len(local_test_df)} rows from test.csv")
    for row_idx, row in local_test_df.iterrows():
        row_id = row['id']
        row_question = row['problem']
        print(f"🧪 [Local Loop] Row {row_idx + 1}/{len(local_test_df)} -> {row_id}")
        predict(
            pl.Series('id', [row_id]),
            pl.Series('problem', [row_question]),
            None,
        )""",
        content,
        flags=re.DOTALL
    )

    content = re.sub(r"(?ms)^(\s*)(?!def\s)set_env\((?:.|\n)*?\)\s*", r"\1# skipped set_env\n", content)

    # 非首 worker 跳过 preload（page cache 已由第一个 worker 预热，避免重复读占内存）
    content = re.sub(
        r"(def _preload_model_weights\(self\)[^:]*:\s*\n)",
        r"""\1        _wid = int(os.environ.get('WORKER_ID', '0'))
        _gpn = int(os.environ.get('GPUS_PER_NODE', '8'))
        if _wid % _gpn != 0:
            print(f"⏭️ [Distributed Patch] Skipping model preload (worker {_wid}, not first on node)")
            return
""",
        content
    )

    # os.walk 会递归进 original/ 子目录，导致 preload 读 122GB 而不是 61GB，改为 os.listdir 只读根目录
    content = re.sub(
        r"for root, _, files in os\.walk\(self\.cfg\.model_path\):\s*\n"
        r"\s*for file_name in files:\s*\n"
        r"\s*file_path = os\.path\.join\(root, file_name\)\s*\n"
        r"\s*\n"
        r"\s*if os\.path\.isfile\(file_path\):\s*\n"
        r"\s*files_to_load\.append\(file_path\)\s*\n"
        r"\s*total_size \+= os\.path\.getsize\(file_path\)",
        "for file_name in os.listdir(self.cfg.model_path):\n"
        "            file_path = os.path.join(self.cfg.model_path, file_name)\n"
        "\n"
        "            if os.path.isfile(file_path):\n"
        "                files_to_load.append(file_path)\n"
        "                total_size += os.path.getsize(file_path)",
        content
    )


    # [Distributed Patch] wait_for_ready 用更长的超时（120秒），独立于 jupyter_timeout
    # jupyter_timeout（通常 6 秒）是代码执行超时，但 kernel 启动需要更久
    # 16 个 kernel 并发启动时 CPU 竞争激烈，6 秒远远不够
    content = re.sub(
        r"(self\._client\.wait_for_ready)\(timeout=self\._default_timeout\)\s*\n(\s*)(self\._owns_kernel = True)",
        r"""\1(timeout=120)
\2# [Distributed Patch] Verify all kernel ZMQ ports are actually bound
\2import socket as _sock_v
\2for _vp_name, _vp in [("hb", self._km.hb_port), ("shell", self._km.shell_port),
\2                       ("iopub", self._km.iopub_port), ("stdin", self._km.stdin_port),
\2                       ("control", self._km.control_port)]:
\2    try:
\2        with _sock_v.socket(_sock_v.AF_INET, _sock_v.SOCK_STREAM) as _vs:
\2            _vs.setsockopt(_sock_v.SOL_SOCKET, _sock_v.SO_REUSEADDR, 0)
\2            _vs.bind(('127.0.0.1', _vp))
\2        # bind succeeded → kernel did NOT bind this port → broken channel
\2        print(f"❌ [Sandbox] Kernel {_vp_name} port {_vp} not bound by kernel!")
\2        try:
\2            self._km.shutdown_kernel(now=True)
\2        except Exception:
\2            pass
\2        raise RuntimeError(f"Kernel {_vp_name} port {_vp} not bound after start - port conflict")
\2    except OSError:
\2        pass  # port in use by kernel = healthy
\2\3""",
        content
    )

    # vLLM 默认 swap_space=4 GiB/GPU，8 GPU 即 32 GiB 常驻 CPU 内存，加 --swap-space 0 释放
    content = re.sub(
        r"('--enable-prefix-caching')",
        r"\1,\n            '--swap-space', '0'",
        content
    )

    # vLLM 端口冲突时自动换端口重试（hostNetwork 下其他进程可能占用端口）
    if '_max_port_retries' not in content:
        content = re.sub(
            r"(self\.server_process = self\._start_server\(\).*?\n"
            r".*?self\.client = OpenAI\(.*?\n"
            r".*?base_url=.*?\n"
            r".*?\).*?\n"
            r".*?self\._wait_for_server\(\))",
            r"""# [Distributed Patch] vLLM port retry on conflict
        _max_port_retries = 5
        for _port_attempt in range(_max_port_retries):
            self.server_process = self._start_server()
            self.client = OpenAI(
                api_key='EMPTY',
                base_url=f'http://localhost:{self.port}/v1',
            )
            try:
                self._wait_for_server()
                break
            except RuntimeError as _e:
                _err_msg = str(_e)
                if 'Address already in use' in _err_msg and _port_attempt < _max_port_retries - 1:
                    self.port += 1
                    print(f"⚠️ [Distributed Patch] vLLM port conflict, retrying with port {self.port} (attempt {_port_attempt+2}/{_max_port_retries})")
                    try:
                        self.server_process.kill()
                        self.server_process.wait()
                    except Exception:
                        pass
                    continue
                raise""",
            content,
            flags=re.DOTALL
        )

    # === Patch: 在 solve_problem 中保存每次并行尝试的 detailed_results ===
    content = re.sub(
        r"(?m)^(\s+)(if detailed_results:\n\s+results_dataframe = pd\.DataFrame\(detailed_results\))",
        r"\1self.last_detailed_results = detailed_results\n\1\2",
        content
    )

    # === Patch: 添加 attempt_accuracy 全局字典 ===
    content = re.sub(
        r"predictions = \{\}\ncorrect_count = 0",
        "predictions = {}\nattempt_accuracy = {}\ncorrect_count = 0",
        content
    )

    # === Patch: 在 predict 的 global 声明中加入 attempt_accuracy ===
    content = re.sub(
        r"global correct_count, total_count, predictions",
        "global correct_count, total_count, predictions, attempt_accuracy",
        content
    )

    # === Patch: 在 predict 中追踪并行推理的正确率 ===
    # 用 (\w+) 捕获变量名（id_value 或 question_id），通过 \2 回引
    content = re.sub(
        r"(predictions\[(\w+)\] = final_answer)",
        r"""\1

    # [Distributed Patch] Track per-attempt accuracy from parallel inference
    if hasattr(solver, 'last_detailed_results'):
        _dr = solver.last_detailed_results
        _total = len(_dr)
        try:
            if \2 in ground_truth:
                _gt = ground_truth[\2]
                _gt_int = int(str(_gt).replace(',', '').strip())
                _correct = sum(1 for _r in _dr if _r.get('Answer') is not None and int(str(_r['Answer']).replace(',', '').strip()) == _gt_int)
                attempt_accuracy[\2] = f"{_correct}/{_total}"
            else:
                _valid = sum(1 for _r in _dr if _r.get('Answer') is not None)
                attempt_accuracy[\2] = f"{_valid}/{_total}"
        except (ValueError, TypeError):
            attempt_accuracy[\2] = f"?/{_total}"
""",
        content
    )

    patched_script_path = os.path.join(work_dir, 'runner_patched.py')
    with open(patched_script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    # 预检语法，避免将损坏的补丁脚本分发给所有 worker
    py_compile.compile(patched_script_path, doraise=True)
        
    print(f"✅ Patched script saved to: {patched_script_path}")
    return patched_script_path

def run_worker(worker_id, script_path, chunk_path, gpu_ids):
    worker_dir = os.path.dirname(chunk_path)
    env = os.environ.copy()
    worker_python = WORKER_PYTHON
    if not os.path.exists(worker_python):
        raise FileNotFoundError(f"Configured WORKER_PYTHON not found: {worker_python}")
    worker_bin = os.path.dirname(worker_python)
    
    # Add notebook dir to pythonpath
    nb_dir = os.path.dirname(os.path.abspath(NOTEBOOK_FILENAME))
    env['PYTHONPATH'] = nb_dir + (os.pathsep + env.get('PYTHONPATH', ''))
    
    # ===修复Bug：缓存隔离===
    #创建独立的缓存目录，防止多个进程同时写同一个缓存文件导致corrupted checksum
    cache_dir = os.path.join(worker_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    # 清理 Triton 临时目录，保留已编译缓存
    triton_cache_dir = os.path.join(cache_dir, 'triton')
    for p in glob.glob(os.path.join(triton_cache_dir, '**', 'tmp.pid_*'), recursive=True):
        shutil.rmtree(p, ignore_errors=True)
    
    env.update({
        'CUDA_VISIBLE_DEVICES': gpu_ids,
        'WORKER_ID': str(worker_id),
        'VLLM_PORT': str(VLLM_PORT_BASE + worker_id * VLLM_PORT_STRIDE),
        'MASTER_ADDR': '127.0.0.1',
        'MASTER_PORT': str(MASTER_PORT_BASE + worker_id),
        'TARGET_CSV': chunk_path,
        'TARGET_MODEL_PATH': MODEL_PATH,
        'TRACE_STAGE1_BUDGET': os.environ.get('TRACE_STAGE1_BUDGET', '0'),
        'PYTHONUNBUFFERED': '1',
        # 限制 CPU 线程数，避免 OpenBLAS 线程创建失败
        'VLLM_DISABLE_TORCH_COMPILE': '1',  
        # 添加上面这一行（禁用 torch.compile 以解决 MoE 兼容性问题）
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'OPENBLAS_NUM_THREADS': '1',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        #隔离Triton缓存
        'TRITON_CACHE_DIR': triton_cache_dir,
        #隔离PyTorch Inductor缓存
        'TORCHINDUCTOR_CACHE_DIR': os.path.join(cache_dir, 'inductor'),
        #隔离其他PyTorch扩展缓存
        'PYTORCH_KERNEL_CACHE_PATH': os.path.join(cache_dir, 'kernels'),
    })
    env['PATH'] = worker_bin + os.pathsep + env.get('PATH', '')
    env.pop('KAGGLE_IS_COMPETITION_RERUN', None)
    
    shutil.copy(script_path, os.path.join(worker_dir, 'runner.py'))
    
    print(f"🚀 [Worker {worker_id}] GPU {gpu_ids} | CSV: {os.path.basename(chunk_path)} | PY: {worker_python}")
    
    with open(os.path.join(worker_dir, 'stdout.log'), 'w') as out, \
         open(os.path.join(worker_dir, 'stderr.log'), 'w') as err:
        proc = subprocess.Popen([worker_python, 'runner.py'], cwd=worker_dir, env=env, stdout=out, stderr=err)
        proc.wait()
        
    return proc.returncode

def merge_results(work_dir):
    print("\n🔗 Merging results...")
    all_dfs = []
    for i in range(NUM_WORKERS):
        p = os.path.join(work_dir, f'worker_{i}', 'submission.csv')
        if os.path.exists(p):
            df = pd.read_csv(p)
            if 'id' in df.columns:
                df['id'] = df['id'].astype(str)
            all_dfs.append(df)
            
    if all_dfs:
        final = pd.concat(all_dfs).sort_values('id')
        correct_count = None
        total_count = None

        # 若参考文件包含 answer，则附加 is_correct 并将正确率写入文件名
        try:
            ref_df = pd.read_csv(LOCAL_REFERENCE_FILE)
            if 'id' in ref_df.columns and 'answer' in ref_df.columns:
                ref_df['id'] = ref_df['id'].astype(str)
                final['id'] = final['id'].astype(str)

                pred_num = pd.to_numeric(final['answer'], errors='coerce')
                ref_num = pd.to_numeric(ref_df['answer'], errors='coerce')

                if ref_df['id'].duplicated().any():
                    # 重复 id 时，按 id 的候选答案集合判断是否命中（避免 merge 膨胀）
                    answer_sets = (
                        ref_df.assign(_ans_num=ref_num)
                        .groupby('id')['_ans_num']
                        .apply(lambda s: set(s.dropna().tolist()))
                        .to_dict()
                    )

                    def _is_correct_row(row):
                        id_ = row['id']
                        ans = pd.to_numeric(row['answer'], errors='coerce')
                        if id_ not in answer_sets or pd.isna(ans):
                            return pd.NA
                        return int(ans in answer_sets[id_])

                    final['is_correct'] = final.apply(_is_correct_row, axis=1)
                else:
                    ref_map = dict(zip(ref_df['id'], ref_num))
                    final['_gt_answer'] = final['id'].map(ref_map)
                    final['is_correct'] = (pred_num == final['_gt_answer']).astype('Int64')
                    final.loc[final['_gt_answer'].isna() | pred_num.isna(), 'is_correct'] = pd.NA
                    final = final.drop(columns=['_gt_answer'])

                valid = final['is_correct'].dropna()
                correct_count = int((valid == 1).sum())
                total_count = int(valid.shape[0])
        except Exception as e:
            print(f"⚠️ Unable to compute correctness column: {e}")

        if correct_count is not None and total_count is not None and total_count > 0:
            output_name = f'final_submission_8gpu_{correct_count}_{total_count}.csv'
        else:
            output_name = 'final_submission_8gpu.csv'

        output_path = os.path.join(work_dir, output_name)
        final.to_csv(output_path, index=False)
        print(f"🎉 Saved to {output_path} ({len(final)} rows)")
    else:
        print("❌ No results found.")

def main():
    start_time = time.time()

    work_dir, chunk_paths = setup_workspace(NUM_WORKERS)
    runner_script = patch_notebook_script(NOTEBOOK_FILENAME, work_dir)

    # 启动内存看门狗（每 30 秒扫描，ipykernel RSS > 30G 自动 kill）
    mem_stop = threading.Event()
    mem_log = os.path.join(work_dir, 'memory_watchdog.log')
    start_memory_monitor(mem_log, interval=30, stop_event=mem_stop)
    print(f"🛡️ Memory watchdog started → {mem_log}")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for i in range(NUM_WORKERS):
            start_gpu = i * GPUS_PER_WORKER
            gpu_ids = ','.join(str(g) for g in range(start_gpu, start_gpu + GPUS_PER_WORKER))
            futures.append(executor.submit(run_worker, i, runner_script, chunk_paths[i], gpu_ids))
            if i < NUM_WORKERS - 1: time.sleep(30)
        for idx, f in enumerate(futures):
            rc = f.result()
            if rc != 0:
                raise RuntimeError(f"Worker {idx} exited with code {rc}")

    mem_stop.set()
    merge_results(work_dir)

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        print(f"\n⏱️ 总运行时长: {hours}小时 {minutes}分 {seconds}秒")
    elif minutes > 0:
        print(f"\n⏱️ 总运行时长: {minutes}分 {seconds}秒")
    else:
        print(f"\n⏱️ 总运行时长: {elapsed:.2f}秒")

if __name__ == '__main__':
    main()
