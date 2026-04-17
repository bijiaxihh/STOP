from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

import contextlib
import gc
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

pd = None
pl = None
display = None
KernelManager = None
OpenAI = None
Author = None
Conversation = None
HarmonyEncodingName = None
Message = None
ReasoningEffort = None
Role = None
SystemContent = None
TextContent = None
ToolNamespaceConfig = None
load_harmony_encoding = None
set_seed = None
kaggle_evaluation = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)


def env_path_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return [path for path in raw.split(os.pathsep) if path]


LOCAL_VLLM_PATH = repo_path("vllm")
LOCAL_KAGGLE_ENV_PATHS = env_path_list(
    "AIMO_KAGGLE_ENV_PATHS",
    [
        repo_path(
            "artifacts",
            "kaggle_env",
            "kaggle",
            "input",
            "ai-mathematical-olympiad-progress-prize-3",
        ),
        repo_path("artifacts", "kaggle_env", "kaggle", "working"),
        repo_path("artifacts", "kaggle_env"),
    ],
)
LOCAL_TIKTOKEN_DIR = repo_path("artifacts", "tiktoken_encodings")
LOCAL_MODEL_PATH = repo_path("artifacts", "models", "gpt-oss-120b")
LOCAL_SCORE_ROOT = repo_path("artifacts", "checkpoints", "best_checkpoint")
LOCAL_LORA_PATH = f"{LOCAL_SCORE_ROOT}/adapter/classifier"
LOCAL_ASSESS_TOKEN_CONFIG = f"{LOCAL_SCORE_ROOT}/assess_token_config.json"
LOCAL_CLASSIFIER_HEAD_PATH = f"{LOCAL_SCORE_ROOT}/classifier_head.pth"
LOCAL_ASSESS_TOKEN_EMBEDDING_PATH = f"{LOCAL_SCORE_ROOT}/assess_token_embedding.pth"
LOCAL_REFERENCE_CSV = repo_path("artifacts", "reference", "reference_val.csv")
LOCAL_VLLM_CLEAN_LORA_DIR = repo_path(
    "artifacts",
    "runtime",
    "vllm_lora_only_adapter_local",
)

def load_assess_token_config(config_path: str) -> dict[str, Any]:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as file_object:
        return json.load(file_object)


ASSESS_TOKEN_CONFIG = load_assess_token_config(LOCAL_ASSESS_TOKEN_CONFIG)


def build_vllm_compatible_lora_dir(adapter_dir: str, dst_dir: str) -> str:
    src_model = os.path.join(adapter_dir, "adapter_model.safetensors")
    src_bin = os.path.join(adapter_dir, "adapter_model.bin")
    src_cfg = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(src_cfg):
        return adapter_dir

    if os.path.exists(src_model):
        state = safe_load_file(src_model, device="cpu")
    elif os.path.exists(src_bin):
        state = torch.load(src_bin, map_location="cpu")
    else:
        return adapter_dir

    keep = {key: value for key, value in state.items() if ".lora_" in key}
    if not keep:
        raise RuntimeError("No LoRA tensors found in adapter checkpoint")

    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src_cfg, os.path.join(dst_dir, "adapter_config.json"))
    safe_save_file(keep, os.path.join(dst_dir, "adapter_model.safetensors"))
    return dst_dir


def resolve_lora_target_modules(adapter_dir: str) -> list[str]:
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        return []

    with open(config_path, "r", encoding="utf-8") as file_object:
        adapter_config = json.load(file_object)

    target_modules = adapter_config.get("target_modules")
    if isinstance(target_modules, list):
        return [str(item).strip() for item in target_modules if str(item).strip()]
    if isinstance(target_modules, str) and target_modules.strip():
        return [target_modules.strip()]
    return []


def resolve_existing_path(candidates: list[str], label: str) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path

    searched = "\n".join(candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{searched}")


def summarize_adapter_state(adapter_dir: str) -> None:
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    model_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    checkpoint_path = model_path if os.path.exists(model_path) else bin_path

    print(f"Adapter dir: {adapter_dir}")
    print(f"Adapter config exists: {os.path.exists(config_path)}")
    if checkpoint_path:
        try:
            size_gb = os.path.getsize(checkpoint_path) / (1024 ** 3)
            print(f"Adapter checkpoint: {checkpoint_path} ({size_gb:.2f} GiB)")
        except OSError:
            print(f"Adapter checkpoint: {checkpoint_path}")
    else:
        print("Adapter checkpoint: <missing>")
        return

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as file_object:
                adapter_config = json.load(file_object)
            print(f"Adapter target_modules: {adapter_config.get('target_modules')}")
            print(f"Adapter rank: {adapter_config.get('r')}")
        except Exception as exc:
            print(f"Failed to read adapter_config.json: {exc}")

    try:
        if checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file as _safe_load_file

            state = _safe_load_file(checkpoint_path, device='cpu')
        else:
            state = torch.load(checkpoint_path, map_location='cpu')
    except Exception as exc:
        print(f"Failed to inspect adapter checkpoint: {exc}")
        return

    keys = list(state.keys())
    print(f"Adapter tensor count: {len(keys)}")
    print(f"Adapter sample keys: {keys[:8]}")
    print(
        "Adapter has full embeddings: "
        f"embed_tokens={'base_model.model.model.embed_tokens.weight' in state}, "
        f"lm_head={'base_model.model.lm_head.weight' in state}"
    )


def setup_runtime() -> None:
    global pd
    global pl
    global display
    global KernelManager
    global OpenAI
    global Author
    global Conversation
    global HarmonyEncodingName
    global Message
    global ReasoningEffort
    global Role
    global SystemContent
    global TextContent
    global ToolNamespaceConfig
    global load_harmony_encoding
    global set_seed
    global kaggle_evaluation

    if LOCAL_VLLM_PATH not in sys.path:
        sys.path.insert(0, LOCAL_VLLM_PATH)
    for path in LOCAL_KAGGLE_ENV_PATHS:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    tiktoken_dir = resolve_existing_path(
        [
            os.environ.get("TIKTOKEN_ENCODINGS_BASE", ""),
            LOCAL_TIKTOKEN_DIR,
            "/kaggle/input/datasets/shx789/riktoken-3-25/tiktoken_encodings",
            "/kaggle/input/riktoken-3-25/tiktoken_encodings",
        ],
        "tiktoken encodings directory",
    )

    subprocess.run(["ls", tiktoken_dir], check=True)

    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TRANSFORMERS_NO_FLAX"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("LOCAL_CUDA_VISIBLE_DEVICES", "0")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = tiktoken_dir
    raw_adapter_path = os.environ.get("AIMO_LORA_PATH", LOCAL_LORA_PATH)
    summarize_adapter_state(raw_adapter_path)
    adapter_path = build_vllm_compatible_lora_dir(
        raw_adapter_path,
        os.environ.get("AIMO_VLLM_CLEAN_LORA_DIR", LOCAL_VLLM_CLEAN_LORA_DIR),
    )
    target_modules = resolve_lora_target_modules(raw_adapter_path)
    os.environ["AIMO_ADAPTER_PATH"] = adapter_path
    if target_modules:
        os.environ["LOCAL_LORA_MODULE_ALLOWLIST"] = ",".join(target_modules)
    os.environ.setdefault("AIMO_CLASSIFIER_HEAD_PATH", LOCAL_CLASSIFIER_HEAD_PATH)
    os.environ.setdefault("AIMO_ASSESS_TOKEN_EMBEDDING_PATH", LOCAL_ASSESS_TOKEN_EMBEDDING_PATH)
    os.environ.setdefault("AIMO_SCORE_MODEL_NAME", "assess_lora")

    import pandas as _pd
    import polars as _pl
    from IPython.display import display as _display
    from jupyter_client import KernelManager as _KernelManager
    from openai import OpenAI as _OpenAI
    from openai_harmony import (
        Author as _Author,
        Conversation as _Conversation,
        HarmonyEncodingName as _HarmonyEncodingName,
        Message as _Message,
        ReasoningEffort as _ReasoningEffort,
        Role as _Role,
        SystemContent as _SystemContent,
        TextContent as _TextContent,
        ToolNamespaceConfig as _ToolNamespaceConfig,
        load_harmony_encoding as _load_harmony_encoding,
    )
    from transformers import set_seed as _set_seed

    import kaggle_evaluation.aimo_3_inference_server as _kaggle_evaluation_aimo_3_inference_server

    pd = _pd
    pl = _pl
    display = _display
    KernelManager = _KernelManager
    OpenAI = _OpenAI
    Author = _Author
    Conversation = _Conversation
    HarmonyEncodingName = _HarmonyEncodingName
    Message = _Message
    ReasoningEffort = _ReasoningEffort
    Role = _Role
    SystemContent = _SystemContent
    TextContent = _TextContent
    ToolNamespaceConfig = _ToolNamespaceConfig
    load_harmony_encoding = _load_harmony_encoding
    set_seed = _set_seed
    kaggle_evaluation = _kaggle_evaluation_aimo_3_inference_server


class CFG:

    system_prompt = (
        'You are an elite mathematical problem solver with expertise at the International '
        'Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through '
        'rigorous mathematical reasoning.\n\n'
        
        '# Problem-Solving Approach:\n'
        '1. UNDERSTAND: Carefully read and rephrase the problem in your own words. '
        'Identify what is given, what needs to be found, and any constraints.\n'
        '2. EXPLORE: Consider multiple solution strategies. Think about relevant theorems, '
        'techniques, patterns, or analogous problems. Don\'t commit to one approach immediately.\n'
        '3. PLAN: Select the most promising approach and outline key steps before executing.\n'
        '4. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly.\n'
        '5. VERIFY: Check your answer by substituting back, testing edge cases, or using '
        'alternative methods. Ensure logical consistency throughout.\n\n'
        
        '# Mathematical Reasoning Principles:\n'
        '- Break complex problems into smaller, manageable sub-problems\n'
        '- Look for patterns, symmetries, and special cases that provide insight\n'
        '- Use concrete examples to build intuition before generalizing\n'
        '- Consider extreme cases and boundary conditions\n'
        '- If stuck, try working backwards from the desired result\n'
        '- Be willing to restart with a different approach if needed\n\n'
        
        '# Verification Requirements:\n'
        '- Cross-check arithmetic and algebraic manipulations\n'
        '- Verify that your solution satisfies all problem constraints\n'
        '- Test your answer with simple cases or special values when possible\n'
        '- Ensure dimensional consistency and reasonableness of the result\n\n'
        
        '# Output Format:\n'
        'The final answer must be a non-negative integer between 0 and 99999.\n'
        'Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'
        
        'Think step-by-step and show your complete reasoning process. Quality of reasoning '
        'is as important as the final answer.'
    )
    
    tool_prompt = (
        'Use this tool to execute Python code for:\n'
        '- Complex calculations that would be error-prone by hand\n'
        '- Numerical verification of analytical results\n'
        '- Generating examples or testing conjectures\n'
        '- Visualizing problem structure when helpful\n'
        '- Brute-force verification for small cases\n\n'
        
        'The environment is a stateful Jupyter notebook. Code persists between executions.\n'
        'Always use print() to display results. Write clear, well-commented code.\n\n'
        
        'Remember: Code should support your mathematical reasoning, not replace it. '
        'Explain what you\'re computing and why before running code.'
    )
    
    preference_prompt = (
        'You have access to `math`, `numpy`, and `sympy` for:\n\n'
        
        '# Symbolic Computation (sympy):\n'
        '- Algebraic manipulation and simplification\n'
        '- Solving equations and systems of equations\n'
        '- Symbolic differentiation and integration\n'
        '- Number theory functions (primes, divisors, modular arithmetic)\n'
        '- Polynomial operations and factorization\n'
        '- Working with mathematical expressions symbolically\n\n'
        
        '# Numerical Computation (numpy):\n'
        '- Array operations and linear algebra\n'
        '- Efficient numerical calculations for large datasets\n'
        '- Matrix operations and eigenvalue problems\n'
        '- Statistical computations\n\n'
        
        '# Mathematical Functions (math):\n'
        '- Standard mathematical functions (trig, log, exp)\n'
        '- Constants like pi and e\n'
        '- Basic operations for single values\n\n'
        
        'Best Practices:\n'
        '- Use sympy for exact symbolic answers when possible\n'
        '- Use numpy for numerical verification and large-scale computation\n'
        '- Combine symbolic and numerical approaches: derive symbolically, verify numerically\n'
        '- Document your computational strategy clearly\n'
        '- Validate computational results against known cases or theoretical bounds'
    )
    


    served_model_name = "gpt-oss"
    score_model_name = "assess_lora"
    model_path = os.getenv("AIMO_MODEL_PATH", LOCAL_MODEL_PATH)
    lora_path = os.getenv("AIMO_LORA_PATH", LOCAL_LORA_PATH)
    classifier_head_path = os.getenv(
        "AIMO_CLASSIFIER_HEAD_PATH",
        LOCAL_CLASSIFIER_HEAD_PATH,
    )
    local_reference_csv = os.getenv("AIMO_REFERENCE_CSV", LOCAL_REFERENCE_CSV)

    kv_cache_dtype = "fp8_e4m3"
    dtype = "auto"

    high_problem_timeout = 900
    base_problem_timeout = 300

    notebook_limit = 17400
    server_timeout = 180

    session_timeout = 960
    jupyter_timeout = 30
    sandbox_timeout = 3

    stream_interval = 200
    context_tokens = 65536
    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 5
    batch_size = 64
    early_stop = 4
    attempts = 8
    workers = 32
    turns = 128
    seed = 42

    prefix_budget = int(os.getenv("LOCAL_PREFIX_BUDGET", "3072"))
    prefix_candidates = int(os.getenv("LOCAL_PREFIX_CANDIDATES", "32"))
    top_prefixes = int(os.getenv("LOCAL_TOP_PREFIXES", "8"))
    continuation_batch_size = int(
        os.getenv("LOCAL_CONTINUATION_BATCH_SIZE", str(top_prefixes))
    )
    num_assess_tokens = int(
        os.getenv(
            "LOCAL_NUM_ASSESS_TOKENS",
            str(ASSESS_TOKEN_CONFIG.get("num_assess_tokens", 4)),
        )
    )
    assess_special_token_id = int(
        os.getenv(
            "ASSESS_SPECIAL_TOKEN_ID",
            str(ASSESS_TOKEN_CONFIG.get("special_token_id", -1)),
        )
    )
    classify_timeout = int(os.getenv("LOCAL_CLASSIFY_TIMEOUT", "120"))
    skip_preload = os.getenv("LOCAL_SKIP_PRELOAD", "0") == "1"
    preload_progress_interval = int(os.getenv("LOCAL_PRELOAD_PROGRESS_INTERVAL", "4"))
    score_debug_dump = os.getenv("LOCAL_SCORE_DEBUG_DUMP", "1") == "1"
    score_debug_dump_dir = os.getenv(
        "LOCAL_SCORE_DEBUG_DUMP_DIR",
        repo_path("artifacts", "runtime", "recorded"),
    )
    result_debug_dump = (
        os.getenv(
            "LOCAL_RESULT_DEBUG_DUMP",
            os.getenv("LOCAL_SCORE_DEBUG_DUMP", "1"),
        )
        == "1"
    )

    gpu_memory_utilization = 0.95
    temperature = 1.2
    min_p = 0.02


@dataclass
class AttemptState:
    attempt_id: int
    seed: int
    sandbox: Any
    tool: Any
    conversation: Any
    completion_turns: int = 0
    generated_tokens: int = 0
    python_calls: int = 0
    python_errors: int = 0
    answer: int | None = None
    logprobs: list[dict[str, float]] = field(default_factory=list)

    def render_prompt_ids(self, encoding: Any) -> list[int]:
        return encoding.render_conversation_for_completion(self.conversation, Role.ASSISTANT)

class AIMO3Template:
    def get_system_content(
        self, system_prompt: str, tool_config: ToolNamespaceConfig
    ) -> SystemContent:
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_config: ToolNamespaceConfig,
    ) -> list[Message]:
        system_content = self.get_system_content(system_prompt, tool_config)
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, user_message]


class AIMO3Sandbox:
    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout: float):
        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None

        ports = self._get_next_ports(5)

        env = os.environ.copy()
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        env["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "0"
        env["JUPYTER_PLATFORM_DIRS"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["MPLBACKEND"] = "Agg"

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(env=env, extra_arguments=["--Application.log_level=CRITICAL"])

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        self.execute(
            "import math\n"
            "import numpy\n"
            "import sympy\n"
            "import itertools\n"
            "import collections\n"
            "import mpmath\n"
            "mpmath.mp.dps = 64\n"
        )

    def _format_error(self, traceback: list[str]) -> str:
        clean_lines = []
        for frame in traceback:
            clean_frame = re.sub(r"\x1b\[[0-9;]*m", "", frame)
            if 'File "' in clean_frame and "ipython-input" not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return "".join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:
        client = self._client
        effective_timeout = timeout or self._default_timeout

        msg_id = client.execute(
            code,
            store_history=True,
            allow_stdin=False,
            stop_on_error=False,
        )

        stdout_parts = []
        stderr_parts = []
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > effective_timeout:
                self._km.interrupt_kernel()
                return f"[ERROR] Execution timed out after {effective_timeout} seconds"

            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_list = content.get("traceback", [])
                stderr_parts.append(self._format_error(traceback_list))
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            return f"{stdout.rstrip()}\n{stderr}" if stdout else stderr

        return stdout if stdout.strip() else "[WARN] No output. Use print() to see results."

    def close(self) -> None:
        import contextlib as _contextlib

        with _contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()

        if self._owns_kernel and self._km is not None:
            with _contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with _contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def reset(self) -> None:
        self.execute(
            "%reset -f\n"
            "import math\n"
            "import numpy\n"
            "import sympy\n"
            "import itertools\n"
            "import collections\n"
            "import mpmath\n"
            "mpmath.mp.dps = 64\n"
        )

    def __del__(self):
        self.close()


class AIMO3Tool:
    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox=None):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        self._owns_session = sandbox is None
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self) -> None:
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code: str) -> str:
        lines = code.strip().split("\n")
        if not lines:
            return code

        last_line = lines[-1].strip()
        if "print" in last_line or "import" in last_line:
            return code
        if not last_line:
            return code
        if last_line.startswith("#"):
            return code

        lines[-1] = "print(" + last_line + ")"
        return "\n".join(lines)

    @property
    def instruction(self) -> str:
        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name="python",
            description=self.instruction,
            tools=[],
        )

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name="python")
        message = Message(author=author, content=[content]).with_recipient("assistant")
        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync_plus(self, message: Message) -> list[Message]:
        self._ensure_session()
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)

        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)
            except TimeoutError as exc:
                output = f"[ERROR] {exc}"

        return [self._make_response(output, channel=message.channel)]


class AIMO3Solver:
    def __init__(self, cfg, port: int = 8000):
        self.cfg = cfg
        self.port = port
        self.base_url = f"http://0.0.0.0:{port}/v1"
        self.api_key = "sk-local"
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        self._score_debug_dump_lock = threading.Lock()
        self._score_debug_dump_counter = 0
        self._score_debug_run_id = time.strftime("%Y%m%d_%H%M%S")

        self._preload_model_weights()
        self.server_process = self._start_server()

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.cfg.session_timeout,
        )

        self._wait_for_server()
        self._initialize_kernels()

        self.notebook_start_time = time.time()
        self.problems_remaining = 50

    def _preload_model_weights(self) -> None:
        if self.cfg.skip_preload:
            print("Skipping model weight page-cache preload.\n")
            return

        print(f"Loading model weights from {self.cfg.model_path} into OS Page Cache...")
        start_time = time.time()

        files_to_load = []
        total_size = 0

        for root, _, files in os.walk(self.cfg.model_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    files_to_load.append(file_path)
                    total_size += os.path.getsize(file_path)

        progress_lock = threading.Lock()
        files_done = 0
        bytes_done = 0

        def _read_file(path: str) -> None:
            nonlocal files_done
            nonlocal bytes_done
            with open(path, "rb") as file_object:
                while file_object.read(1024 * 1024 * 1024):
                    pass
            file_size = os.path.getsize(path)
            with progress_lock:
                files_done += 1
                bytes_done += file_size
                if (
                    files_done == len(files_to_load)
                    or files_done % max(1, self.cfg.preload_progress_interval) == 0
                ):
                    elapsed = time.time() - start_time
                    print(
                        f"Preload progress: {files_done}/{len(files_to_load)} files, "
                        f"{bytes_done / 1e9:.2f}/{total_size / 1e9:.2f} GB, "
                        f"{elapsed:.2f}s elapsed"
                    )

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            list(executor.map(_read_file, files_to_load))

        elapsed = time.time() - start_time
        print(
            f"Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n"
        )

    def _start_server(self) -> subprocess.Popen:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--seed",
            str(self.cfg.seed),
            "--model",
            self.cfg.model_path,
            "--served-model-name",
            self.cfg.served_model_name,
            "--tensor-parallel-size",
            "1",
            "--max-num-seqs",
            str(self.cfg.batch_size),
            "--gpu-memory-utilization",
            str(self.cfg.gpu_memory_utilization),
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--dtype",
            self.cfg.dtype,
            "--kv-cache-dtype",
            self.cfg.kv_cache_dtype,
            "--max-model-len",
            str(self.cfg.context_tokens),
            "--stream-interval",
            str(self.cfg.stream_interval),
            "--async-scheduling",
            "--disable-log-stats",
            "--enable-prefix-caching",
            "--enable-lora",
            "--max-loras",
            "1",
            "--max-cpu-loras",
            "1",
            "--max-lora-rank",
            "64",
            "--lora-modules",
            f"{self.cfg.score_model_name}={os.environ.get('AIMO_ADAPTER_PATH', self.cfg.lora_path)}",
        ]

        print(
            "Launching vLLM with "
            f"adapter={os.environ.get('AIMO_ADAPTER_PATH', self.cfg.lora_path)} "
            f"classifier_head={os.environ.get('AIMO_CLASSIFIER_HEAD_PATH', '<unset>')} "
            f"score_model={self.cfg.score_model_name}"
        )
        print(f"vLLM command: {' '.join(cmd)}")

        self.log_file = open("vllm_server.log", "w")
        return subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _wait_for_server(self) -> None:
        print("Waiting for vLLM server...")
        start_time = time.time()

        for _ in range(self.cfg.server_timeout):
            return_code = self.server_process.poll()
            if return_code is not None:
                self.log_file.flush()
                with open("vllm_server.log", "r") as log_file:
                    logs = log_file.read()
                raise RuntimeError(f"Server died with code {return_code}. Full logs:\n{logs}\n")

            try:
                self.client.models.list()
                elapsed = time.time() - start_time
                print(f"Server is ready (took {elapsed:.2f} seconds).\n")
                return
            except Exception:
                time.sleep(1)

        raise RuntimeError("Server failed to start (timeout).\n")

    def _initialize_kernels(self) -> None:
        print(f"Initializing {self.cfg.workers} persistent Jupyter kernels...")
        start_time = time.time()

        self.sandbox_pool = queue.Queue()

        def _create_sandbox():
            return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(_create_sandbox) for _ in range(self.cfg.workers)]
            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())

        elapsed = time.time() - start_time
        print(f"Kernels initialized in {elapsed:.2f} seconds.\n")

    def _scan_for_answer(self, text: str) -> int | None:
        pattern = r"\\boxed\s*\{\s*([0-9,]+)\s*\}"
        matches = re.findall(pattern, text)
        if matches:
            try:
                clean_value = matches[-1].replace(",", "")
                value = int(clean_value)
                if 0 <= value <= 99999:
                    return value
            except ValueError:
                pass

        pattern = r"final\s+answer\s+is\s*([0-9,]+)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                clean_value = matches[-1].replace(",", "")
                value = int(clean_value)
                if 0 <= value <= 99999:
                    return value
            except ValueError:
                pass

        return None

    def _compute_mean_entropy(self, logprobs_buffer: list) -> float:
        if not logprobs_buffer:
            return float("inf")

        total_entropy = 0.0
        token_count = 0

        for top_logprobs_dict in logprobs_buffer:
            if not isinstance(top_logprobs_dict, dict):
                continue
            if not top_logprobs_dict:
                continue

            token_entropy = 0.0
            for _, log_prob in top_logprobs_dict.items():
                prob = math.exp(log_prob)
                if prob > 0:
                    token_entropy -= prob * math.log2(prob)

            total_entropy += token_entropy
            token_count += 1

        if token_count == 0:
            return float("inf")

        return total_entropy / token_count

    def _normalize_logprobs(self, top_logprobs: Any) -> dict[str, float]:
        if isinstance(top_logprobs, dict):
            return {str(key): float(value) for key, value in top_logprobs.items()}
        if isinstance(top_logprobs, list):
            normalized = {}
            for item in top_logprobs:
                token = getattr(item, "token", None)
                logprob = getattr(item, "logprob", None)
                if token is None or logprob is None:
                    continue
                normalized[str(token)] = float(logprob)
            return normalized
        return {}

    def _render_score_prompt_ids(self, state: AttemptState) -> list[int]:
        token_ids = state.render_prompt_ids(self.encoding)
        if len(token_ids) >= 2 and token_ids[-2:] == [200006, 173781]:
            return token_ids[:-2]
        return token_ids

    def _safe_debug_name(self, value: Any) -> str:
        text = str(value or "unknown")
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
        return text[:80] or "unknown"

    def _dump_token_artifact(
        self,
        *,
        kind: str,
        token_ids: list[int],
        question_id: str | None,
        attempt_id: int,
        stage: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.cfg.score_debug_dump:
            return

        with self._score_debug_dump_lock:
            self._score_debug_dump_counter += 1
            artifact_index = self._score_debug_dump_counter

        safe_question_id = self._safe_debug_name(question_id)
        dump_dir = os.path.join(self.cfg.score_debug_dump_dir, safe_question_id)
        os.makedirs(dump_dir, exist_ok=True)
        safe_kind = self._safe_debug_name(kind)
        base_name = (
            f"{self._score_debug_run_id}-"
            f"{artifact_index:04d}-"
            f"a{attempt_id:02d}-"
            f"{safe_kind}"
        )
        text_path = os.path.join(dump_dir, f"{base_name}-text.txt")
        token_path = os.path.join(dump_dir, f"{base_name}-tokens.txt")
        meta_path = os.path.join(dump_dir, f"{base_name}-meta.json")

        try:
            decoded_text = self.encoding.decode(token_ids)
        except Exception as exc:
            decoded_text = f"<failed to decode token ids: {exc}>"

        metadata = {
            "kind": kind,
            "question_id": question_id,
            "attempt_id": attempt_id,
            "stage": stage,
            "token_count": len(token_ids),
            "tail_tokens": token_ids[-32:],
            "endswith_end": token_ids[-1:] == [200007],
            "endswith_call": token_ids[-1:] == [200012],
            "endswith_end_start_assistant": token_ids[-3:] == [200007, 200006, 173781],
            "text_path": text_path,
            "token_path": token_path,
        }
        if extra:
            metadata.update(extra)

        with open(text_path, "w", encoding="utf-8") as file_object:
            file_object.write(decoded_text)
            file_object.write("\n")
        with open(token_path, "w", encoding="utf-8") as file_object:
            for token_id in token_ids:
                file_object.write(f"{int(token_id)}\n")
        with open(meta_path, "w", encoding="utf-8") as file_object:
            json.dump(metadata, file_object, ensure_ascii=False, indent=2)
            file_object.write("\n")

    def _dump_candidate_token_artifacts(
        self,
        *,
        candidates: list[dict[str, Any]],
        token_ids_list: list[list[int]],
        kind: str,
        question_id: str | None,
        stage: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.cfg.score_debug_dump:
            return

        for candidate, token_ids in zip(candidates, token_ids_list):
            candidate_extra = {
                "candidate_attempt_id": candidate.get("attempt_id"),
                "candidate_score": candidate.get("score"),
            }
            if extra:
                candidate_extra.update(extra)
            self._dump_token_artifact(
                kind=kind,
                token_ids=token_ids,
                question_id=question_id,
                attempt_id=int(candidate["attempt_id"]),
                stage=stage,
                extra=candidate_extra,
            )

    def _build_vote_summary(self, detailed_results: list) -> list[dict[str, Any]]:
        answer_weights = defaultdict(float)
        answer_votes = defaultdict(int)

        for result in detailed_results:
            answer = result["Answer"]
            entropy = result["Entropy"]
            if answer is not None:
                weight = 1.0 / max(entropy, 1e-9)
                answer_weights[answer] += weight
                answer_votes[answer] += 1

        scored_answers = []
        for answer, total_weight in answer_weights.items():
            scored_answers.append(
                {
                    "answer": answer,
                    "votes": answer_votes[answer],
                    "score": total_weight,
                }
            )

        scored_answers.sort(key=lambda x: x["score"], reverse=True)
        return scored_answers

    def _dump_question_summary(
        self,
        *,
        question_id: str | None,
        problem: str,
        detailed_results: list[dict[str, Any]],
        vote_summary: list[dict[str, Any]],
        final_answer: int,
        elapsed: float,
        budget: float,
    ) -> None:
        if not self.cfg.result_debug_dump:
            return

        safe_question_id = self._safe_debug_name(question_id)
        dump_dir = os.path.join(self.cfg.score_debug_dump_dir, safe_question_id)
        os.makedirs(dump_dir, exist_ok=True)

        with self._score_debug_dump_lock:
            self._score_debug_dump_counter += 1
            artifact_index = self._score_debug_dump_counter

        base_name = (
            f"{self._score_debug_run_id}-"
            f"{artifact_index:04d}-"
            "question_summary"
        )
        summary_path = os.path.join(dump_dir, f"{base_name}.json")
        results_csv_path = os.path.join(dump_dir, f"{base_name}-attempts.csv")
        votes_csv_path = os.path.join(dump_dir, f"{base_name}-votes.csv")

        summary = {
            "question_id": question_id,
            "problem": problem,
            "final_answer": final_answer,
            "elapsed": elapsed,
            "budget": budget,
            "attempt_count": len(detailed_results),
            "valid_answer_count": sum(
                1 for result in detailed_results if result.get("Answer") is not None
            ),
            "vote_summary": vote_summary,
            "results_csv_path": results_csv_path,
            "votes_csv_path": votes_csv_path,
        }

        with open(summary_path, "w", encoding="utf-8") as file_object:
            json.dump(summary, file_object, ensure_ascii=False, indent=2)
            file_object.write("\n")

        results_dataframe = pd.DataFrame(detailed_results)
        if not results_dataframe.empty and "Entropy" in results_dataframe.columns:
            results_dataframe["Entropy"] = results_dataframe["Entropy"].round(6)
        if not results_dataframe.empty and "Answer" in results_dataframe.columns:
            results_dataframe["Answer"] = results_dataframe["Answer"].astype("Int64")
        results_dataframe.to_csv(results_csv_path, index=False)

        votes_dataframe = pd.DataFrame(vote_summary)
        if not votes_dataframe.empty and "score" in votes_dataframe.columns:
            votes_dataframe["score"] = votes_dataframe["score"].round(6)
        votes_dataframe.to_csv(votes_csv_path, index=False)

        print(
            "Question summary dump: "
            f"id={question_id} final={final_answer} "
            f"summary={summary_path} attempts={results_csv_path} votes={votes_csv_path}"
        )

    def _create_attempt_state(
        self,
        problem: str,
        system_prompt: str,
        attempt_index: int,
    ) -> AttemptState:
        sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
        local_tool = AIMO3Tool(
            local_jupyter_timeout=self.cfg.jupyter_timeout,
            tool_prompt=self.cfg.tool_prompt,
            sandbox=sandbox,
        )
        messages = self.template.apply_chat_template(
            system_prompt,
            problem,
            local_tool.tool_config,
        )
        conversation = Conversation.from_messages(messages)
        return AttemptState(
            attempt_id=attempt_index + 1,
            seed=int(math.pow(self.cfg.seed + attempt_index, 2)),
            sandbox=sandbox,
            tool=local_tool,
            conversation=conversation,
        )

    def _release_attempt_state(self, state: AttemptState | None) -> None:
        if state is None:
            return
        sandbox = state.sandbox
        if sandbox is None:
            return
        state.sandbox = None
        state.tool = None
        try:
            sandbox.reset()
        finally:
            self.sandbox_pool.put(sandbox)

    def _build_attempt_result(
        self,
        state: AttemptState | None,
        *,
        attempt_id: int,
        stage: str,
        status: str,
        score: float | None = None,
    ) -> dict[str, Any]:
        entropy = self._compute_mean_entropy(state.logprobs) if state is not None else float("inf")
        return {
            "Attempt": attempt_id,
            "Stage": stage,
            "Status": status,
            "Response Length": state.generated_tokens if state is not None else 0,
            "Python Calls": state.python_calls if state is not None else 0,
            "Python Errors": state.python_errors if state is not None else 0,
            "Entropy": entropy,
            "Score": score,
            "Answer": state.answer if state is not None else None,
        }

    def _run_attempt_until(
        self,
        state: AttemptState,
        stop_event: threading.Event,
        deadline: float,
        *,
        stage_token_budget: int | None = None,
    ) -> str:
        for _ in range(self.cfg.turns):
            if stop_event.is_set():
                return "stopped"
            if time.time() > deadline:
                return "deadline"
            if state.answer is not None:
                return "answered"
            if (
                stage_token_budget is not None
                and state.generated_tokens >= stage_token_budget
            ):
                return "prefix_ready"

            prompt_ids = state.render_prompt_ids(self.encoding)
            available_context = self.cfg.context_tokens - len(prompt_ids)
            if available_context < self.cfg.buffer_tokens:
                return "context_exhausted"

            max_tokens = available_context
            if stage_token_budget is not None:
                remaining_stage_tokens = stage_token_budget - state.generated_tokens
                if remaining_stage_tokens <= 0:
                    return "prefix_ready"
                max_tokens = min(max_tokens, remaining_stage_tokens)

            stream = self.client.completions.create(
                model=self.cfg.served_model_name,
                temperature=self.cfg.temperature,
                logprobs=self.cfg.top_logprobs,
                max_tokens=max_tokens,
                prompt=prompt_ids,
                seed=state.seed,
                stream=True,
                extra_body={
                    "min_p": self.cfg.min_p,
                    "stop_token_ids": self.stop_token_ids,
                    "return_token_ids": True,
                },
            )

            try:
                token_buffer: list[int] = []
                text_chunks: list[str] = []
                last_finish_reason: str | None = None
                last_chunk_summary: dict[str, Any] | None = None

                for chunk in stream:
                    if stop_event.is_set():
                        return "stopped"
                    if time.time() > deadline:
                        return "deadline"

                    choice = chunk.choices[0]
                    last_finish_reason = getattr(choice, "finish_reason", None)
                    new_tokens = choice.token_ids or []
                    new_text = choice.text or ""
                    last_chunk_summary = {
                        "finish_reason": last_finish_reason,
                        "has_token_ids": bool(choice.token_ids),
                        "token_ids_len": len(choice.token_ids or []),
                        "has_text": bool(choice.text),
                        "text_len": len(choice.text or ""),
                    }

                    if new_tokens:
                        token_buffer.extend(new_tokens)
                        state.generated_tokens += len(new_tokens)
                        text_chunks.append(new_text)

                        chunk_logprobs = choice.logprobs
                        if chunk_logprobs is not None and chunk_logprobs.top_logprobs:
                            for item in chunk_logprobs.top_logprobs:
                                normalized = self._normalize_logprobs(item)
                                if normalized:
                                    state.logprobs.append(normalized)

                    if "}" in new_text:
                        search_text = "".join(text_chunks[-self.cfg.search_tokens :])
                        answer = self._scan_for_answer(search_text)
                        if answer is not None:
                            state.answer = answer
                            break
            finally:
                stream.close()

            if state.answer is not None:
                return "answered"
            if not token_buffer:
                print(
                    f"Attempt {state.attempt_id} produced no output. "
                    f"Last chunk summary: {last_chunk_summary}"
                )
                return f"no_output:{last_finish_reason or 'unknown'}"

            new_messages = self.encoding.parse_messages_from_completion_tokens(
                token_buffer,
                Role.ASSISTANT,
            )
            if not new_messages:
                return "no_messages"

            state.conversation.messages.extend(new_messages)
            state.completion_turns += 1
            last_message = new_messages[-1]

            if last_message.channel == "final":
                answer_text = last_message.content[0].text
                state.answer = self._scan_for_answer(answer_text)
                return "answered" if state.answer is not None else "final_without_box"

            if last_message.recipient == "python":
                state.python_calls += 1
                tool_responses = state.tool.process_sync_plus(last_message)
                response_text = tool_responses[0].content[0].text

                if (
                    response_text.startswith("[ERROR]")
                    or "Traceback" in response_text
                    or "Error:" in response_text
                ):
                    state.python_errors += 1

                state.conversation.messages.extend(tool_responses)

        if stage_token_budget is not None and state.generated_tokens >= stage_token_budget:
            return "prefix_ready"
        if state.answer is not None:
            return "answered"
        return "turn_limit"

    def _run_prefix_attempt(
        self,
        problem: str,
        system_prompt: str,
        attempt_index: int,
        stop_event: threading.Event,
        deadline: float,
    ) -> dict[str, Any]:
        if stop_event.is_set() or time.time() > deadline:
            return {
                "attempt_id": attempt_index + 1,
                "status": "skipped",
                "state": None,
                "prompt_token_ids": None,
                "score": None,
                "result": self._build_attempt_result(
                    None,
                    attempt_id=attempt_index + 1,
                    stage="prefix",
                    status="skipped",
                ),
            }

        state = None
        try:
            state = self._create_attempt_state(problem, system_prompt, attempt_index)
            status = self._run_attempt_until(
                state,
                stop_event,
                deadline,
                stage_token_budget=self.cfg.prefix_budget,
            )

            if status == "prefix_ready":
                return {
                    "attempt_id": state.attempt_id,
                    "status": status,
                    "state": state,
                    "prompt_token_ids": self._render_score_prompt_ids(state),
                    "score": None,
                    "result": None,
                }

            result = self._build_attempt_result(
                state,
                attempt_id=state.attempt_id,
                stage="prefix",
                status=status,
            )
            self._release_attempt_state(state)
            return {
                "attempt_id": attempt_index + 1,
                "status": status,
                "state": None,
                "prompt_token_ids": None,
                "score": None,
                "result": result,
            }
        except Exception as exc:
            if state is not None:
                state.python_errors += 1
                result = self._build_attempt_result(
                    state,
                    attempt_id=state.attempt_id,
                    stage="prefix",
                    status=f"error: {exc}",
                )
                self._release_attempt_state(state)
            else:
                result = self._build_attempt_result(
                    None,
                    attempt_id=attempt_index + 1,
                    stage="prefix",
                    status=f"error: {exc}",
                )
            return {
                "attempt_id": attempt_index + 1,
                "status": "error",
                "state": None,
                "prompt_token_ids": None,
                "score": None,
                "result": result,
            }

    def _score_prefixes_http(self, prefix_token_ids_list: list[list[int]]) -> list[float]:
        if not prefix_token_ids_list:
            return []
        if self.cfg.assess_special_token_id < 0:
            return [0.0] * len(prefix_token_ids_list)

        assess_suffix = [self.cfg.assess_special_token_id] * self.cfg.num_assess_tokens
        classify_inputs = [token_ids + assess_suffix for token_ids in prefix_token_ids_list]
        payload = {
            "model": self.cfg.score_model_name,
            "input": classify_inputs,
            "add_special_tokens": False,
            "truncate_prompt_tokens": self.cfg.context_tokens,
            "assess_tail_len": len(assess_suffix),
        }
        request_body = json.dumps(payload).encode("utf-8")
        last_error = None
        path_errors: list[str] = []

        for path in ("/classify", "/v1/classify"):
            request = urllib_request.Request(
                f"http://0.0.0.0:{self.port}{path}",
                data=request_body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib_request.urlopen(request, timeout=self.cfg.classify_timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                items = data.get("data", [])
                scores = []
                for item in items:
                    probs = item.get("probs", [])
                    if not probs:
                        scores.append(0.0)
                    elif len(probs) > 1:
                        scores.append(float(probs[1]))
                    else:
                        scores.append(float(probs[0]))
                if len(scores) == len(prefix_token_ids_list):
                    return scores
                last_error = ValueError(
                    f"Expected {len(prefix_token_ids_list)} classify scores, got {len(scores)}"
                )
                path_errors.append(f"{path} -> {last_error}")
            except urllib_error.HTTPError as exc:
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    error_body = "<unavailable>"
                last_error = RuntimeError(
                    f"{path} -> HTTP {exc.code}: {error_body}"
                )
                path_errors.append(str(last_error))
            except (urllib_error.URLError, TimeoutError, ValueError) as exc:
                last_error = exc
                path_errors.append(f"{path} -> {exc}")

        print("Prefix scoring fallback to zeros:")
        for item in path_errors:
            print(f"  {item}")
        return [0.0] * len(prefix_token_ids_list)

    def _run_continuation_attempt(
        self,
        candidate: dict[str, Any],
        stop_event: threading.Event,
        deadline: float,
    ) -> dict[str, Any]:
        state: AttemptState = candidate["state"]
        score = candidate.get("score")
        try:
            status = self._run_attempt_until(
                state,
                stop_event,
                deadline,
                stage_token_budget=None,
            )
            result = self._build_attempt_result(
                state,
                attempt_id=state.attempt_id,
                stage="continuation",
                status=status,
                score=score,
            )
            self._release_attempt_state(state)
            return result
        except Exception as exc:
            state.python_errors += 1
            result = self._build_attempt_result(
                state,
                attempt_id=state.attempt_id,
                stage="continuation",
                status=f"error: {exc}",
                score=score,
            )
            self._release_attempt_state(state)
            return result

    def _select_answer(self, detailed_results: list) -> int:
        scored_answers = self._build_vote_summary(detailed_results)
        vote_data = []
        for item in scored_answers:
            vote_data.append((item["answer"], item["votes"], item["score"]))

        vote_dataframe = pd.DataFrame(vote_data, columns=["Answer", "Votes", "Score"])
        vote_dataframe = vote_dataframe.round({"Score": 3})
        display(vote_dataframe)

        if not scored_answers:
            print("\nFinal Answer: 0\n")
            return 0

        final_answer = scored_answers[0]["answer"]
        print(f"\nFinal Answer: {final_answer}\n")
        return final_answer

    def solve_problem(self, problem: str, question_id: Optional[str] = None) -> int:
        question_started_at = time.time()

        if question_id is not None:
            print(f"\nQuestion ID: {question_id}")
        print(f"Problem: {problem}\n")

        user_input = f"{problem} {self.cfg.preference_prompt}"

        elapsed_global = time.time() - self.notebook_start_time
        time_left = self.cfg.notebook_limit - elapsed_global
        problems_left_others = max(0, self.problems_remaining - 1)
        reserved_time = problems_left_others * self.cfg.base_problem_timeout

        budget = time_left - reserved_time
        budget = min(budget, self.cfg.high_problem_timeout)
        budget = max(budget, self.cfg.base_problem_timeout)

        deadline = time.time() + budget

        print(f"Budget: {budget:.2f} seconds | Deadline: {deadline:.2f}\n")

        detailed_results = []
        valid_answers = []
        stop_event = threading.Event()
        scored_candidates: list[dict[str, Any]] = []

        def register_result(result: dict[str, Any]) -> None:
            detailed_results.append(result)
            if result["Answer"] is not None:
                valid_answers.append(result["Answer"])
                counts = Counter(valid_answers).most_common(1)
                if counts and counts[0][1] >= self.cfg.early_stop:
                    stop_event.set()

        prefix_workers = min(self.cfg.workers, self.cfg.prefix_candidates)
        with ThreadPoolExecutor(max_workers=prefix_workers) as prefix_executor:
            prefix_futures = [
                prefix_executor.submit(
                    self._run_prefix_attempt,
                    user_input,
                    self.cfg.system_prompt,
                    attempt_index,
                    stop_event,
                    deadline,
                )
                for attempt_index in range(self.cfg.prefix_candidates)
            ]

            for future in as_completed(prefix_futures):
                try:
                    outcome = future.result()
                except Exception as exc:
                    print(f"Prefix stage future failed: {exc}")
                    continue

                if outcome["result"] is not None:
                    register_result(outcome["result"])
                    continue

                scored_candidates.append(outcome)

        if not stop_event.is_set() and scored_candidates:
            score_prefix_token_ids_list = [
                candidate["prompt_token_ids"] for candidate in scored_candidates
            ]
            assess_suffix = (
                [self.cfg.assess_special_token_id] * self.cfg.num_assess_tokens
                if self.cfg.assess_special_token_id >= 0
                else []
            )
            classify_input_token_ids_list = [
                token_ids + assess_suffix for token_ids in score_prefix_token_ids_list
            ]
            self._dump_candidate_token_artifacts(
                candidates=scored_candidates,
                token_ids_list=score_prefix_token_ids_list,
                kind="score_prefix",
                question_id=question_id,
                stage="prefix_scoring",
                extra={"assess_suffix": assess_suffix},
            )
            self._dump_candidate_token_artifacts(
                candidates=scored_candidates,
                token_ids_list=classify_input_token_ids_list,
                kind="classify_input",
                question_id=question_id,
                stage="prefix_scoring",
                extra={"assess_suffix": assess_suffix},
            )
            prefix_scores = self._score_prefixes_http(score_prefix_token_ids_list)
            for candidate, score in zip(scored_candidates, prefix_scores):
                candidate["score"] = score

            ranked_candidates = sorted(
                scored_candidates,
                key=lambda item: (item.get("score", 0.0), -item["attempt_id"]),
                reverse=True,
            )
            selected_candidates = ranked_candidates[: self.cfg.top_prefixes]
            selected_attempt_ids = {
                candidate["attempt_id"] for candidate in selected_candidates
            }

            for candidate in scored_candidates:
                if candidate["attempt_id"] in selected_attempt_ids:
                    continue
                dropped_result = self._build_attempt_result(
                    candidate["state"],
                    attempt_id=candidate["attempt_id"],
                    stage="prefix",
                    status="dropped_after_scoring",
                    score=candidate.get("score"),
                )
                self._release_attempt_state(candidate["state"])
                register_result(dropped_result)

            continuation_workers = min(
                self.cfg.workers,
                self.cfg.continuation_batch_size,
                len(selected_candidates),
            )
            continuation_prompt_token_ids_list = [
                candidate["state"].render_prompt_ids(self.encoding)
                for candidate in selected_candidates
            ]
            self._dump_candidate_token_artifacts(
                candidates=selected_candidates,
                token_ids_list=continuation_prompt_token_ids_list,
                kind="continuation_prompt",
                question_id=question_id,
                stage="continuation_start",
                extra={"selected_after_scoring": True},
            )
            with ThreadPoolExecutor(max_workers=max(1, continuation_workers)) as continuation_executor:
                continuation_futures = [
                    continuation_executor.submit(
                        self._run_continuation_attempt,
                        candidate,
                        stop_event,
                        deadline,
                    )
                    for candidate in selected_candidates
                ]

                for future in as_completed(continuation_futures):
                    try:
                        result = future.result()
                        register_result(result)
                    except Exception as exc:
                        print(f"Continuation stage future failed: {exc}")
                        continue
        else:
            for candidate in scored_candidates:
                skipped_result = self._build_attempt_result(
                    candidate["state"],
                    attempt_id=candidate["attempt_id"],
                    stage="prefix",
                    status="skipped_after_early_stop",
                    score=candidate.get("score"),
                )
                self._release_attempt_state(candidate["state"])
                register_result(skipped_result)

        self.problems_remaining = max(0, self.problems_remaining - 1)

        if detailed_results:
            results_dataframe = pd.DataFrame(detailed_results)
            results_dataframe["Entropy"] = results_dataframe["Entropy"].round(3)
            results_dataframe["Answer"] = results_dataframe["Answer"].astype("Int64")
            display(results_dataframe)

        elapsed = time.time() - question_started_at

        if not valid_answers:
            print("\nResult: 0")
            print(f"Elapsed: {elapsed:.2f} seconds\n")
            self._dump_question_summary(
                question_id=question_id,
                problem=problem,
                detailed_results=detailed_results,
                vote_summary=[],
                final_answer=0,
                elapsed=elapsed,
                budget=budget,
            )
            return 0

        final_answer = self._select_answer(detailed_results)
        self._dump_question_summary(
            question_id=question_id,
            problem=problem,
            detailed_results=detailed_results,
            vote_summary=self._build_vote_summary(detailed_results),
            final_answer=final_answer,
            elapsed=elapsed,
            budget=budget,
        )
        print(f"Elapsed: {elapsed:.2f} seconds\n")
        return final_answer

    def __del__(self):
        if hasattr(self, "server_process"):
            self.server_process.terminate()
            self.server_process.wait()

        if hasattr(self, "log_file"):
            self.log_file.close()

        if hasattr(self, "sandbox_pool"):
            while not self.sandbox_pool.empty():
                try:
                    sb = self.sandbox_pool.get_nowait()
                    sb.close()
                except Exception:
                    pass


solver: AIMO3Solver | None = None


def predict(*frames: pl.DataFrame) -> pl.DataFrame:
    if len(frames) == 4:
        _, id_, question, _ = frames
    elif len(frames) == 3:
        id_, question, _ = frames
    elif len(frames) == 2:
        id_, question = frames
    else:
        raise TypeError(f"Unexpected predict() argument count: {len(frames)}")

    id_value = id_.item(0)
    question_text = question.item(0)

    gc.disable()
    final_answer = solver.solve_problem(question_text, question_id=str(id_value))
    gc.enable()
    gc.collect()

    return pl.DataFrame({"id": id_value, "answer": final_answer})


def prepare_local_gateway_reference(reference_csv: str) -> str:
    order_mode = os.getenv("LOCAL_REFERENCE_ORDER", "").strip().lower()
    if order_mode not in {"reverse", "last_first"}:
        return reference_csv

    reference_df = pd.read_csv(reference_csv)
    if reference_df.empty:
        return reference_csv

    reordered_df = reference_df.iloc[::-1].reset_index(drop=True)
    tmp_dir = os.getenv("LOCAL_REFERENCE_TMPDIR", "/tmp")
    file_descriptor, reordered_path = tempfile.mkstemp(
        prefix="aimo_local_gateway_",
        suffix=".csv",
        dir=tmp_dir,
    )
    os.close(file_descriptor)
    reordered_df.to_csv(reordered_path, index=False)
    print(
        "Local gateway reference order override: "
        f"mode={order_mode}, source={reference_csv}, reordered={reordered_path}"
    )
    return reordered_path


def main() -> None:
    global solver

    setup_runtime()
    set_seed(CFG.seed)
    solver = AIMO3Solver(CFG)
    inference_server = kaggle_evaluation.AIMO3InferenceServer(predict)

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    else:
        local_reference_csv = prepare_local_gateway_reference(CFG.local_reference_csv)
        inference_server.run_local_gateway((local_reference_csv,))


if __name__ == "__main__":
    main()
