import argparse
import json
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Harmony prefix records with a vLLM /classify endpoint."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="assess_lora")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--context-tokens", type=int, default=65536)
    parser.add_argument("--num-assess-tokens", type=int, required=True)
    parser.add_argument("--assess-special-token-id", type=int, required=True)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file_object:
        for line_no, line in enumerate(file_object, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc
    return records


def chunk_list(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def request_scores(
    host: str,
    port: int,
    model_name: str,
    classify_inputs: list[list[int]],
    assess_tail_len: int,
    timeout: int,
    context_tokens: int,
) -> list[float]:
    payload = {
        "model": model_name,
        "input": classify_inputs,
        "add_special_tokens": False,
        "truncate_prompt_tokens": context_tokens,
        "assess_tail_len": assess_tail_len,
    }
    request_body = json.dumps(payload).encode("utf-8")
    path_errors: list[str] = []

    for path in ("/classify", "/v1/classify"):
        request = urllib_request.Request(
            f"http://{host}:{port}{path}",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = "<unavailable>"
            path_errors.append(f"{path} -> HTTP {exc.code}: {error_body}")
            continue
        except (urllib_error.URLError, TimeoutError, ValueError) as exc:
            path_errors.append(f"{path} -> {exc}")
            continue

        items = data.get("data", [])
        scores: list[float] = []
        for item in items:
            probs = item.get("probs", [])
            if not probs:
                scores.append(0.0)
            elif len(probs) > 1:
                scores.append(float(probs[1]))
            else:
                scores.append(float(probs[0]))

        if len(scores) != len(classify_inputs):
            path_errors.append(
                f"{path} -> expected {len(classify_inputs)} scores, got {len(scores)}"
            )
            continue
        return scores

    error_text = "\n".join(path_errors) if path_errors else "unknown classify error"
    raise RuntimeError(f"Failed to score prefixes via vLLM classify endpoint:\n{error_text}")


def resolve_target(record: dict[str, Any]) -> float | None:
    for key in ("soft_label", "Soft_label", "label_prob", "good_probability", "score"):
        if key in record and record[key] is not None:
            return float(record[key])
    return None


def resolve_hard_label(record: dict[str, Any]) -> int | None:
    for key in ("Hard_label", "hard_label", "label", "class_label"):
        if key in record and record[key] is not None:
            return int(record[key])
    return None


def main() -> None:
    args = parse_args()

    if args.assess_special_token_id < 0:
        raise ValueError("--assess-special-token-id must be non-negative")
    if args.num_assess_tokens <= 0:
        raise ValueError("--num-assess-tokens must be positive")

    records = load_jsonl(args.input_jsonl)
    assess_suffix = [args.assess_special_token_id] * args.num_assess_tokens

    scored_records: list[dict[str, Any]] = []
    mse_terms: list[float] = []
    hard_matches = 0
    hard_total = 0

    for batch in chunk_list(records, args.batch_size):
        classify_inputs: list[list[int]] = []
        for record in batch:
            prefix_token_ids = record.get("prefix_token_ids")
            if not isinstance(prefix_token_ids, list) or not prefix_token_ids:
                raise ValueError(
                    f"Record missing usable prefix_token_ids: id={record.get('id')}"
                )
            classify_inputs.append([int(token_id) for token_id in prefix_token_ids] + assess_suffix)

        scores = request_scores(
            host=args.host,
            port=args.port,
            model_name=args.model_name,
            classify_inputs=classify_inputs,
            assess_tail_len=len(assess_suffix),
            timeout=args.timeout,
            context_tokens=args.context_tokens,
        )

        for record, score in zip(batch, scores):
            output_record = dict(record)
            output_record["vllm_score"] = float(score)
            target = resolve_target(record)
            if target is not None:
                output_record["vllm_score_error"] = float(score) - target
                mse_terms.append((float(score) - target) ** 2)
            hard_label = resolve_hard_label(record)
            if hard_label is not None:
                predicted = int(float(score) >= 0.5)
                output_record["vllm_predicted_label"] = predicted
                output_record["vllm_predicted_correct"] = predicted == hard_label
                hard_matches += int(predicted == hard_label)
                hard_total += 1
            scored_records.append(output_record)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_object:
        for record in scored_records:
            file_object.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "output_jsonl": str(output_path.resolve()),
        "num_records": len(scored_records),
        "model_name": args.model_name,
        "host": args.host,
        "port": args.port,
        "num_assess_tokens": args.num_assess_tokens,
        "assess_special_token_id": args.assess_special_token_id,
        "mean_score": (
            sum(record["vllm_score"] for record in scored_records) / len(scored_records)
            if scored_records
            else None
        ),
        "mse_to_soft_label": (sum(mse_terms) / len(mse_terms) if mse_terms else None),
        "hard_label_accuracy": (hard_matches / hard_total if hard_total else None),
        "hard_label_eval_count": hard_total,
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file_object:
        json.dump(summary, file_object, ensure_ascii=False, indent=2)

    print(
        f"Saved {len(scored_records)} scored records to {output_path} "
        f"and summary to {summary_path}"
    )


if __name__ == "__main__":
    main()
