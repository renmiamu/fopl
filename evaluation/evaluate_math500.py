from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from .evaluate_gsm8k import (
        MathVerifyBackend,
        boxed_contents,
        check_server,
        math_verify_grade,
        request_seed,
        require_package,
        resolve_path,
        visible_answer_text,
    )
except ImportError:  # Support ``python evaluation/evaluate_math500.py``.
    from evaluate_gsm8k import (  # type: ignore[no-redef]
        MathVerifyBackend,
        boxed_contents,
        check_server,
        math_verify_grade,
        request_seed,
        require_package,
        resolve_path,
        visible_answer_text,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache/huggingface"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation/results/math500.json"
DEFAULT_DATASET = "HuggingFaceH4/MATH-500"
SYSTEM_PROMPT = (
    "You are an expert competition mathematics solver. Solve the problem "
    "carefully and show your reasoning. End your response with the final "
    "answer in the form \\boxed{ANSWER}."
)
FINAL_ANSWER_MARKER = re.compile(
    r"(?:final(?:\s+(?:answer|result))?\b|answer\s*(?:is|:|=)|therefore|thus)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a model served by a vLLM OpenAI-compatible server on "
            "MATH-500 with one generation per problem."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Served model name exposed by the vLLM /v1/models endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL.",
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--hf-endpoint",
        help="Alternative Hugging Face endpoint, e.g. https://hf-mirror.com.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--num-samples",
        type=int,
        choices=(1,),
        default=1,
        help="Number of generations per problem; MATH-500 evaluation uses 1.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-tokens",
        "--max-new-tokens",
        dest="max_tokens",
        type=int,
        default=8192,
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument(
        "--strict-boxed",
        action="store_true",
        help="Reject responses that do not contain a \\boxed answer.",
    )
    parser.add_argument(
        "--grader",
        choices=("auto", "math_verify", "exact"),
        default="auto",
        help=(
            "Answer grader. 'auto' uses math-verify when installed and otherwise "
            "uses normalized exact matching (default: auto)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from an existing output and skip completed problems.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive for pass@k sampling")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.top_k < 0:
        parser.error("--top-k cannot be negative")
    if args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.retries <= 0:
        parser.error("--retries must be positive")
    if args.retry_delay < 0:
        parser.error("--retry-delay cannot be negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def resolve_math_verify_backend(
    requested: str,
) -> tuple[MathVerifyBackend | None, str]:
    if requested == "exact":
        return None, "exact"
    try:
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except ImportError as exc:
        if requested == "math_verify":
            raise SystemExit(
                "--grader math_verify requires the 'math-verify' package. "
                "Install it with: pip install math-verify"
            ) from exc
        print(
            "math-verify is not installed; using normalized exact matching. "
            "This fallback does not recognize algebraically equivalent answers.",
            flush=True,
        )
        return None, "exact"

    return (
        MathVerifyBackend(
            parse=parse,
            verify=verify,
            prediction_config=(LatexExtractionConfig(), ExprExtractionConfig()),
            gold_config=(
                LatexExtractionConfig(boxed_match_priority=0),
                ExprExtractionConfig(),
            ),
        ),
        "math_verify",
    )


def text_field(item: dict[str, Any], *names: str) -> str:
    for name in names:
        value = item.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_benchmark(
    datasets: Any,
    dataset_name: str,
    split: str,
    cache_dir: Path,
    limit: int | None,
) -> list[dict[str, Any]]:
    try:
        dataset = datasets.load_dataset(
            dataset_name,
            split=split,
            cache_dir=str(resolve_path(cache_dir)),
        )
    except Exception as exc:
        raise SystemExit(
            f"Could not load {dataset_name} split {split!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    rows: list[dict[str, Any]] = []
    for index, raw_item in enumerate(dataset):
        item = dict(raw_item)
        problem = text_field(item, "problem", "question", "prompt")
        gold_answer = text_field(item, "answer", "gold_answer")
        raw_id = text_field(item, "unique_id", "id", "problem_id") or str(index)
        if not problem or not gold_answer:
            raise SystemExit(
                f"Invalid MATH-500 row {index}: expected problem and answer fields."
            )
        row: dict[str, Any] = {
            "problem_id": f"math500/{raw_id}",
            "problem": problem,
            "gold_answer": gold_answer,
        }
        for field in ("subject", "level"):
            if item.get(field) is not None:
                row[field] = item[field]
        rows.append(row)

    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(
            f"No rows loaded from {dataset_name} split {split!r}."
        )
    print(
        f"Loaded {len(rows)} MATH-500 problems from {dataset_name} "
        f"split {split!r}."
    )
    return rows


def normalize_exact_answer(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1]
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\(?:,|!|;|:)", "", text)
    text = re.sub(r"\\(?:quad|qquad)\b", "", text)
    text = re.sub(r"\s+", "", text)
    return text.rstrip(".,;:")


def boxed_gold_answer(gold_answer: str) -> str:
    return gold_answer if boxed_contents(gold_answer) else rf"\boxed{{{gold_answer}}}"


def extract_math500_answer(
    response: str,
    strict_boxed: bool,
) -> tuple[str | None, str]:
    visible = visible_answer_text(response)
    boxes = boxed_contents(visible)
    if boxes:
        answer = boxes[-1].strip()
        return (answer or None), ("boxed" if answer else "not_found")

    bare_box = re.findall(r"\\boxed\s+([^\s.,;]+)", visible, re.IGNORECASE)
    if bare_box:
        return bare_box[-1].strip(), "boxed_without_braces"
    if strict_boxed:
        return None, "not_found"

    markers = list(FINAL_ANSWER_MARKER.finditer(visible))
    if markers:
        tail = visible[markers[-1].end() :].strip()
        first_line = tail.splitlines()[0].strip().rstrip(".,;:") if tail else ""
        if first_line:
            return first_line, "final_answer_phrase"
    return None, "not_found"


def grade_completion(
    response: str,
    gold_answer: str,
    strict_boxed: bool,
    math_verify_backend: MathVerifyBackend | None,
) -> dict[str, Any]:
    parsed_answer, parse_method = extract_math500_answer(response, strict_boxed)
    exact_correct = (
        parsed_answer is not None
        and normalize_exact_answer(parsed_answer) == normalize_exact_answer(gold_answer)
    )
    result: dict[str, Any] = {
        "parsed_answer": parsed_answer,
        "parse_method": parse_method,
        "math_verify_answer": None,
        "math_verify_status": "disabled",
        "grading_method": "exact",
        "correct": exact_correct,
    }
    if math_verify_backend is None:
        return result

    visible = visible_answer_text(response)
    if strict_boxed and not re.search(r"\\boxed\b", visible, re.IGNORECASE):
        result["math_verify_status"] = "strict_boxed_missing"
        result["grading_method"] = "strict_boxed"
        result["correct"] = False
        return result

    math_answer, math_correct, status = math_verify_grade(
        visible,
        boxed_gold_answer(gold_answer),
        math_verify_backend,
    )
    result["math_verify_answer"] = math_answer
    result["math_verify_status"] = status
    if math_correct is None:
        result["grading_method"] = "exact_fallback"
    else:
        result["grading_method"] = "math_verify"
        result["correct"] = math_correct
    return result


def build_client(args: argparse.Namespace) -> Any:
    openai = require_package("openai")
    return openai.OpenAI(
        api_key=args.api_key,
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout,
        max_retries=0,
    )


def request_completions(
    client: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["problem"]},
    ]
    extra_body = {
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }
    for attempt in range(1, args.retries + 1):
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=messages,
                n=args.num_samples,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=request_seed(args.seed, row["problem_id"]),
                extra_body=extra_body,
            )
            choices = sorted(completion.choices, key=lambda choice: choice.index)
            responses = [choice.message.content or "" for choice in choices]
            if len(responses) != args.num_samples:
                raise RuntimeError(
                    f"vLLM returned {len(responses)} choices; "
                    f"expected {args.num_samples}."
                )
            return responses
        except Exception as exc:
            if attempt >= args.retries:
                raise RuntimeError(
                    f"{row['problem_id']} failed after {args.retries} attempts: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            delay = args.retry_delay * (2 ** (attempt - 1))
            print(
                f"{row['problem_id']} attempt {attempt}/{args.retries} failed: "
                f"{type(exc).__name__}: {exc}; retrying in {delay:g}s",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def score_problem(
    row: dict[str, Any],
    completions: list[str],
    strict_boxed: bool,
    math_verify_backend: MathVerifyBackend | None,
) -> dict[str, Any]:
    if len(completions) != 1:
        raise ValueError("MATH-500 evaluation requires exactly one completion")
    samples = []
    for sample_index, response in enumerate(completions):
        samples.append(
            {
                "sample_index": sample_index,
                "response": response,
                **grade_completion(
                    response,
                    row["gold_answer"],
                    strict_boxed,
                    math_verify_backend,
                ),
            }
        )
    correct = bool(samples[0]["correct"])
    return {
        **row,
        "samples": samples,
        "correct": correct,
        "pass@1": float(correct),
    }


def benchmark_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "num_problems": 0,
            "accuracy": None,
            "pass@1": None,
            "num_correct": 0,
        }
    num_correct = sum(bool(row["correct"]) for row in results)
    accuracy = num_correct / len(results)
    return {
        "num_problems": len(results),
        "accuracy": accuracy,
        "pass@1": accuracy,
        "num_correct": num_correct,
    }


def refresh_metrics(payload: dict[str, Any]) -> None:
    results = payload["benchmarks"]["math500"]["results"]
    metrics = benchmark_metrics(results)
    payload["benchmarks"]["math500"]["metrics"] = metrics
    payload["overall"] = dict(metrics)


def regrade_payload(
    payload: dict[str, Any],
    strict_boxed: bool,
    math_verify_backend: MathVerifyBackend | None,
) -> int:
    results = payload.get("benchmarks", {}).get("math500", {}).get("results", [])
    regraded = 0
    for row in results:
        gold_answer = row.get("gold_answer")
        samples = row.get("samples")
        if not isinstance(gold_answer, str) or not isinstance(samples, list):
            raise SystemExit(
                "Cannot regrade resume output: a MATH-500 result is missing "
                "gold_answer or samples."
            )
        for sample in samples:
            response = sample.get("response") if isinstance(sample, dict) else None
            if not isinstance(response, str):
                raise SystemExit(
                    "Cannot regrade resume output: a sample is missing its response."
                )
            sample.update(
                grade_completion(
                    response,
                    gold_answer,
                    strict_boxed,
                    math_verify_backend,
                )
            )
            regraded += 1
        if len(samples) != 1:
            raise SystemExit(
                "Cannot resume this output: MATH-500 now requires exactly one "
                "sample per problem. Use a new output file for the single-sample run."
            )
        correct = bool(samples[0]["correct"])
        row.pop("num_correct", None)
        row.pop("pass@4", None)
        row["correct"] = correct
        row["pass@1"] = float(correct)
    refresh_metrics(payload)
    return regraded


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def new_payload(args: argparse.Namespace, grading_backend: str) -> dict[str, Any]:
    return {
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "num_samples": args.num_samples,
        "strict_boxed": args.strict_boxed,
        "split": args.split,
        "grading": {
            "backend": grading_backend,
            "math_verify_fallback": (
                "exact" if grading_backend == "math_verify" else None
            ),
        },
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
        },
        "benchmarks": {
            "math500": {
                "dataset": args.dataset,
                "results": [],
                "metrics": {},
            }
        },
        "overall": {},
    }


def load_or_create_payload(
    args: argparse.Namespace,
    grading_backend: str,
    math_verify_backend: MathVerifyBackend | None,
) -> dict[str, Any]:
    output_path = resolve_path(args.output)
    if not args.resume or not output_path.exists():
        return new_payload(args, grading_backend)
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not resume from {output_path}: {exc}") from exc

    expected = new_payload(args, grading_backend)
    for field in ("model", "base_url", "num_samples", "strict_boxed", "split"):
        if payload.get(field) != expected[field]:
            raise SystemExit(
                f"Resume output has {field}={payload.get(field)!r}, "
                f"but this run uses {expected[field]!r}."
            )
    if payload.get("generation") != expected["generation"]:
        raise SystemExit("Resume output uses different generation settings.")
    actual_dataset = (
        payload.get("benchmarks", {}).get("math500", {}).get("dataset")
    )
    if actual_dataset != args.dataset:
        raise SystemExit(
            f"Resume output uses dataset={actual_dataset!r}, "
            f"but this run uses {args.dataset!r}."
        )
    if payload.get("grading") != expected["grading"]:
        regraded = regrade_payload(payload, args.strict_boxed, math_verify_backend)
        payload["grading"] = expected["grading"]
        write_output(args.output, payload)
        print(
            f"Regraded {regraded} cached MATH-500 samples with "
            f"{grading_backend} grading.",
            flush=True,
        )
    return payload


def print_metrics(metrics: dict[str, Any]) -> None:
    if not metrics.get("num_problems"):
        return
    print(
        f"math500: problems={metrics['num_problems']}, "
        f"correct={metrics['num_correct']}, "
        f"accuracy={metrics['accuracy']:.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")

    math_verify_backend, grading_backend = resolve_math_verify_backend(args.grader)
    print(f"Using MATH-500 grading backend: {grading_backend}.")
    datasets = require_package("datasets")
    client = build_client(args)
    check_server(client, args.model)
    rows = load_benchmark(
        datasets,
        args.dataset,
        args.split,
        args.cache_dir,
        args.limit,
    )
    payload = load_or_create_payload(args, grading_backend, math_verify_backend)

    completed_ids = {
        row["problem_id"]
        for row in payload["benchmarks"]["math500"]["results"]
    }
    pending = [row for row in rows if row["problem_id"] not in completed_ids]
    completed_now = len(rows) - len(pending)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(request_completions, client, row, args): row
            for row in pending
        }
        try:
            for future in as_completed(futures):
                row = futures[future]
                completions = future.result()
                result = score_problem(
                    row,
                    completions,
                    args.strict_boxed,
                    math_verify_backend,
                )
                payload["benchmarks"]["math500"]["results"].append(result)
                completed_now += 1
                refresh_metrics(payload)
                write_output(args.output, payload)
                metrics = payload["benchmarks"]["math500"]["metrics"]
                print(
                    f"[{completed_now}/{len(rows)}] {row['problem_id']}: "
                    f"correct={result['correct']} | "
                    f"running accuracy={metrics['accuracy']:.4f}",
                    flush=True,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    refresh_metrics(payload)
    write_output(args.output, payload)
    print_metrics(payload["benchmarks"]["math500"]["metrics"])
    print(f"Saved detailed results to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
