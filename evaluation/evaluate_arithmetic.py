"""Shared zero-shot SVAMP/MultiArith evaluator for OpenAI-compatible servers.

--split all evaluates the complete challenge pool, INCLUDING the HF train split.
Use --split test if your model was trained on the train split. No examples or
reference equations/answers are ever included in the model prompt.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

try:
    from .evaluate_gsm8k import check_server, parse_numeric_value, request_seed, require_package
    from .evaluate_multiple_choice import (
        aggregate, boxed_contents, message_response_text,
        normalize_answer_presentation, visible_answer_text,
    )
except ImportError:
    from evaluate_gsm8k import check_server, parse_numeric_value, request_seed, require_package
    from evaluate_multiple_choice import (
        aggregate, boxed_contents, message_response_text,
        normalize_answer_presentation, visible_answer_text,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARSER_VERSION = 1
PROMPT_VERSION = 1
SYSTEM_PROMPT = r"Solve the math problem. Put the final numeric answer inside \boxed{}."
NUMBER = r"[+\-]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?"
FINAL_MARKER = re.compile(
    r"(?:\b(?:final\s+(?:answer|result)|(?:the\s+)?answer)\s*"
    r"(?:is\b|equals\b|[:=])|(?:最终答案|最终结果|答案)\s*(?:是|为|[:：=])|^\s*####)\s*",
    re.I | re.M,
)


@dataclass(frozen=True)
class ArithmeticSpec:
    key: str
    display_name: str
    dataset: str


def answer_presentation(text: str) -> str:
    # Protect arithmetic exponentiation from Markdown normalization, which
    # otherwise turns '10**2' into the completely different number '102'.
    text = re.sub(r"(?<=[\d)])\s*\*\*\s*(?=[+\-\d(])", "^", text)
    return normalize_answer_presentation(text)


def numeric_candidate(text: str) -> str | None:
    """Parse one answer, not an arbitrary number somewhere inside prose."""
    if not text or len(text) > 512:
        return None
    text = answer_presentation(visible_answer_text(text))
    text = text.translate(str.maketrans({"−": "-", "–": "-", "﹣": "-"}))
    text = text.replace(r"\,", "").replace(r"\!", "").replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\$", "$").strip().strip("$`\"'").strip()
    text = re.sub(r"^[€£¥]\s*", "", text)
    # Treat punctuation as presentation, not part of a numeric expression.
    text = re.sub(r"[。.!；;]\s*$", "", text).strip()
    markers = list(FINAL_MARKER.finditer(text))
    if markers and not text[:markers[-1].start()].strip():
        text = text[markers[-1].end():].strip()
    # An assignment or evaluated arithmetic chain can end in an explicit value.
    if "=" in text:
        left, right = text.rsplit("=", 1)
        if not re.fullmatch(r"[\w\s+\-*/().,\\{}^×·=]+", left) or re.search(r"[A-Za-z]{2,}", left):
            return None
        text = right.strip()
    # Reject malformed commas (e.g. '1,2') and resource-exhausting exponents.
    for token in re.findall(r"\d[\d,]*", text):
        if "," in token and not re.fullmatch(r"\d{1,3}(?:,\d{3})+", token):
            return None
    if any(abs(int(value)) > 1000 for value in re.findall(r"[eE]([+\-]?\d+)", text)):
        return None
    scientific = re.fullmatch(rf"{NUMBER}\s*(?:\\times|\\cdot|×|·)\s*10\s*\^\s*(?:\{{\s*([+\-]?\d+)\s*\}}|([+\-]?\d+))", text)
    if scientific:
        exponent = int(scientific[1] or scientific[2])
        return parse_numeric_value(text) if abs(exponent) <= 1000 else None
    if "^" in text or "**" in text:
        # Only small arithmetic powers are useful on these elementary tasks.
        if not re.fullmatch(r"[\d\s.eE+\-*/()^]+", text):
            return None
        if any(abs(int(value)) > 100 for value in re.findall(r"(?:\^|\*\*)\s*([+\-]?\d+)", text)):
            return None
    direct = parse_numeric_value(text)
    if direct is not None:
        return direct
    # One scalar followed by units is unambiguous; alternatives/explanations are
    # not. Never reduce an invalid fraction to its numerator or denominator.
    match = re.fullmatch(r"(.+?)\s+([A-Za-z][A-Za-z\s-]{0,60}|[\u4e00-\u9fff]{1,8})", text)
    if match and not re.search(r"\b(?:or|and|because|maybe|approximately|if|not|but|is|are|was|were|equals|cannot|unknown|undetermined)\b", match[2], re.I):
        return numeric_candidate(match[1])
    mixed = re.fullmatch(r"([+\-]?\d+)\s+(\d+)\s*/\s*(\d+)", text)
    if mixed and int(mixed[3]) != 0:
        sign = "-" if mixed[1].startswith("-") else ""
        return parse_numeric_value(f"{sign}({mixed[1].lstrip('+-')}+{mixed[2]}/{mixed[3]})")
    return None


def extract_answer(response: str, strict_boxed: bool = False) -> tuple[str | None, str]:
    text = visible_answer_text(response)
    commands = list(re.finditer(r"(?<!\w)(?:\\+(?:boxed|fbox)\b|(?:boxed|fbox)\s*(?=[{\[(]))", text, re.I))
    if commands:
        # The last explicit box is authoritative, even when invalid. Falling
        # back to an earlier box can silently score a rejected answer as correct.
        tail = text[commands[-1].start():]
        if not tail.startswith("\\"):
            tail = "\\" + tail
        contents = boxed_contents(tail)
        if contents:
            answer = numeric_candidate(contents[-1])
            return answer, "boxed" if answer is not None else "invalid_boxed"
        alternative = re.fullmatch(r"\\+(?:boxed|fbox)\s*[\[(](.*?)[\])]\s*[.!。]?", tail, re.I | re.S)
        if alternative:
            answer = numeric_candidate(alternative[1])
            return answer, "alternative_box" if answer is not None else "invalid_boxed"
        # A single unclosed outer brace may result from generation truncation.
        partial = re.fullmatch(r"\\+(?:boxed|fbox)\s*\{([^{}\n]{1,512})", tail, re.I)
        if partial:
            answer = numeric_candidate(partial[1])
            return answer, "truncated_box" if answer is not None else "invalid_boxed"
        bare = re.fullmatch(r"\\+(?:boxed|fbox)\s+([^{}\n]{1,512})", tail, re.I)
        if bare:
            answer = numeric_candidate(bare[1])
            return answer, "bare_box" if answer is not None else "invalid_boxed"
        return None, "invalid_boxed"
    if strict_boxed:
        return None, "unparsed"
    formatted = answer_presentation(text)
    markers = list(FINAL_MARKER.finditer(formatted))
    if markers:
        tail = formatted[markers[-1].end():].strip()
        # A labelled answer is usually on the same line as the marker.
        answer = numeric_candidate(tail.splitlines()[0] if tail else "")
        return answer, "final_marker" if answer is not None else "invalid_final_marker"
    fence = re.search(r"```(?:json|text|math|latex)?\s*\n?(.*?)\s*```\s*$", text[-4096:], re.I | re.S)
    tail = fence[1].strip() if fence else text[-4096:].strip()
    for candidate in (tail, tail.splitlines()[-1] if tail else ""):
        if candidate.startswith("{"):
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and set(data) <= {"answer", "final_answer"} and len(data) == 1:
                    answer = numeric_candidate(str(next(iter(data.values()))))
                    return answer, "json" if answer is not None else "invalid_json"
            except (ValueError, TypeError):
                pass
    xml = re.search(r"<(answer|final_answer)>\s*(.*?)\s*</\1>\s*$", tail, re.I | re.S)
    if xml:
        answer = numeric_candidate(xml[2])
        return answer, "xml" if answer is not None else "invalid_xml"
    answer = numeric_candidate(tail.splitlines()[-1] if tail else "")
    return answer, "final_line" if answer is not None else "unparsed"


def grade(response: str, row: dict[str, Any], strict_boxed: bool = False) -> dict[str, Any]:
    answer, method = extract_answer(response, strict_boxed)
    return {"parsed_answer": answer, "parse_method": method,
            "correct": answer is not None and Decimal(answer) == Decimal(row["gold_answer"])}


def normalize_row(raw: dict[str, Any], spec: ArithmeticSpec, split: str, index: int) -> dict[str, Any]:
    if spec.key == "svamp":
        body, question = raw.get("Body"), raw.get("Question")
        problem = " ".join(str(part).strip() for part in (body, question) if part is not None and str(part).strip())
        if not problem:
            problem = raw.get("question_concat", "")
        gold = raw.get("Answer")
        identifier = raw.get("ID")
    else:
        problem = raw.get("question", raw.get("sQuestion", ""))
        gold = raw.get("final_ans", raw.get("lSolutions"))
        identifier = raw.get("id", raw.get("iIndex"))
    if isinstance(gold, list) and len(gold) == 1:
        gold = gold[0]
    if isinstance(gold, (bool, list, dict)) or gold is None:
        raise ValueError(f"{split} row {index}: expected one numeric reference answer")
    answer = numeric_candidate(str(gold))
    if not isinstance(problem, str) or not problem.strip() or answer is None:
        raise ValueError(f"{split} row {index}: missing problem or invalid reference answer")
    return {"problem_id": f"{spec.key}:{split}:{identifier if identifier is not None else index}",
            "problem": problem.strip(), "gold_answer": answer, "source_split": split, "order": index}


def load_rows(args: argparse.Namespace, spec: ArithmeticSpec) -> list[dict[str, Any]]:
    if args.data_file:
        path = args.data_file.expanduser()
        if path.suffix == ".jsonl":
            raw_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            raw_rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_rows, list):
            raise ValueError("--data-file must contain a JSON array or JSONL rows")
        sources = [("local", raw_rows)]
    else:
        datasets = require_package("datasets")
        options: dict[str, Any] = {"cache_dir": str(args.cache_dir.expanduser())}
        if args.revision:
            options["revision"] = args.revision
        if args.split == "all":
            data = datasets.load_dataset(args.dataset, args.dataset_config, **options)
            # A deterministic order also makes seeds/IDs and limits reproducible.
            sources = [(name, data[name]) for name in sorted(data)]
        else:
            sources = [(args.split, datasets.load_dataset(args.dataset, args.dataset_config, split=args.split, **options))]
    rows = []
    for split, source in sources:
        for index, raw in enumerate(source):
            if not isinstance(raw, dict):
                raise ValueError(f"{split} row {index}: expected an object")
            row = normalize_row(raw, spec, split, index)
            row["order"] = len(rows)
            rows.append(row)
    ids = [row["problem_id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Dataset is empty or contains duplicate problem IDs")
    return rows[:args.limit] if args.limit is not None else rows


def build_prompt(row: dict[str, Any]) -> str:
    return row["problem"] + "\n\n" + r"Solve the problem. Put the final numeric answer inside \boxed{}."


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def run_identity(args: argparse.Namespace, spec: ArithmeticSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"benchmark": spec.key, "model": args.model, "base_url": args.base_url.rstrip("/"),
            "dataset": args.dataset, "dataset_config": args.dataset_config, "revision": args.revision,
            "split": "local" if args.data_file else args.split, "dataset_sha256": digest(rows),
            "prompt_version": PROMPT_VERSION, "system_prompt": SYSTEM_PROMPT,
            "prompt_sha256": digest([build_prompt(row) for row in rows]),
            "num_samples": 1, "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty, "max_tokens": args.max_tokens,
            "seed": args.seed, "strict_boxed": args.strict_boxed}


def refresh_metrics(payload: dict[str, Any], spec: ArithmeticSpec) -> None:
    benchmark = payload["benchmarks"][spec.key]
    benchmark["results"].sort(key=lambda row: row["order"])
    benchmark["metrics"] = aggregate(benchmark["results"])
    payload["overall"] = dict(benchmark["metrics"])
    payload["parser_version"] = PARSER_VERSION


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_payload(args: argparse.Namespace, spec: ArithmeticSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    identity = run_identity(args, spec, rows)
    if args.output.exists():
        if not args.resume:
            raise ValueError("Output already exists; use --resume or choose a new --output (will not overwrite it)")
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload.get("run_fingerprint") != digest(identity):
            raise ValueError("Resume settings, prompt, model, or dataset changed; choose a new --output")
        originals = {row["problem_id"]: row for row in rows}
        completed = set()
        for result in payload["benchmarks"][spec.key]["results"]:
            identifier = result["problem_id"]
            if identifier in completed or identifier not in originals:
                raise ValueError("Resume output contains duplicate or unexpected problem IDs")
            if not isinstance(result.get("response"), str) or not result["response"].strip():
                raise ValueError("Resume output contains an empty/invalid response")
            for key, value in originals[identifier].items():
                if result.get(key) != value:
                    raise ValueError(f"Resume row {identifier} does not match the dataset")
            # Regrade saved responses with the current parser, without regenerating.
            result.update(grade(result["response"], result, args.strict_boxed))
            completed.add(identifier)
    else:
        payload = {"model": args.model, "base_url": args.base_url, "prompt_version": PROMPT_VERSION,
                   "split": identity["split"], "num_samples": 1,
                   "prompting": {"strategy": "zero_shot_solve"},
                   "generation": {key: identity[key] for key in
                       ("max_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "seed")},
                   "run_config": identity, "run_fingerprint": digest(identity),
                   "benchmarks": {spec.key: {"dataset": args.dataset,
                       "display_name": spec.display_name, "dataset_config": args.dataset_config,
                       "split": identity["split"], "num_total": len(rows), "results": []}}}
    refresh_metrics(payload, spec)
    return payload


def request_completion(client: Any, args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    error: Exception | None = None
    extra_body = {"repetition_penalty": args.repetition_penalty}
    if args.top_k > 0:
        extra_body["top_k"] = args.top_k
    for attempt in range(args.retries + 1):
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": build_prompt(row)}],
                n=1, temperature=args.temperature, top_p=args.top_p,
                max_tokens=args.max_tokens, seed=request_seed(args.seed, row["problem_id"]),
                extra_body=extra_body,
            )
            choice = completion.choices[0]
            response = message_response_text(choice.message)
            return {**row, "prompt_sha256": digest(build_prompt(row)), "response": response,
                    "finish_reason": choice.finish_reason, **grade(response, row, args.strict_boxed)}
        except Exception as exc:
            error = exc
            if attempt < args.retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"{row['problem_id']}: request failed after {args.retries + 1} attempts: {error}") from error


def parse_args(spec: ArithmeticSpec) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Zero-shot {spec.display_name}, one completion per problem")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen2.5-1.5b-full-sft_1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--dataset", default=spec.dataset)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--split", default="all", help="all merges ALL HF splits (including train); or use test")
    parser.add_argument("--data-file", type=Path, help="Local JSON array / JSONL; bypasses HF loading")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".cache/huggingface")
    parser.add_argument("--hf-endpoint", help="Override HF_ENDPOINT before importing datasets")
    parser.add_argument("--download-only", action="store_true", help="Validate/cache the dataset without calling vLLM")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--num-samples", type=int, choices=[1], default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20, help="Same default as existing evaluators; 0 omits top_k")
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--strict-boxed", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / f"evaluation/results/{spec.key}.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for key in ("temperature", "top_p", "repetition_penalty", "request_timeout"):
        if not math.isfinite(getattr(args, key)):
            parser.error(f"--{key.replace('_', '-')} must be finite")
    for key in ("workers", "max_tokens", "request_timeout", "repetition_penalty"):
        if not getattr(args, key) > 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.retries < 0 or args.top_k < 0 or not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1:
        parser.error("Invalid retries, temperature, or top-p")
    args.output = args.output.expanduser().resolve()
    return args


def run(spec: ArithmeticSpec) -> None:
    args = parse_args(spec)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
    try:
        rows = load_rows(args, spec)
        print(f"Loaded {len(rows)} {spec.display_name} rows (split={'local' if args.data_file else args.split}), zero-shot, one sample.", flush=True)
        if args.download_only:
            return
        payload = load_payload(args, spec, rows)
        write_output(args.output, payload)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise SystemExit(f"Cannot prepare {spec.display_name}: {exc}") from exc
    results = payload["benchmarks"][spec.key]["results"]
    completed = {result["problem_id"] for result in results}
    pending = [row for row in rows if row["problem_id"] not in completed]
    failures = []
    if pending:
        openai = require_package("openai")
        with openai.OpenAI(base_url=args.base_url, api_key=args.api_key,
                           timeout=args.request_timeout, max_retries=0) as client:
            check_server(client, args.model)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(request_completion, client, args, row): row for row in pending}
                try:
                    for future in as_completed(futures):
                        try:
                            results.append(future.result())
                        except Exception as exc:
                            failures.append(str(exc))
                            print(f"FAILED: {exc}", flush=True)
                        refresh_metrics(payload, spec)
                        # Atomic per-completion snapshots: no torn JSONL tail on SIGTERM.
                        write_output(args.output, payload)
                        if results and (len(results) % 25 == 0 or len(results) == len(rows)):
                            metrics = payload["overall"]
                            print(f"{spec.key}: {len(results)}/{len(rows)}, accuracy={metrics['accuracy']:.4f}, parse_rate={metrics['parse_rate']:.4f}", flush=True)
                except KeyboardInterrupt:
                    for future in futures:
                        future.cancel()
                    write_output(args.output, payload)
                    print("Interrupted; saved completed responses. Use --resume.", flush=True)
                    raise
    metrics = payload["overall"]
    print(f"{spec.key}: {len(results)}/{len(rows)}, accuracy={metrics['accuracy'] or 0:.4f}, parse_rate={metrics['parse_rate'] or 0:.4f}")
    print(f"Saved detailed results to {args.output}")
    if failures:
        raise SystemExit(f"{len(failures)} requests failed; saved successful rows. Use --resume to retry missing rows.")
