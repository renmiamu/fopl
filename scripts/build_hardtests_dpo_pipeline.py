from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cot_prompt import build_cot_reconstruction_prompt


DEFAULT_DATASET = "sigcp/hardtests_problems"
DEFAULT_PROFILES = (
    "deepseek_v4_pro",
    "glm_5_2",
    "kimi_k2p5",
    "qwen3_235b_a22b_thinking_2507",
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model_configs.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "dpo_pipeline" / "hardtests"
DEFAULT_HF_CACHE = PROJECT_ROOT / ".cache" / "huggingface"

TITLE_FIELDS = ("title", "problem_title", "question_title", "name", "problem_name")
CONTENT_FIELDS = (
    "question_content",
    "problem",
    "problem_statement",
    "statement",
    "question",
    "prompt",
    "description",
    "text",
)
STARTER_CODE_FIELDS = ("starter_code", "starter", "template", "boilerplate")
ID_FIELDS = ("problem_id", "id", "question_id", "slug", "name")

SOLVE_SYSTEM_PROMPT = """You are an expert competitive programming assistant.
Solve the problem carefully and provide both a detailed reasoning trace and a correct,
efficient final solution. If reasoning is not returned in a separate API field, format
the response as <think>reasoning</think> followed by the final answer.
"""

LOGGER = logging.getLogger("hardtests_dpo_pipeline")


class FatalAPIError(RuntimeError):
    """An API failure that should stop the current stage instead of being retried."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build HardTests DPO data through answer sampling, common-problem "
            "filtering, cross-model CoT reconstruction, and preference pairing."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("all", "solve", "filter", "reconstruct", "build"),
        default="all",
        help="Pipeline stage to run (default: all).",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", help="Hugging Face split; train is preferred.")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--problems-file",
        type=Path,
        help=(
            "Optional local JSON/JSONL source. Useful for an offline run or for "
            "pinning an externally selected problem set."
        ),
    )
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--hf-endpoint")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(DEFAULT_PROFILES),
        help="Model profile names used as both target and reconstruction models.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        help="Final training JSON path (default: OUTPUT_DIR/hardtests_dpo.json).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--solve-max-tokens",
        type=int,
        help="Override every profile's max token setting for the solve stage.",
    )
    parser.add_argument(
        "--reconstruction-max-tokens",
        type=int,
        default=32768,
    )
    parser.add_argument(
        "--skip-identical-pairs",
        action="store_true",
        help="Drop DPO rows whose chosen and rejected CoTs are exactly equal.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create/load the problem snapshot and print the work plan without API calls.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.offset < 0:
        parser.error("--offset cannot be negative")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.retries <= 0:
        parser.error("--retries must be positive")
    if args.solve_max_tokens is not None and args.solve_max_tokens <= 0:
        parser.error("--solve-max-tokens must be positive")
    if args.reconstruction_max_tokens <= 0:
        parser.error("--reconstruction-max-tokens must be positive")
    if len(args.profiles) < 2:
        parser.error("--profiles requires at least two model profiles")
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("--profiles contains duplicate names")
    return args


def setup_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def require_package(package: str) -> Any:
    try:
        return __import__(package, fromlist=["*"])
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency '{package}'. Install it with: pip install {package}"
        ) from exc


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    with path.open("ab+") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
        handle.seek(0, os.SEEK_END)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOGGER.warning(
                        "Skipping malformed checkpoint row %s:%d: %s",
                        path,
                        line_number,
                        exc,
                    )
                    continue
                if not isinstance(item, dict):
                    raise ValueError(f"line {line_number} is not an object")
                rows.append(item)
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"Could not read JSONL file {path}: {exc}") from exc
    return rows


def records_by_id(path: Path, required_field: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for item in read_jsonl(path):
        pid = str(item.get("problem_id") or "").strip()
        if pid and str(item.get(required_field) or "").strip():
            records[pid] = item
    return records


def first_text(item: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def multilingual_content(item: dict[str, Any]) -> str:
    contents = item.get("question_content_all_languages")
    if not isinstance(contents, list):
        return ""
    fallback = ""
    for candidate in contents:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        fallback = fallback or content.strip()
        language = str(candidate.get("language") or candidate.get("lang") or "").lower()
        if language in {"english", "en", "eng"}:
            return content.strip()
    return fallback


def source_problem_id(item: dict[str, Any], index: int) -> str:
    for field in ID_FIELDS:
        value = item.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"row_{index}"


def normalize_problem(
    item: dict[str, Any], index: int, dataset: str, split: str
) -> dict[str, Any] | None:
    title = first_text(item, TITLE_FIELDS)
    statement = multilingual_content(item) or first_text(item, CONTENT_FIELDS)
    if not statement:
        string_values = [
            value.strip() for value in item.values() if isinstance(value, str) and value.strip()
        ]
        statement = max(string_values, key=len, default="")
    if not statement:
        return None
    starter = first_text(item, STARTER_CODE_FIELDS)
    difficulty = str(item.get("difficulty") or "unknown")
    parts = []
    if title:
        parts.append(f"Problem title: {title}")
    parts.extend((f"Difficulty: {difficulty}", "", "Problem statement:", statement))
    if starter:
        parts.extend(("", "Starter code:", f"```python\n{starter}\n```"))
    return {
        "problem_id": source_problem_id(item, index),
        "title": title,
        "problem": "\n".join(parts),
        "statement": statement,
        "starter_code": starter,
        "difficulty": difficulty,
        "source": dataset,
        "split": split,
        "source_index": index,
    }


def choose_split(dataset: Any, requested_split: str | None) -> tuple[str, Any]:
    if not hasattr(dataset, "keys"):
        return requested_split or "default", dataset
    if requested_split:
        if requested_split not in dataset:
            raise SystemExit(
                f"Split '{requested_split}' was not found. Available: "
                + ", ".join(dataset.keys())
            )
        return requested_split, dataset[requested_split]
    if "train" in dataset:
        return "train", dataset["train"]
    first_split = next(iter(dataset.keys()))
    return first_split, dataset[first_split]


def read_local_problem_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = load_json(path)
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        data = data["records"]
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise SystemExit(f"{path} must contain a JSON list (or object.records list).")
    return data


def create_problem_snapshot(args: argparse.Namespace, snapshot_path: Path) -> list[dict[str, Any]]:
    if snapshot_path.exists():
        snapshot = load_json(snapshot_path)
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("records"), list):
            raise SystemExit(f"Invalid cached problem snapshot: {snapshot_path}")
        records = snapshot["records"]
        LOGGER.info("Loaded pinned problem snapshot: %d rows from %s", len(records), snapshot_path)
        return records

    if args.problems_file:
        source_rows = read_local_problem_rows(args.problems_file)
        split_name = args.split or "local"
        dataset_name = str(args.problems_file)
    else:
        if args.hf_endpoint:
            os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
        datasets = require_package("datasets")
        try:
            loaded = datasets.load_dataset(
                args.dataset,
                split=args.split,
                cache_dir=str(args.cache_dir),
                streaming=not args.no_streaming,
            )
        except Exception as exc:
            endpoint = args.hf_endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
            raise SystemExit(
                f"Could not load {args.dataset} through {endpoint}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        split_name, source_rows = choose_split(loaded, args.split)
        dataset_name = args.dataset

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_item in enumerate(source_rows):
        if index < args.offset:
            continue
        if len(records) >= args.limit:
            break
        item = dict(raw_item)
        normalized = normalize_problem(item, index, dataset_name, split_name)
        if normalized is None:
            LOGGER.warning("Skipping source row %d because no problem text was found", index)
            continue
        pid = normalized["problem_id"]
        if pid in seen_ids:
            raise SystemExit(f"Duplicate problem_id '{pid}' in selected problem snapshot.")
        seen_ids.add(pid)
        records.append(normalized)

    if not records:
        raise SystemExit("No usable problems were selected.")
    snapshot = {
        "dataset": dataset_name,
        "split": split_name,
        "offset": args.offset,
        "requested_limit": args.limit,
        "records": records,
    }
    atomic_write_json(snapshot_path, snapshot)
    LOGGER.info("Pinned %d problems in %s", len(records), snapshot_path)
    return records


def load_profiles(config_path: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    config = load_json(config_path)
    profiles = config.get("profiles") if isinstance(config, dict) else None
    if not isinstance(profiles, dict):
        raise SystemExit(f"{config_path} must contain a profiles object.")
    missing = [name for name in names if name not in profiles]
    if missing:
        raise SystemExit(
            f"Unknown profile(s): {', '.join(missing)}. Available: "
            + ", ".join(sorted(profiles))
        )
    selected: dict[str, dict[str, Any]] = {}
    for name in names:
        profile = profiles[name]
        if not isinstance(profile, dict):
            raise SystemExit(f"Profile '{name}' must be an object.")
        for field in ("base_url", "model"):
            if not profile.get(field):
                raise SystemExit(f"Profile '{name}' is missing '{field}'.")
        if not isinstance(profile.get("request_options", {}), dict):
            raise SystemExit(f"Profile '{name}'.request_options must be an object.")
        selected[name] = dict(profile)
    return selected


def create_client(openai: Any, profile_name: str, profile: dict[str, Any]) -> Any:
    provider = str(profile.get("provider") or "openai").lower()
    api_key = str(profile.get("api_key") or "")
    api_key_env = profile.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(str(api_key_env), "")
    if not api_key and provider not in {"vllm", "local_vllm"}:
        raise SystemExit(f"Set {api_key_env or 'the configured API key'} for '{profile_name}'.")
    return openai.OpenAI(api_key=api_key or "EMPTY", base_url=profile["base_url"])


def object_value(obj: Any, field: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(field)
    value = getattr(obj, field, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(field)
    return None


def coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            text = object_value(item, "text")
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return "" if value is None else str(value)


def response_texts(response: Any) -> tuple[str, str]:
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("Model response did not contain choices[0].message") from exc
    reasoning_value = (
        object_value(message, "reasoning_content")
        or object_value(message, "reasoning")
        or object_value(message, "analysis")
    )
    reasoning = coerce_text(reasoning_value).strip()
    content = coerce_text(object_value(message, "content")).strip()
    return reasoning, content


def extract_tagged_think(text: str) -> tuple[str, str]:
    stripped = text.strip()
    open_tag = stripped.find("<think>")
    if open_tag < 0:
        return "", stripped
    cot_start = open_tag + len("<think>")
    close_tag = stripped.find("</think>", cot_start)
    if close_tag < 0:
        return stripped[cot_start:].strip(), ""
    return (
        stripped[cot_start:close_tag].strip(),
        stripped[close_tag + len("</think>") :].strip(),
    )


def parse_solve_response(response: Any) -> tuple[str, str, str, str]:
    reasoning, content = response_texts(response)
    tagged_cot, tagged_final = extract_tagged_think(content)
    if reasoning:
        final_output = tagged_final if tagged_cot and tagged_final else content
        return reasoning, final_output.strip(), reasoning, content
    if tagged_cot and tagged_final:
        return tagged_cot, tagged_final, reasoning, content
    return "", content, reasoning, content


def parse_reconstruction_response(response: Any) -> tuple[str, str, str]:
    reasoning, content = response_texts(response)
    tagged_cot, tagged_final = extract_tagged_think(content)
    if tagged_final:
        synthetic_cot = tagged_final
    elif tagged_cot:
        synthetic_cot = tagged_cot
    else:
        synthetic_cot = content or reasoning
    return synthetic_cot.strip(), reasoning, content


def fatal_api_error_message(exc: Exception) -> str | None:
    status_code = getattr(exc, "status_code", None)
    if status_code == 402:
        return "API balance is insufficient (HTTP 402)."
    if status_code in {400, 401, 403, 404, 422}:
        return f"Non-retryable HTTP {status_code}: {exc}"
    return None


def call_chat(
    client: Any,
    profile: dict[str, Any],
    messages: list[dict[str, str]],
    max_tokens: int,
    retries: int,
) -> Any:
    request_options = dict(profile.get("request_options") or {})
    max_tokens_parameter = str(profile.get("max_tokens_parameter") or "max_tokens")
    request_payload: dict[str, Any] = {
        "model": profile["model"],
        "messages": messages,
        "timeout": float(profile.get("timeout", 900)),
        max_tokens_parameter: max_tokens,
    }
    request_payload.update(request_options)
    for attempt in range(1, retries + 1):
        try:
            return client.chat.completions.create(**request_payload)
        except Exception as exc:
            fatal_message = fatal_api_error_message(exc)
            if fatal_message:
                raise FatalAPIError(fatal_message) from exc
            if attempt == retries:
                raise
            wait_seconds = min(2**attempt, 30)
            LOGGER.warning(
                "API attempt %d/%d failed for %s: %s: %s; retrying in %ds",
                attempt,
                retries,
                profile["model"],
                type(exc).__name__,
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def solve_messages(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SOLVE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{problem}\n\nReason through the task, then provide a complete "
                "submission in the final answer."
            ),
        },
    ]


def reconstruction_messages(problem: str, final_output: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": build_cot_reconstruction_prompt(problem, final_output),
        }
    ]


def run_futures(
    tasks: list[tuple[str, Callable[[], dict[str, Any] | None]]],
    workers: int,
    on_success: Callable[[dict[str, Any]], None],
    failure_path: Path,
    stage_name: str,
) -> tuple[int, int]:
    if not tasks:
        return 0, 0
    executor = ThreadPoolExecutor(max_workers=workers)
    futures: dict[Future[dict[str, Any] | None], str] = {
        executor.submit(task): identifier for identifier, task in tasks
    }
    succeeded = 0
    failed = 0
    try:
        for completed, future in enumerate(as_completed(futures), start=1):
            identifier = futures[future]
            try:
                result = future.result()
                if result is None:
                    raise RuntimeError("response was missing a required output field")
            except FatalAPIError:
                raise
            except Exception as exc:
                failed += 1
                append_jsonl(
                    failure_path,
                    {
                        "stage": stage_name,
                        "job": identifier,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    },
                )
                LOGGER.error(
                    "[%d/%d] %s failed: %s: %s",
                    completed,
                    len(tasks),
                    identifier,
                    type(exc).__name__,
                    exc,
                )
                continue
            on_success(result)
            succeeded += 1
            LOGGER.info("[%d/%d] completed %s", completed, len(tasks), identifier)
    except BaseException:
        for pending_future in futures:
            pending_future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=False)
    return succeeded, failed


def solve_stage(
    args: argparse.Namespace,
    problems: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> None:
    LOGGER.info("=== Stage 1/4: sample real solutions from %d models ===", len(profiles))
    openai = require_package("openai")
    failure_path = args.output_dir / "failures.jsonl"
    for profile_name, profile in profiles.items():
        output_path = args.output_dir / "solutions" / f"{profile_name}.jsonl"
        existing = records_by_id(output_path, "cot")
        pending = [problem for problem in problems if problem["problem_id"] not in existing]
        LOGGER.info(
            "Solve profile=%s model=%s: total=%d cached=%d pending=%d",
            profile_name,
            profile["model"],
            len(problems),
            len(problems) - len(pending),
            len(pending),
        )
        if not pending:
            continue
        client = create_client(openai, profile_name, profile)
        max_tokens = args.solve_max_tokens or int(profile.get("max_tokens", 32768))
        tasks: list[tuple[str, Callable[[], dict[str, Any] | None]]] = []
        for problem in pending:
            pid = problem["problem_id"]

            def task(problem: dict[str, Any] = problem) -> dict[str, Any] | None:
                response = call_chat(
                    client,
                    profile,
                    solve_messages(problem["problem"]),
                    max_tokens,
                    args.retries,
                )
                cot, final_output, _, _ = parse_solve_response(response)
                if not cot or not final_output:
                    return None
                return {
                    "problem_id": problem["problem_id"],
                    "title": problem["title"],
                    "problem": problem["problem"],
                    "source_index": problem["source_index"],
                    "profile": profile_name,
                    "model": profile["model"],
                    "cot": cot,
                    "final_output": final_output,
                }

            tasks.append((f"{profile_name}/{pid}", task))
        succeeded, failed = run_futures(
            tasks,
            args.workers,
            lambda row, path=output_path: append_jsonl(path, row),
            failure_path,
            "solve",
        )
        LOGGER.info(
            "Solve profile=%s finished: saved=%d failed=%d",
            profile_name,
            succeeded,
            failed,
        )


def common_stage(
    args: argparse.Namespace,
    problems: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    LOGGER.info("=== Stage 2/4: intersect valid solved problems ===")
    solutions = {
        name: records_by_id(args.output_dir / "solutions" / f"{name}.jsonl", "cot")
        for name in profiles
    }
    for name, rows in solutions.items():
        valid = sum(1 for row in rows.values() if str(row.get("final_output") or "").strip())
        LOGGER.info("Valid real solutions for %s: %d", name, valid)
    common_ids = set.intersection(
        *(
            {
                pid
                for pid, row in rows.items()
                if str(row.get("cot") or "").strip()
                and str(row.get("final_output") or "").strip()
            }
            for rows in solutions.values()
        )
    )
    records: list[dict[str, Any]] = []
    for problem in problems:
        pid = problem["problem_id"]
        if pid not in common_ids:
            continue
        records.append(
            {
                **problem,
                "solutions": {
                    name: {
                        "profile": name,
                        "model": profiles[name]["model"],
                        "cot": solutions[name][pid]["cot"],
                        "final_output": solutions[name][pid]["final_output"],
                    }
                    for name in profiles
                },
            }
        )
    common_path = args.output_dir / "common_problems.json"
    atomic_write_json(
        common_path,
        {
            "profiles": list(profiles),
            "count": len(records),
            "records": records,
        },
    )
    if not records:
        raise SystemExit("No problem has a valid cot and final_output from every profile.")
    LOGGER.info("Common solved problems: %d; wrote %s", len(records), common_path)
    return records


def load_common_records(path: Path, expected_profiles: list[str]) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise SystemExit(f"Invalid common-problem file: {path}")
    stored_profiles = data.get("profiles")
    if stored_profiles != expected_profiles:
        raise SystemExit(
            f"Profile order in {path} is {stored_profiles}, expected {expected_profiles}. "
            "Run --stage filter with the current --profiles first."
        )
    return data["records"]


def reconstruction_path(output_dir: Path, target: str, generator: str) -> Path:
    return output_dir / "reconstructions" / f"target_{target}__generator_{generator}.jsonl"


def reconstruct_stage(
    args: argparse.Namespace,
    common_records: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> None:
    routes = len(profiles) * len(profiles)
    LOGGER.info(
        "=== Stage 3/4: reconstruct CoTs (%d problems x %d routes = %d jobs) ===",
        len(common_records),
        routes,
        len(common_records) * routes,
    )
    openai = require_package("openai")
    failure_path = args.output_dir / "failures.jsonl"
    for target_name in profiles:
        for generator_name, generator_profile in profiles.items():
            output_path = reconstruction_path(args.output_dir, target_name, generator_name)
            existing = records_by_id(output_path, "synthetic_cot")
            pending = [row for row in common_records if row["problem_id"] not in existing]
            LOGGER.info(
                "Reconstruction target=%s generator=%s: total=%d cached=%d pending=%d",
                target_name,
                generator_name,
                len(common_records),
                len(common_records) - len(pending),
                len(pending),
            )
            if not pending:
                continue
            client = create_client(openai, generator_name, generator_profile)
            tasks: list[tuple[str, Callable[[], dict[str, Any] | None]]] = []
            for common_row in pending:
                pid = common_row["problem_id"]

                def task(common_row: dict[str, Any] = common_row) -> dict[str, Any] | None:
                    target_solution = common_row["solutions"][target_name]
                    response = call_chat(
                        client,
                        generator_profile,
                        reconstruction_messages(
                            common_row["problem"], target_solution["final_output"]
                        ),
                        args.reconstruction_max_tokens,
                        args.retries,
                    )
                    synthetic_cot, _, _ = parse_reconstruction_response(response)
                    if not synthetic_cot:
                        return None
                    return {
                        "problem_id": common_row["problem_id"],
                        "source_index": common_row["source_index"],
                        "target_profile": target_name,
                        "target_model": profiles[target_name]["model"],
                        "generator_profile": generator_name,
                        "generator_model": generator_profile["model"],
                        "final_output": target_solution["final_output"],
                        "synthetic_cot": synthetic_cot,
                    }

                tasks.append((f"{target_name}/{generator_name}/{pid}", task))
            succeeded, failed = run_futures(
                tasks,
                args.workers,
                lambda row, path=output_path: append_jsonl(path, row),
                failure_path,
                "reconstruct",
            )
            LOGGER.info(
                "Reconstruction target=%s generator=%s finished: saved=%d failed=%d",
                target_name,
                generator_name,
                succeeded,
                failed,
            )


def build_dpo_rows(
    common_records: list[dict[str, Any]],
    profile_names: list[str],
    reconstruction_records: dict[tuple[str, str], dict[str, dict[str, Any]]],
    skip_identical: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped_incomplete = 0
    for common_row in common_records:
        pid = common_row["problem_id"]
        if any(
            pid not in reconstruction_records[(target, generator)]
            for target in profile_names
            for generator in profile_names
        ):
            skipped_incomplete += 1
            continue

        for target in profile_names:
            target_solution = common_row["solutions"][target]
            prompt = build_cot_reconstruction_prompt(
                common_row["problem"], target_solution["final_output"]
            )
            synthetic = {
                generator: reconstruction_records[(target, generator)][pid]["synthetic_cot"]
                for generator in profile_names
            }

            for generator in profile_names:
                chosen = target_solution["cot"]
                rejected = synthetic[generator]
                if skip_identical and chosen.strip() == rejected.strip():
                    continue
                rows.append(
                    {
                        "problem_id": pid,
                        "target_profile": target,
                        "target_model": target_solution["model"],
                        "pair_type": "real_vs_synthetic",
                        "chosen_source": f"real:{target}",
                        "rejected_source": f"synthetic:{generator}",
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )

    return rows, skipped_incomplete


def build_stage(
    args: argparse.Namespace,
    common_records: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> None:
    LOGGER.info("=== Stage 4/4: build DPO preference pairs ===")
    reconstruction_records = {
        (target, generator): records_by_id(
            reconstruction_path(args.output_dir, target, generator), "synthetic_cot"
        )
        for target in profiles
        for generator in profiles
    }
    rows, skipped_incomplete = build_dpo_rows(
        common_records,
        list(profiles),
        reconstruction_records,
        args.skip_identical_pairs,
    )
    if not rows:
        raise SystemExit(
            "No complete DPO rows could be built. Finish the reconstruction stage first."
        )
    output_path = args.output or args.output_dir / "hardtests_dpo.json"
    atomic_write_json(output_path, rows)
    complete_problems = len(common_records) - skipped_incomplete
    LOGGER.info(
        "DPO dataset complete: problems=%d skipped_incomplete=%d pairs=%d output=%s",
        complete_problems,
        skipped_incomplete,
        len(rows),
        output_path,
    )
    expected_pairs = complete_problems * len(profiles) ** 2
    if not args.skip_identical_pairs and len(rows) != expected_pairs:
        raise AssertionError(f"Expected {expected_pairs} pairs, built {len(rows)}")


def print_dry_run_plan(
    args: argparse.Namespace,
    problems: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> None:
    solve_jobs = len(problems) * len(profiles)
    reconstruction_jobs = len(problems) * len(profiles) ** 2
    max_pairs = len(problems) * len(profiles) ** 2
    LOGGER.info(
        "Dry run: problems=%d profiles=%d solve_jobs=%d max_reconstruction_jobs=%d max_pairs=%d",
        len(problems),
        len(profiles),
        solve_jobs,
        reconstruction_jobs,
        max_pairs,
    )
    for name, profile in profiles.items():
        LOGGER.info(
            "  profile=%s provider=%s model=%s base_url=%s",
            name,
            profile.get("provider", "openai"),
            profile["model"],
            profile["base_url"],
        )


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    setup_logging(args.output_dir, args.log_level)
    profiles = load_profiles(args.config, args.profiles)
    snapshot_path = args.output_dir / "problems.json"

    if args.stage in {"all", "solve"} or args.dry_run:
        problems = create_problem_snapshot(args, snapshot_path)
    else:
        if not snapshot_path.exists():
            raise SystemExit(f"Missing {snapshot_path}; run --stage solve first.")
        problems = create_problem_snapshot(args, snapshot_path)

    LOGGER.info(
        "Pipeline stage=%s profiles=%s output_dir=%s",
        args.stage,
        ",".join(profiles),
        args.output_dir,
    )
    if args.dry_run:
        print_dry_run_plan(args, problems, profiles)
        return

    if args.stage in {"all", "solve"}:
        solve_stage(args, problems, profiles)
        if args.stage == "solve":
            return

    if args.stage in {"all", "filter"}:
        common_records = common_stage(args, problems, profiles)
        if args.stage == "filter":
            return
    else:
        common_records = load_common_records(
            args.output_dir / "common_problems.json", list(profiles)
        )

    if args.stage in {"all", "reconstruct"}:
        reconstruct_stage(args, common_records, profiles)
        if args.stage == "reconstruct":
            return

    if args.stage in {"all", "build"}:
        build_stage(args, common_records, profiles)


if __name__ == "__main__":
    main()
