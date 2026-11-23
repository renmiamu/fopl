from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Iterable, Iterator
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cot_prompt import build_cot_reconstruction_prompt


DEFAULT_DATASET = "open-r1/OpenR1-Math-220k"
DEFAULT_DATASET_CONFIG = "default"
DEFAULT_SPLIT = "train"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/model_configs.json"
DEFAULT_HF_CACHE = PROJECT_ROOT / ".cache/huggingface"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/sft"
DEFAULT_REFERENCE_PROFILES = (
    "deepseek_v4_pro",
    "kimi_k2p5",
    "glm_5_2",
)
DEFAULT_LOCAL_PROFILE_NAME = "local_reconstruction"
DEFAULT_LOCAL_MODEL_PATH = "reconstructor"
DEFAULT_SYNTHETIC_BASE_URL = "http://localhost:8000/v1"
REFERENCE_SYSTEM_PROMPT = (
    "You are an expert math reasoning assistant. Solve the problem carefully. "
    "Return a detailed reasoning trace and a final answer. If the API does not "
    "provide reasoning in a separate field, format the response as "
    "<think>reasoning</think> followed by the final answer."
)
SFT_SYSTEM_PROMPT = (
    "You are an expert math reasoning assistant. In the final output, provide "
    "the answer and a brief explanation."
)
FINAL_ONLY_SFT_SYSTEM_PROMPT = (
    "You are an expert math assistant. Give only the answer and a brief explanation."
)
FINAL_ONLY_REFERENCE_SYSTEM_PROMPT = (
    "You are an expert math assistant. Solve the problem internally. In your "
    "response, give only the final answer and a brief explanation. Do not include "
    "a reasoning trace or <think> tags."
)
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class FatalAPIError(RuntimeError):
    """An API error that should stop the route instead of skipping every row."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the OpenR1-Math SFT matrix: generate reference "
            "outputs with multiple API models, save final-only variants, then "
            "reconstruct every reference answer with every configured API model "
            "and a local reconstruction model."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("all", "reference", "reconstruct", "synthetic", "final-only"),
        default="all",
        help=(
            "Run the complete pipeline, reference generation, reconstruction, "
            "or final-only conversion. synthetic is an alias for reconstruct "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset name (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--dataset-config",
        default=DEFAULT_DATASET_CONFIG,
        help=(
            "Hugging Face dataset config/subset, such as default, extended, or "
            f"all (default: {DEFAULT_DATASET_CONFIG})."
        ),
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split (default: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_HF_CACHE,
        help=f"Hugging Face cache directory (default: {DEFAULT_HF_CACHE}).",
    )
    parser.add_argument(
        "--hf-endpoint",
        help="Alternative Hugging Face endpoint, for example https://hf-mirror.com.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download/materialize the dataset instead of reading it in streaming mode.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Model config JSON path (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Matrix output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--reference-profiles",
        nargs="+",
        default=list(DEFAULT_REFERENCE_PROFILES),
        help=(
            "Profiles that generate reference outputs. Defaults to "
            "deepseek_v4_pro, kimi_k2p5, and glm_5_2."
        ),
    )
    parser.add_argument(
        "--reconstruction-profiles",
        nargs="+",
        default=list(DEFAULT_REFERENCE_PROFILES),
        help=(
            "Configured API profiles used to reconstruct each reference final "
            "output. The local reconstruction profile is added separately."
        ),
    )
    parser.add_argument(
        "--local-profile-name",
        default=DEFAULT_LOCAL_PROFILE_NAME,
        help=(
            "Name used in local reconstruction output filenames "
            f"(default: {DEFAULT_LOCAL_PROFILE_NAME})."
        ),
    )
    parser.add_argument(
        "--local-model-path",
        default=DEFAULT_LOCAL_MODEL_PATH,
        help=(
            "Local reconstruction checkpoint and default vLLM served model name "
            f"(default: {DEFAULT_LOCAL_MODEL_PATH})."
        ),
    )
    parser.add_argument(
        "--local-served-model-name",
        help=(
            "Override the model name sent to the local OpenAI-compatible server; "
            "defaults to --local-model-path."
        ),
    )
    parser.add_argument(
        "--no-local-reconstruction",
        action="store_true",
        help="Do not include the local reconstruction model in the matrix.",
    )
    parser.add_argument(
        "--local-reconstruction-only",
        action="store_true",
        help=(
            "Run only the local reconstruction model; ignore configured API "
            "reconstruction profiles."
        ),
    )
    parser.add_argument(
        "--reference-profile",
        default=None,
        help=(
            "Legacy singular form of --reference-profiles; when supplied it "
            "replaces that list."
        ),
    )
    parser.add_argument(
        "--synthetic-profile",
        help="Optional local/vLLM profile from the model config.",
    )
    parser.add_argument(
        "--synthetic-base-url",
        "--local-base-url",
        "--base-url",
        default=DEFAULT_SYNTHETIC_BASE_URL,
        help=(
            "Local OpenAI-compatible base URL "
            f"(default: {DEFAULT_SYNTHETIC_BASE_URL})."
        ),
    )
    parser.add_argument(
        "--synthetic-api-key",
        default="EMPTY",
        help="API key for the local OpenAI-compatible server (default: EMPTY).",
    )
    parser.add_argument(
        "--synthetic-model",
        "--model",
        help=(
            "Legacy alias for --local-served-model-name. If omitted, the local "
            "model path is sent as the served model name."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Target this many successfully saved rows per reference dataset; "
            "failed source rows are skipped and replaced by later rows."
        ),
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many source rows before generation (default: 0).",
    )
    parser.add_argument(
        "--reference-max-tokens",
        type=int,
        help="Override the reference API profile's max_tokens.",
    )
    parser.add_argument(
        "--synthetic-max-tokens",
        type=int,
        default=32768,
        help="Maximum reconstruction completion tokens per problem (default: 32768).",
    )
    parser.add_argument(
        "--synthetic-temperature",
        type=float,
        default=1.0,
        help="Synthetic sampling temperature (default: 1.0).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Maximum attempts per model call (default: 1; failure is skipped).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel rows to generate (default: 1).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Deprecated compatibility flag; matrix outputs always resume safely.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard outputs/checkpoints for the requested stage before running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selected rows and prompts without calling the model or writing output.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.offset < 0:
        parser.error("--offset cannot be negative")
    if args.reference_max_tokens is not None and args.reference_max_tokens <= 0:
        parser.error("--reference-max-tokens must be positive")
    if args.synthetic_max_tokens <= 0:
        parser.error("--synthetic-max-tokens must be positive")
    if args.retries <= 0:
        parser.error("--retries must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.reference_profile:
        args.reference_profiles = [args.reference_profile]
    if len(set(args.reference_profiles)) != len(args.reference_profiles):
        parser.error("--reference-profiles contains duplicate names")
    if len(set(args.reconstruction_profiles)) != len(args.reconstruction_profiles):
        parser.error("--reconstruction-profiles contains duplicate names")
    if args.local_reconstruction_only and args.no_local_reconstruction:
        parser.error(
            "--local-reconstruction-only cannot be combined with "
            "--no-local-reconstruction"
        )
    if not args.local_profile_name.strip():
        parser.error("--local-profile-name cannot be empty")
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_package(package: str, install_name: str | None = None) -> Any:
    try:
        return __import__(package, fromlist=["*"])
    except ImportError as exc:
        name = install_name or package
        raise SystemExit(
            f"Missing dependency '{name}'. Install it with: pip install {name}"
        ) from exc


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object.")
    return data


def load_profile(config_path: Path, profile_name: str) -> dict[str, Any]:
    config = load_json_object(resolve_path(config_path))
    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or profile_name not in profiles:
        available = ", ".join(sorted(profiles)) if isinstance(profiles, dict) else ""
        raise SystemExit(
            f"Unknown model profile '{profile_name}' in {config_path}. "
            f"Available profiles: {available}"
        )
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise SystemExit(f"Model profile '{profile_name}' must be an object.")
    return dict(profile)


def load_profiles(
    config_path: Path, profile_names: Iterable[str]
) -> dict[str, dict[str, Any]]:
    names = list(profile_names)
    config = load_json_object(resolve_path(config_path))
    configured = config.get("profiles")
    if not isinstance(configured, dict):
        raise SystemExit(f"{config_path} must contain a JSON 'profiles' object.")
    missing = [name for name in names if name not in configured]
    if missing:
        raise SystemExit(
            f"Unknown model profile(s) in {config_path}: {', '.join(missing)}. "
            f"Available profiles: {', '.join(sorted(configured))}"
        )
    selected: dict[str, dict[str, Any]] = {}
    for name in names:
        profile = configured[name]
        if not isinstance(profile, dict):
            raise SystemExit(f"Model profile '{name}' must be an object.")
        for required in ("base_url", "model"):
            if not str(profile.get(required) or "").strip():
                raise SystemExit(f"Model profile '{name}' is missing '{required}'.")
        if not isinstance(profile.get("request_options", {}), dict):
            raise SystemExit(f"Model profile '{name}'.request_options must be an object.")
        if profile.get("reference_output_mode", "cot_and_final") not in {
            "cot_and_final", "final_only"
        }:
            raise SystemExit(
                f"Model profile '{name}'.reference_output_mode must be "
                "'cot_and_final' or 'final_only'."
            )
        selected[name] = dict(profile)
    return selected


def synthetic_profile_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.synthetic_profile:
        return load_profile(args.config, args.synthetic_profile)
    return {
        "provider": "vllm",
        "base_url": args.synthetic_base_url,
        "api_key": args.synthetic_api_key,
        "model": (
            args.synthetic_model
            or getattr(args, "local_served_model_name", None)
            or getattr(args, "local_model_path", DEFAULT_LOCAL_MODEL_PATH)
        ),
        "model_path": getattr(args, "local_model_path", DEFAULT_LOCAL_MODEL_PATH),
        "timeout": 900,
        "request_options": {"temperature": args.synthetic_temperature},
    }


def request_timeout(profile: dict[str, Any], fallback: float = 900.0) -> float:
    try:
        return float(profile.get("timeout", fallback))
    except (TypeError, ValueError):
        return fallback


def request_max_tokens(profile: dict[str, Any], fallback: int) -> int:
    try:
        return int(profile.get("max_tokens", fallback))
    except (TypeError, ValueError):
        return fallback


def create_client(openai: Any, profile: dict[str, Any]) -> Any:
    api_key = str(profile.get("api_key") or "")
    api_key_env = profile.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(str(api_key_env), "")
    if not api_key:
        api_key = "EMPTY"
    return openai.OpenAI(
        api_key=api_key, base_url=profile["base_url"], max_retries=0
    )


def reference_is_final_only(profile: dict[str, Any]) -> bool:
    return profile.get("reference_output_mode") == "final_only"


def is_local_vllm_profile(profile: dict[str, Any]) -> bool:
    if str(profile.get("provider") or "").lower() == "vllm":
        return True
    hostname = urlsplit(str(profile.get("base_url") or "")).hostname
    return hostname in {"localhost", "127.0.0.1", "::1"}


def is_timeout_exception(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (
            isinstance(current, TimeoutError)
            or "timeout" in type(current).__name__.lower()
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def fatal_api_error_message(exc: BaseException) -> str | None:
    status_code = getattr(exc, "status_code", None)
    if status_code == 402:
        return "API balance is insufficient (HTTP 402)."
    if status_code in {401, 403, 404}:
        return f"Non-retryable HTTP {status_code}: {exc}"
    return None


def reference_prompt(
    problem: str, final_only: bool = False
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                FINAL_ONLY_REFERENCE_SYSTEM_PROMPT
                if final_only else REFERENCE_SYSTEM_PROMPT
            ),
        },
        {
            "role": "user",
            "content": (
                "Solve the following MATH problem.\n\n"
                f"Problem:\n{problem}\n\n"
                "In the final output, briefly explain the answer instead of "
                "only stating it."
            ),
        },
    ]


def synthetic_prompt(problem: str, final_output: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": build_cot_reconstruction_prompt(problem, final_output),
        },
    ]


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
        parts: list[str] = []
        for item in value:
            text = object_value(item, "text")
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return "" if value is None else str(value)


def call_chat(
    client: Any,
    profile: dict[str, Any],
    messages: list[dict[str, str]],
    max_tokens: int,
    retries: int,
    extra_request_options: dict[str, Any] | None = None,
    retry_on_timeout: bool = True,
) -> Any:
    request_options = dict(profile.get("request_options") or {})
    if extra_request_options:
        request_options.update(extra_request_options)
    if profile.get("supports_temperature") is False:
        request_options.pop("temperature", None)
    max_tokens_parameter = str(profile.get("max_tokens_parameter") or "max_tokens")
    request_payload: dict[str, Any] = {
        "model": profile["model"],
        "messages": messages,
        "timeout": request_timeout(profile),
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
            if not retry_on_timeout and is_timeout_exception(exc):
                raise
            if attempt == retries:
                raise
            wait_seconds = min(2**attempt, 30) + random.uniform(0, 1)
            headers = getattr(getattr(exc, "response", None), "headers", None)
            if headers:
                try:
                    wait_seconds = max(
                        wait_seconds, min(float(headers.get("retry-after", 0)), 120)
                    )
                except (TypeError, ValueError):
                    pass
            print(
                f"  attempt {attempt}/{retries} failed for {profile['model']}: "
                f"{type(exc).__name__}: {exc}; retrying in {wait_seconds:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def extract_cot_and_final(result: str) -> tuple[str, str]:
    result = result.strip()
    if not result:
        return "", ""
    open_index = result.find(THINK_OPEN)
    close_index = result.find(THINK_CLOSE)
    if open_index != -1:
        cot_start = open_index + len(THINK_OPEN)
        close_after_open = result.find(THINK_CLOSE, cot_start)
        if close_after_open != -1:
            return (
                result[cot_start:close_after_open].strip(),
                result[close_after_open + len(THINK_CLOSE) :].strip(),
            )
        return result[cot_start:].strip(), ""
    if close_index != -1:
        return (
            result[:close_index].strip(),
            result[close_index + len(THINK_CLOSE) :].strip(),
        )
    return "", result


def response_trace_and_output(response: Any) -> tuple[str, str, str]:
    message = response.choices[0].message
    reasoning = coerce_text(
        object_value(message, "reasoning_content")
        or object_value(message, "reasoning")
        or object_value(message, "analysis")
    )
    content = coerce_text(object_value(message, "content"))
    if reasoning.strip():
        return reasoning.strip(), content.strip(), content
    cot, final_output = extract_cot_and_final(content)
    if cot:
        return cot, final_output, content
    return "", content.strip(), content


def final_only_text(content: str) -> str:
    """Discard visible think blocks and reject incomplete reasoning tags."""
    content = re.sub(
        r"<think\b[^>]*>.*?</think\s*>", "", content, flags=re.IGNORECASE | re.DOTALL
    )
    if re.search(r"<think\b", content, flags=re.IGNORECASE):
        return ""
    closing_tags = list(re.finditer(r"</think\s*>", content, flags=re.IGNORECASE))
    if closing_tags:
        content = content[closing_tags[-1].end():]
    return content.strip()


def response_final_only(response: Any) -> str:
    """Use only visible assistant content, never a provider's reasoning field."""
    message = response.choices[0].message
    return final_only_text(coerce_text(object_value(message, "content")))


def response_trace_only(response: Any) -> tuple[str, str]:
    message = response.choices[0].message
    content = coerce_text(object_value(message, "content"))
    if not content.strip():
        return "", content
    cot, _ = extract_cot_and_final(content)
    return (cot or content).strip(), content


def text_field(item: dict[str, Any], field: str) -> str:
    value = item.get(field)
    return value.strip() if isinstance(value, str) else ""


def source_row_task(
    args: argparse.Namespace, index: int, item: dict[str, Any]
) -> dict[str, Any] | None:
    problem = text_field(item, "problem")
    if not problem:
        return None
    pid = text_field(item, "uuid") or f"{args.dataset_config}_{args.split}_{index}"
    task: dict[str, Any] = {
        "source_index": index,
        "problem_id": pid,
        "problem": problem,
    }
    for field in ("problem_type", "question_type", "source", "correctness_count"):
        if item.get(field) is not None:
            task[field] = item[field]
    return task


def load_dataset_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
    datasets = require_package("datasets")
    try:
        return datasets.load_dataset(
            args.dataset,
            args.dataset_config,
            split=args.split,
            cache_dir=str(resolve_path(args.cache_dir)),
            streaming=not args.no_streaming,
        )
    except Exception as exc:
        endpoint = args.hf_endpoint or os.environ.get(
            "HF_ENDPOINT", "https://huggingface.co"
        )
        raise SystemExit(
            f"Could not load {args.dataset}/{args.dataset_config}:{args.split} "
            f"through {endpoint}.\n"
            "Check your Hugging Face login, Internet/proxy settings, or try "
            "--hf-endpoint https://hf-mirror.com.\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def pending_tasks(
    args: argparse.Namespace,
    dataset_rows: Iterable[dict[str, Any]],
    completed_ids: set[str],
    max_tasks: int | None = None,
) -> Iterator[dict[str, Any]]:
    queued = 0
    seen_ids = set(completed_ids)
    for index, raw_item in enumerate(dataset_rows):
        if index < args.offset:
            continue
        task = source_row_task(args, index, dict(raw_item))
        if task is None or task["problem_id"] in seen_ids:
            continue
        seen_ids.add(task["problem_id"])
        yield task
        queued += 1
        if max_tasks is not None and queued >= max_tasks:
            return


def format_response(cot: str, final_output: str) -> str:
    return f"{THINK_OPEN}\n{cot}\n{THINK_CLOSE}\n{final_output}"


def copy_task_metadata(record: dict[str, Any], task: dict[str, Any]) -> None:
    for field in ("problem_type", "question_type", "source", "correctness_count"):
        if field in task:
            record[field] = task[field]


def build_reference_record(
    args: argparse.Namespace,
    reference_profile: dict[str, Any],
    task: dict[str, Any],
    reference_cot: str,
    final_output: str,
    raw_response: str,
    reference_profile_name: str | None = None,
) -> dict[str, Any]:
    reference_profile_name = (
        reference_profile_name
        or getattr(args, "reference_profile", None)
        or str(reference_profile["model"])
    )
    if reference_is_final_only(reference_profile):
        return final_only_reference_record({
            "problem_id": task["problem_id"],
            "problem": task["problem"],
            "final_output": final_output,
            "source_index": task["source_index"],
            "final_source": reference_profile["model"],
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "model": reference_profile["model"],
            "profile": reference_profile_name,
            **{
                field: task[field]
                for field in (
                    "problem_type", "question_type", "source", "correctness_count"
                )
                if field in task
            },
        })
    response = format_response(reference_cot, final_output)
    record = {
        "problem_id": task["problem_id"],
        "problem": task["problem"],
        "cot": reference_cot,
        "reference_cot": reference_cot,
        "final_output": final_output,
        "response": response,
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": task["problem"]},
            {"role": "assistant", "content": response},
        ],
        "data_type": "reference",
        "cot_source": reference_profile["model"],
        "final_source": reference_profile["model"],
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "source_index": task["source_index"],
        "model": reference_profile["model"],
        "profile": reference_profile_name,
        "dataset_name": f"{reference_profile_name}_ref",
        "raw_response": raw_response,
    }
    copy_task_metadata(record, task)
    return record


def build_reference_final_only_record(
    reference_record: dict[str, Any],
) -> dict[str, Any]:
    final_output = str(reference_record["final_output"]).strip()
    record = {
        "problem_id": reference_record["problem_id"],
        "problem": reference_record["problem"],
        "final_output": final_output,
        "response": final_output,
        "messages": [
            {"role": "system", "content": FINAL_ONLY_SFT_SYSTEM_PROMPT},
            {"role": "user", "content": reference_record["problem"]},
            {"role": "assistant", "content": final_output},
        ],
        "data_type": "reference_final_only",
        "source_index": reference_record["source_index"],
    }
    for field in (
        "final_source",
        "dataset",
        "dataset_config",
        "split",
        "model",
        "profile",
        "problem_type",
        "question_type",
        "source",
        "correctness_count",
    ):
        if field in reference_record:
            record[field] = reference_record[field]
    profile_name = str(reference_record.get("profile") or "").strip()
    if profile_name:
        record["dataset_name"] = f"{profile_name}_final_only"
    return record


def final_only_reference_record(reference_record: dict[str, Any]) -> dict[str, Any]:
    """Remove stored reasoning from a reference row while keeping its identity."""
    final_output = final_only_text(str(reference_record["final_output"]))
    if not final_output:
        raise ValueError("Reference row has no usable final output")
    record = build_reference_final_only_record({
        **reference_record, "final_output": final_output
    })
    record["data_type"] = "reference"
    if record.get("profile"):
        record["dataset_name"] = f"{record['profile']}_ref"
    return record


def build_sft_record(
    args: argparse.Namespace,
    reference_profile: dict[str, Any],
    synthetic_profile: dict[str, Any],
    task: dict[str, Any],
    reference_cot: str,
    synthetic_cot: str,
    final_output: str,
    raw_reference_response: str,
    raw_synthetic_response: str,
    reference_profile_name: str | None = None,
    synthetic_profile_name: str | None = None,
) -> dict[str, Any]:
    reference_profile_name = (
        reference_profile_name
        or getattr(args, "reference_profile", None)
        or str(reference_profile["model"])
    )
    synthetic_profile_name = synthetic_profile_name or str(
        synthetic_profile.get("profile_name") or synthetic_profile["model"]
    )
    response = format_response(synthetic_cot, final_output)
    record = {
        "problem_id": task["problem_id"],
        "problem": task["problem"],
        "cot": synthetic_cot,
        "synthetic_cot": synthetic_cot,
        "reference_cot": reference_cot,
        "final_output": final_output,
        "response": response,
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": task["problem"]},
            {"role": "assistant", "content": response},
        ],
        "data_type": "synthetic_correct",
        "cot_source": synthetic_profile["model"],
        "final_source": reference_profile["model"],
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "source_index": task["source_index"],
        "model": synthetic_profile["model"],
        "reference_model": reference_profile["model"],
        "reference_profile": reference_profile_name,
        "reconstruction_profile": synthetic_profile_name,
        "dataset_name": (
            f"{synthetic_profile_name}_syn_from_{reference_profile_name}"
        ),
        "raw_reference_response": raw_reference_response,
        "raw_synthetic_response": raw_synthetic_response,
    }
    copy_task_metadata(record, task)
    return record


def generate_reference_one(
    args: argparse.Namespace,
    openai: Any,
    reference_profile: dict[str, Any],
    task: dict[str, Any],
    reference_profile_name: str | None = None,
) -> dict[str, Any]:
    reference_client = create_client(openai, reference_profile)
    final_only = reference_is_final_only(reference_profile)
    for response_attempt in range(1, args.retries + 1):
        reference_response = call_chat(
            client=reference_client,
            profile=reference_profile,
            messages=reference_prompt(task["problem"], final_only=final_only),
            max_tokens=(
                args.reference_max_tokens
                or request_max_tokens(reference_profile, 32768)
            ),
            retries=args.retries,
        )
        if final_only:
            final_output = response_final_only(reference_response)
            reference_cot, raw_reference = "", ""
        else:
            reference_cot, final_output, raw_reference = response_trace_and_output(
                reference_response
            )
        if final_output and (final_only or reference_cot):
            return build_reference_record(
                args=args,
                reference_profile=reference_profile,
                task=task,
                reference_cot=reference_cot,
                final_output=final_output,
                raw_response=raw_reference,
                reference_profile_name=reference_profile_name,
            )
        if response_attempt < args.retries:
            wait_seconds = min(2**response_attempt, 30)
            print(
                f"  reference response {response_attempt}/{args.retries} for "
                f"{task['problem_id']} did not contain a usable "
                f"{'final output' if final_only else 'CoT and final output'}; "
                f"retrying in {wait_seconds}s",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(
        f"Reference API response for {task['problem_id']} did not contain a usable "
        f"{'final output' if final_only else 'CoT and final output'} "
        f"after {args.retries} attempt(s)"
    )


def generate_synthetic_one(
    args: argparse.Namespace,
    openai: Any,
    reference_profile: dict[str, Any],
    synthetic_profile: dict[str, Any],
    task: dict[str, Any],
    reference_record: dict[str, Any],
    reference_profile_name: str | None = None,
    synthetic_profile_name: str | None = None,
) -> dict[str, Any]:
    final_output = str(reference_record["final_output"])
    synthetic_client = create_client(openai, synthetic_profile)
    for response_attempt in range(1, args.retries + 1):
        synthetic_response = call_chat(
            client=synthetic_client,
            profile=synthetic_profile,
            messages=synthetic_prompt(task["problem"], final_output),
            max_tokens=args.synthetic_max_tokens,
            retries=args.retries,
            extra_request_options={"temperature": args.synthetic_temperature},
            retry_on_timeout=not is_local_vllm_profile(synthetic_profile),
        )
        synthetic_cot, raw_synthetic = response_trace_only(synthetic_response)
        if synthetic_cot:
            return build_sft_record(
                args=args,
                reference_profile=reference_profile,
                synthetic_profile=synthetic_profile,
                task=task,
                reference_cot=str(
                    reference_record.get("reference_cot")
                    or reference_record.get("cot")
                    or ""
                ),
                synthetic_cot=synthetic_cot,
                final_output=final_output,
                raw_reference_response=str(
                    reference_record.get("raw_response") or ""
                ),
                raw_synthetic_response=raw_synthetic,
                reference_profile_name=reference_profile_name,
                synthetic_profile_name=synthetic_profile_name,
            )
        if response_attempt < args.retries:
            wait_seconds = min(2**response_attempt, 30)
            print(
                f"  synthetic response {response_attempt}/{args.retries} for "
                f"{task['problem_id']} did not contain a CoT; retrying in "
                f"{wait_seconds}s",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(
        f"Local API response for {task['problem_id']} did not contain a CoT "
        f"after {args.retries} attempt(s)"
    )


def journal_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".partial.jsonl")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"Skipping malformed checkpoint row {path}:{line_number}: "
                        f"{exc}",
                        file=sys.stderr,
                    )
                    continue
                if not isinstance(item, dict):
                    print(
                        f"Skipping non-object checkpoint row {path}:{line_number}",
                        file=sys.stderr,
                    )
                    continue
                records.append(item)
    except (OSError, UnicodeError) as exc:
        raise SystemExit(f"Could not read JSONL {path}: {exc}") from exc
    return records


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        return load_jsonl_records(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise SystemExit(f"Could not read existing output {path}: {exc}") from exc
    except json.JSONDecodeError:
        # Also accept JSONL content that was copied or renamed with a .json suffix.
        return load_jsonl_records(path)
    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise SystemExit(
            f"Existing output {path} must contain a JSON array, a JSON object, "
            "or one JSON object per line."
        )
    return data


def load_journal_records(path: Path) -> list[dict[str, Any]]:
    return load_jsonl_records(path)


def index_sft_records(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        pid = str(record.get("problem_id") or "").strip()
        if (
            pid
            and record.get("messages")
            and record.get("synthetic_cot")
            and record.get("final_output")
        ):
            indexed[pid] = record
    return indexed


def index_reference_records(
    records: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        pid = str(record.get("problem_id") or "").strip()
        if (
            pid
            and record.get("messages")
            and record.get("problem")
            and record.get("final_output")
        ):
            indexed[pid] = record
    return indexed


def append_journal(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
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


def write_sft_json(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            if path.suffix.lower() == ".jsonl":
                for record in records:
                    handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    )
                    handle.write("\n")
            else:
                handle.write("[\n")
                for index, record in enumerate(records):
                    if index:
                        handle.write(",\n")
                    handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n]\n")
        temporary_path.replace(path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def validate_output_name(name: str) -> str:
    """Keep profile-derived output paths inside the requested output directory."""
    stripped = name.strip()
    if not stripped or stripped in {".", ".."} or Path(stripped).name != stripped:
        raise SystemExit(f"Invalid profile name for an output filename: {name!r}")
    return stripped


def reference_output_path(output_dir: Path, profile_name: str) -> Path:
    return output_dir / f"{validate_output_name(profile_name)}_ref.json"


def final_only_output_path(output_dir: Path, profile_name: str) -> Path:
    return output_dir / f"{validate_output_name(profile_name)}_final_only.json"


def synthetic_output_path(
    output_dir: Path, reconstruction_name: str, reference_name: str
) -> Path:
    reconstruction_name = validate_output_name(reconstruction_name)
    reference_name = validate_output_name(reference_name)
    return output_dir / f"{reconstruction_name}_syn_from_{reference_name}.json"


def sorted_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda item: int(item.get("source_index", 0)))


def reset_paths(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def combined_reference_records(path: Path) -> dict[str, dict[str, Any]]:
    return index_reference_records(
        [*load_json_records(path), *load_journal_records(journal_path(path))]
    )


def combined_synthetic_records(path: Path) -> dict[str, dict[str, Any]]:
    return index_sft_records(
        [*load_json_records(path), *load_journal_records(journal_path(path))]
    )


def failure_record(
    stage: str,
    profile_name: str,
    problem_id: str,
    exc: BaseException,
    reference_name: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stage": stage,
        "profile": profile_name,
        "problem_id": problem_id,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if reference_name is not None:
        record["reference_profile"] = reference_name
    return record


def generate_reference_dataset(
    args: argparse.Namespace,
    openai: Any,
    profile_name: str,
    profile: dict[str, Any],
    tasks: Iterable[dict[str, Any]],
    output_dir: Path,
) -> tuple[int, int]:
    output_path = reference_output_path(output_dir, profile_name)
    partial_path = journal_path(output_path)
    final_only_path = final_only_output_path(output_dir, profile_name)
    if args.overwrite:
        reset_paths(output_path, partial_path, final_only_path)
    records = combined_reference_records(output_path)
    if reference_is_final_only(profile):
        # Clean previously saved CoT rows when resuming this profile.
        cleaned_records = {}
        for pid, row in records.items():
            try:
                cleaned_records[pid] = final_only_reference_record(row)
            except ValueError:
                print(
                    f"Discarding cached {profile_name} row {pid}: "
                    "no usable final output",
                    file=sys.stderr,
                )
        records = cleaned_records
        # A failed run should also leave its existing checkpoint free of CoT.
        write_sft_json(output_path, sorted_records(records.values()))
        reset_paths(partial_path)
    target = args.limit
    pending = (task for task in tasks if task["problem_id"] not in records)
    print(
        f"Reference {profile_name}: cached_successes={len(records)} "
        f"target_successes={target if target is not None else 'all source rows'}"
    )
    failures = 0
    generated = 0
    failure_path = output_dir / "failures.jsonl"
    completed = 0
    batch_size = max(args.workers * 4, 1)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while target is None or len(records) < target:
            request_count = (
                batch_size
                if target is None
                else min(batch_size, target - len(records))
            )
            task_batch = list(islice(pending, request_count))
            if not task_batch:
                break
            futures = {
                executor.submit(
                    generate_reference_one,
                    args,
                    openai,
                    profile,
                    task,
                    profile_name,
                ): task
                for task in task_batch
            }
            for future in as_completed(futures):
                completed += 1
                task = futures[future]
                try:
                    record = future.result()
                except FatalAPIError:
                    for pending_future in futures:
                        pending_future.cancel()
                    raise
                except Exception as exc:
                    failures += 1
                    append_journal(
                        failure_path,
                        failure_record(
                            "reference", profile_name, task["problem_id"], exc
                        ),
                    )
                    print(
                        f"Reference {profile_name} [attempted={completed}] failed "
                        f"{task['problem_id']}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                records[record["problem_id"]] = record
                append_journal(partial_path, record)
                generated += 1
                print(
                    f"Reference {profile_name} [success={len(records)}"
                    f"/{target if target is not None else '?'}] saved "
                    f"{task['problem_id']}"
                )

    ordered = sorted_records(records.values())
    write_sft_json(output_path, ordered)
    reset_paths(partial_path)
    final_only_records = [build_reference_final_only_record(row) for row in ordered]
    write_sft_json(final_only_path, final_only_records)
    print(
        f"Wrote {len(ordered)} rows to {output_path} and "
        f"{len(final_only_records)} rows to {final_only_path} "
        f"({generated} new, {failures} failed)."
    )
    if target is not None and len(ordered) < target:
        print(
            f"Warning: source rows were exhausted for {profile_name}; saved "
            f"{len(ordered)}/{target} successful reference rows.",
            file=sys.stderr,
        )
    return generated, failures


def write_final_only_datasets(
    output_dir: Path, reference_names: Iterable[str]
) -> None:
    for reference_name in reference_names:
        reference_path = reference_output_path(output_dir, reference_name)
        records = combined_reference_records(reference_path)
        if not records:
            raise SystemExit(
                f"No valid reference rows found for '{reference_name}' in "
                f"{reference_path} or its partial journal."
            )
        final_rows = [
            build_reference_final_only_record(row)
            for row in sorted_records(records.values())
        ]
        output_path = final_only_output_path(output_dir, reference_name)
        write_sft_json(output_path, final_rows)
        print(f"Wrote {len(final_rows)} final-only rows to {output_path}")


def generate_reconstruction_dataset(
    args: argparse.Namespace,
    openai: Any,
    reference_name: str,
    reference_profile: dict[str, Any],
    reconstruction_name: str,
    reconstruction_profile: dict[str, Any],
    reference_records: dict[str, dict[str, Any]],
    output_dir: Path,
) -> tuple[int, int]:
    output_path = synthetic_output_path(
        output_dir, reconstruction_name, reference_name
    )
    partial_path = journal_path(output_path)
    if args.overwrite:
        reset_paths(output_path, partial_path)
    records = combined_synthetic_records(output_path)
    target = args.limit
    pending_references = [
        row
        for row in sorted_records(reference_records.values())
        if row["problem_id"] not in records
    ]
    print(
        f"Reconstruct {reconstruction_name} syn from {reference_name}: "
        f"available_references={len(reference_records)} "
        f"cached_successes={len(records)} "
        f"target_successes={target if target is not None else len(reference_records)}"
    )
    tasks = [
        {
            "problem_id": row["problem_id"],
            "problem": row["problem"],
            "source_index": row.get("source_index", 0),
            **{
                field: row[field]
                for field in (
                    "problem_type",
                    "question_type",
                    "source",
                    "correctness_count",
                )
                if field in row
            },
        }
        for row in pending_references
    ]
    reference_by_id = {
        row["problem_id"]: row for row in pending_references
    }
    generated = 0
    failures = 0
    failure_path = output_dir / "failures.jsonl"
    completed = 0
    batch_size = max(args.workers * 4, 1)
    pending_task_iterator = iter(tasks)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while target is None or len(records) < target:
            request_count = (
                batch_size
                if target is None
                else min(batch_size, target - len(records))
            )
            task_batch = list(islice(pending_task_iterator, request_count))
            if not task_batch:
                break
            futures = {
                executor.submit(
                    generate_synthetic_one,
                    args,
                    openai,
                    reference_profile,
                    reconstruction_profile,
                    task,
                    reference_by_id[task["problem_id"]],
                    reference_name,
                    reconstruction_name,
                ): task
                for task in task_batch
            }
            for future in as_completed(futures):
                completed += 1
                task = futures[future]
                try:
                    record = future.result()
                except FatalAPIError:
                    for pending_future in futures:
                        pending_future.cancel()
                    raise
                except Exception as exc:
                    failures += 1
                    append_journal(
                        failure_path,
                        failure_record(
                            "reconstruct",
                            reconstruction_name,
                            task["problem_id"],
                            exc,
                            reference_name,
                        ),
                    )
                    print(
                        f"Reconstruct {reconstruction_name} syn from {reference_name} "
                        f"[attempted={completed}] failed {task['problem_id']}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                records[record["problem_id"]] = record
                append_journal(partial_path, record)
                generated += 1
                print(
                    f"Reconstruct {reconstruction_name} syn from {reference_name} "
                    f"[success={len(records)}/"
                    f"{target if target is not None else len(reference_records)}] "
                    f"saved {task['problem_id']}"
                )

    ordered = sorted_records(records.values())
    write_sft_json(output_path, ordered)
    reset_paths(partial_path)
    print(
        f"Wrote {len(ordered)} rows to {output_path} "
        f"({generated} new, {failures} failed)."
    )
    expected = target if target is not None else len(reference_records)
    if len(ordered) < expected:
        print(
            f"Warning: {reconstruction_name} syn from {reference_name} saved "
            f"{len(ordered)}/{expected} successful rows because only "
            f"{len(reference_records)} reference rows were available.",
            file=sys.stderr,
        )
    return generated, failures


def source_tasks(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    """Stream every usable candidate so failures can be replaced by later rows."""
    return pending_tasks(args, load_dataset_rows(args), set(), max_tasks=None)


def selected_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Materialize only the requested prefix for a bounded dry-run preview."""
    return list(
        pending_tasks(
            args,
            load_dataset_rows(args),
            set(),
            max_tasks=args.limit,
        )
    )


def print_matrix_dry_run(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    reference_profiles: dict[str, dict[str, Any]],
    reconstruction_profiles: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    print(
        f"Dry run: {len(tasks)} problems, {len(reference_profiles)} reference "
        f"models, {len(reconstruction_profiles)} reconstruction models, "
        f"{len(tasks) * len(reference_profiles)} reference calls, "
        f"{len(tasks) * len(reference_profiles) * len(reconstruction_profiles)} "
        "reconstruction calls."
    )
    for name in reference_profiles:
        print(f"  {name} ref -> {reference_output_path(output_dir, name)}")
        print(f"  {name} final only -> {final_only_output_path(output_dir, name)}")
    for reference_name in reference_profiles:
        for reconstruction_name in reconstruction_profiles:
            print(
                f"  {reconstruction_name} syn from {reference_name} -> "
                f"{synthetic_output_path(output_dir, reconstruction_name, reference_name)}"
            )
    if tasks:
        print(
            json.dumps(
                {
                    "first_problem_id": tasks[0]["problem_id"],
                    "reference_messages": reference_prompt(
                        tasks[0]["problem"],
                        final_only=reference_is_final_only(
                            next(iter(reference_profiles.values()))
                        ),
                    ),
                    "reconstruction_messages": synthetic_prompt(
                        tasks[0]["problem"], "<REFERENCE_FINAL_OUTPUT>"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    print("No model calls or output writes were made.")


def main() -> None:
    args = parse_args()
    if args.stage == "synthetic":
        args.stage = "reconstruct"
    output_dir = resolve_path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    needs_reconstruction_profiles = (
        args.stage in {"all", "reconstruct"} or args.dry_run
    )
    requested_reconstruction_names = (
        args.reconstruction_profiles
        if needs_reconstruction_profiles and not args.local_reconstruction_only
        else []
    )
    all_configured_names = list(
        dict.fromkeys([*args.reference_profiles, *requested_reconstruction_names])
    )
    configured_profiles = load_profiles(args.config, all_configured_names)
    reference_profiles = {
        name: configured_profiles[name] for name in args.reference_profiles
    }
    reconstruction_profiles = {
        name: configured_profiles[name] for name in requested_reconstruction_names
    }
    if needs_reconstruction_profiles and not args.no_local_reconstruction:
        local_name = validate_output_name(args.local_profile_name)
        if local_name in reconstruction_profiles:
            raise SystemExit(
                f"Local profile name '{local_name}' conflicts with a configured "
                "--reconstruction-profiles entry."
            )
        local_profile = synthetic_profile_from_args(args)
        local_profile["profile_name"] = local_name
        reconstruction_profiles = {
            local_name: local_profile,
            **reconstruction_profiles,
        }

    tasks: list[dict[str, Any]] = []
    if args.dry_run:
        tasks = selected_tasks(args)
        if not tasks:
            raise SystemExit("No usable OpenR1-Math problems were selected.")

    if args.dry_run:
        print_matrix_dry_run(
            args,
            tasks,
            reference_profiles,
            reconstruction_profiles,
            output_dir,
        )
        return

    if args.stage == "final-only":
        write_final_only_datasets(output_dir, reference_profiles)
        return

    openai = require_package("openai")
    total_generated = 0
    total_failures = 0
    if args.stage in {"all", "reference"}:
        for profile_name, profile in reference_profiles.items():
            generated, failures = generate_reference_dataset(
                args,
                openai,
                profile_name,
                profile,
                source_tasks(args),
                output_dir,
            )
            total_generated += generated
            total_failures += failures
        if args.stage == "reference":
            print(
                f"Reference stage complete: {total_generated} new rows, "
                f"{total_failures} failed calls."
            )
            return

    if args.stage in {"all", "reconstruct"}:
        for reference_name, reference_profile in reference_profiles.items():
            reference_path = reference_output_path(output_dir, reference_name)
            reference_records = combined_reference_records(reference_path)
            if not reference_records:
                raise SystemExit(
                    f"No valid reference rows found for '{reference_name}' in "
                    f"{reference_path}. Run --stage reference first."
                )
            # Keep final-only data synchronized even when reconstruction is run alone.
            final_rows = [
                build_reference_final_only_record(row)
                for row in sorted_records(reference_records.values())
            ]
            write_sft_json(
                final_only_output_path(output_dir, reference_name), final_rows
            )
            for reconstruction_name, reconstruction_profile in (
                reconstruction_profiles.items()
            ):
                generated, failures = generate_reconstruction_dataset(
                    args=args,
                    openai=openai,
                    reference_name=reference_name,
                    reference_profile=reference_profile,
                    reconstruction_name=reconstruction_name,
                    reconstruction_profile=reconstruction_profile,
                    reference_records=reference_records,
                    output_dir=output_dir,
                )
                total_generated += generated
                total_failures += failures

    print(
        f"Pipeline complete: {total_generated} new rows, "
        f"{total_failures} failed calls. Outputs: {output_dir}"
    )


if __name__ == "__main__":
    main()
