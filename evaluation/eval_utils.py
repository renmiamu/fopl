from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
DEFAULT_REFERENCE_FIELDS = ("reference_cot", "real_cot", "target_cot", "cot")
DEFAULT_CANDIDATE_FIELDS = ("synthetic_cot", "cot", "chosen", "rejected")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def safe_filename_part(value: str) -> str:
    cleaned: list[str] = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif cleaned and cleaned[-1] != "_":
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "unknown"


def load_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise SystemExit(f"{path} must contain a JSON list or object.")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise SystemExit(f"{path}[{index}] must be a JSON object.")
        rows.append(item)
    return rows


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
            cot = result[cot_start:close_after_open].strip()
            final_output = result[close_after_open + len(THINK_CLOSE) :].strip()
            return cot, final_output
        return result[cot_start:].strip(), ""
    if close_index != -1:
        cot = result[:close_index].strip()
        final_output = result[close_index + len(THINK_CLOSE) :].strip()
        return cot, final_output
    return "", result


def boxed_contents(text: str) -> list[str]:
    """Return balanced contents from every ``\\boxed{...}`` occurrence."""
    contents: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    contents.append(text[start:index].strip())
                    break
    return contents


def extract_boxed_answer(text: str) -> str | None:
    """Extract the final balanced boxed answer from visible output."""
    answers = boxed_contents(text.rsplit(THINK_CLOSE, 1)[-1])
    return answers[-1] if answers and answers[-1] else None


def normalize_boxed_answer(answer: str) -> str:
    """Conservatively normalize LaTeX formatting for exact agreement checks."""
    normalized = answer.strip().strip("$").strip()
    normalized = normalized.replace("−", "-").replace("–", "-")
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(r"\dfrac", r"\frac")
    normalized = normalized.replace(r"\tfrac", r"\frac")
    normalized = re.sub(r"\\(?:,|!|;|:|quad|qquad)", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.casefold()


def normalized_boxed_answer_from_record(item: dict[str, Any]) -> str | None:
    """Read and normalize a record's boxed final answer."""
    final_output = text_field(item, "final_output", "answer", "final_answer")
    if not final_output:
        result = item.get("result")
        if isinstance(result, str):
            _, final_output = extract_cot_and_final(result)
    answer = extract_boxed_answer(final_output)
    return normalize_boxed_answer(answer) if answer is not None else None


def text_field(item: dict[str, Any], *fields: str) -> str:
    for field in fields:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def record_key(index: int, item: dict[str, Any], match_key: str) -> str:
    value = item.get(match_key)
    if value is not None and str(value).strip():
        return str(value).strip()
    return f"index:{index}"


def infer_reference_field(item: dict[str, Any], requested: str | None) -> str | None:
    if requested:
        value = item.get(requested)
        return requested if isinstance(value, str) and value.strip() else None
    for field in DEFAULT_REFERENCE_FIELDS:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return field
    result = item.get("result")
    if isinstance(result, str):
        cot, _ = extract_cot_and_final(result)
        if cot:
            return "result_cot"
    return None


def cot_from_field(item: dict[str, Any], field: str) -> str:
    if field == "result_cot":
        result = item.get("result")
        if isinstance(result, str):
            cot, _ = extract_cot_and_final(result)
            return cot
        return ""
    value = item.get(field)
    return value.strip() if isinstance(value, str) else ""


def candidate_cots(
    item: dict[str, Any],
    candidate_fields: list[str] | None,
    reference_field: str | None,
) -> list[tuple[str, str]]:
    fields = candidate_fields or [
        field
        for field in DEFAULT_CANDIDATE_FIELDS
        if field != reference_field and item.get(field)
    ]

    candidates: list[tuple[str, str]] = []
    for field in fields:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            candidates.append((field, value.strip()))

    result = item.get("result")
    if not candidates and reference_field != "result_cot" and isinstance(result, str):
        cot, _ = extract_cot_and_final(result)
        if cot:
            candidates.append(("result_cot", cot))
    return candidates


def build_reference_map(
    reference_path: Path,
    match_key: str,
    reference_field_arg: str | None,
) -> dict[str, tuple[int, dict[str, Any], str, str]]:
    reference_map: dict[str, tuple[int, dict[str, Any], str, str]] = {}
    for index, item in enumerate(load_json_list(reference_path)):
        key = record_key(index, item, match_key)
        if key in reference_map:
            raise SystemExit(
                f"Duplicate reference key {key!r} in {reference_path}; "
                f"use --match-key with unique values."
            )
        field = infer_reference_field(item, reference_field_arg)
        if not field:
            continue
        reference_map[key] = (index, item, field, cot_from_field(item, field))
    return reference_map


def load_candidate_rows(
    input_paths: list[Path],
    limit: int | None,
) -> list[tuple[Path, int, dict[str, Any]]]:
    rows: list[tuple[Path, int, dict[str, Any]]] = []
    for path in input_paths:
        for index, item in enumerate(load_json_list(path)):
            rows.append((path, index, item))
            if limit is not None and len(rows) >= limit:
                return rows
    return rows
