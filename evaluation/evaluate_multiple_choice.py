from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Iterable
import unicodedata

try:
    from .evaluate_gsm8k import (
        check_server,
        request_seed,
        require_package,
    )
except ImportError:  # Support direct execution through a benchmark wrapper.
    from evaluate_gsm8k import (  # type: ignore[no-redef]
        check_server,
        request_seed,
        require_package,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache/huggingface"
PARSER_VERSION = 6
# Prompt changes must invalidate resume fingerprints so old generations are
# never mixed with the solve-only evaluation.
PROMPT_VERSION = 4
MAX_RESPONSE_CHARS = 262_144
MAX_BOX_CONTENT_CHARS = 8192
OPTION_LABELS = tuple(chr(ord("A") + index) for index in range(26))


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    display_name: str
    dataset: str
    dataset_config: str | None
    split: str
    output: Path
    system_prompt: str
    default_max_tokens: int = 512


MMLU_STEM_SUBJECTS = {
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
}
MMLU_HUMANITIES_SUBJECTS = {
    "formal_logic",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "prehistory",
    "professional_law",
    "world_religions",
}
MMLU_SOCIAL_SCIENCES_SUBJECTS = {
    "econometrics",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_sexuality",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(spec: BenchmarkSpec) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Evaluate an OpenAI-compatible vLLM model on {spec.display_name} "
            "with robust multiple-choice answer parsing."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Served model name exposed by the vLLM /v1/models endpoint.",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--dataset", default=spec.dataset)
    parser.add_argument("--dataset-config", default=spec.dataset_config)
    parser.add_argument("--split", default=spec.split)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--hf-endpoint",
        help="Alternative Hugging Face endpoint, e.g. https://hf-mirror.com.",
    )
    parser.add_argument("--output", type=Path, default=spec.output)
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help=(
            "Atomically refresh the full JSON output after this many new rows. "
            "Every row is still appended immediately to a JSONL checkpoint."
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-tokens",
        "--max-new-tokens",
        dest="max_tokens",
        type=int,
        default=spec.default_max_tokens,
        help="Maximum generated tokens.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generations and regrade cached responses when needed.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
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


def clean_text(value: Any) -> str:
    return str(value).replace("\x00", "").strip()


def casefold_field(item: dict[str, Any], *names: str) -> Any:
    lookup = {str(key).casefold(): value for key, value in item.items()}
    for name in names:
        if name.casefold() in lookup:
            return lookup[name.casefold()]
    return None


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def label_for_index(index: int) -> str:
    if not 0 <= index < len(OPTION_LABELS):
        raise ValueError(f"Choice index {index} is outside the supported range")
    return OPTION_LABELS[index]


def answer_index(value: Any, original_labels: list[str], num_choices: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value < num_choices:
            return value
    normalized = clean_text(value).casefold()
    for index, label in enumerate(original_labels):
        if normalized == clean_text(label).casefold():
            return index
    if re.fullmatch(r"\d+", normalized):
        integer = int(normalized)
        # Datasets such as MMLU/HellaSwag expose zero-based class indices when
        # they do not provide source labels. ARC provides explicit labels and
        # commonly uses the one-based strings "1"-"4".
        if not original_labels and 0 <= integer < num_choices:
            return integer
        if 1 <= integer <= num_choices:
            return integer - 1
        if 0 <= integer < num_choices:
            return integer
    if len(normalized) == 1 and normalized.upper() in OPTION_LABELS[:num_choices]:
        return ord(normalized.upper()) - ord("A")
    raise ValueError(f"Cannot map answer {value!r} to {num_choices} choices")


def make_row(
    benchmark: str,
    index: int,
    raw_id: Any,
    question: Any,
    choice_texts: Iterable[Any],
    gold_index: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    texts = [clean_text(text) for text in choice_texts]
    question_text = clean_text(question)
    if not question_text:
        raise ValueError("question is empty")
    if len(texts) < 2 or len(texts) > len(OPTION_LABELS):
        raise ValueError(f"expected 2-26 choices, found {len(texts)}")
    if any(not text for text in texts):
        raise ValueError("one or more choices are empty")
    if not 0 <= gold_index < len(texts):
        raise ValueError(f"gold index {gold_index} is invalid")
    identifier = clean_text(raw_id) or str(index)
    row = {
        "problem_id": f"{benchmark}/{identifier}/{index}",
        "order": index,
        "question": question_text,
        "choices": [
            {"label": label_for_index(choice_index), "text": text}
            for choice_index, text in enumerate(texts)
        ],
        "gold_answer": label_for_index(gold_index),
    }
    if metadata:
        row.update({key: value for key, value in metadata.items() if value is not None})
    return row


def normalize_mmlu(item: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    del seed
    choices = list(item.get("choices") or [])
    gold_index = answer_index(item.get("answer"), [], len(choices))
    subject = clean_text(item.get("subject")) or "unknown"
    return make_row(
        "mmlu",
        index,
        f"{subject}/{index}",
        item.get("question"),
        choices,
        gold_index,
        {"subject": subject, "category": mmlu_category(subject)},
    )


def unpack_arc_choices(value: Any) -> tuple[list[str], list[str]]:
    if isinstance(value, dict):
        labels = list(value.get("label") or [])
        texts = list(value.get("text") or [])
    elif isinstance(value, list):
        labels = [choice.get("label") for choice in value]
        texts = [choice.get("text") for choice in value]
    else:
        raise ValueError("choices must be a dictionary or list")
    if len(labels) != len(texts):
        raise ValueError("choice labels and texts have different lengths")
    return [clean_text(label) for label in labels], [clean_text(text) for text in texts]


def normalize_arc(item: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    del seed
    labels, texts = unpack_arc_choices(item.get("choices"))
    gold_index = answer_index(item.get("answerKey"), labels, len(texts))
    return make_row(
        "arc_challenge",
        index,
        item.get("id", index),
        item.get("question"),
        texts,
        gold_index,
    )


def normalize_openbookqa(
    item: dict[str, Any], index: int, seed: int
) -> dict[str, Any]:
    del seed
    labels, texts = unpack_arc_choices(item.get("choices"))
    normalized_labels = [label.casefold() for label in labels]
    if len(set(normalized_labels)) != len(labels):
        raise ValueError("choice labels must be unique")
    if clean_text(item.get("answerKey")).casefold() not in normalized_labels:
        raise ValueError("answerKey does not match a choice label")
    gold_index = answer_index(item.get("answerKey"), labels, len(texts))
    return make_row(
        "openbookqa",
        index,
        item.get("id", index),
        item.get("question_stem"),
        texts,
        gold_index,
    )


def preprocess_hellaswag(value: Any) -> str:
    text = clean_text(value).replace(" [title]", ". ")
    text = re.sub(r"\[[^]]*\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_hellaswag(
    item: dict[str, Any], index: int, seed: int
) -> dict[str, Any]:
    del seed
    choices = [preprocess_hellaswag(value) for value in item.get("endings") or []]
    gold_index = answer_index(item.get("label"), [], len(choices))
    raw_id = item.get("ind", item.get("source_id", index))
    return make_row(
        "hellaswag",
        index,
        raw_id,
        preprocess_hellaswag(item.get("ctx")),
        choices,
        gold_index,
        {"activity_label": clean_text(item.get("activity_label"))},
    )


def normalize_gpqa(item: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    question = casefold_field(item, "Question")
    correct = casefold_field(item, "Correct Answer")
    incorrect = [
        casefold_field(item, f"Incorrect Answer {number}") for number in range(1, 4)
    ]
    if any(value is None for value in [question, correct, *incorrect]):
        raise ValueError("expected Question, Correct Answer, and three Incorrect Answers")

    shuffled = [(clean_text(correct), True)] + [
        (clean_text(value), False) for value in incorrect
    ]
    question_hash = stable_digest(clean_text(question))[:16]
    shuffle_seed = int(stable_digest(f"{seed}:gpqa:{question_hash}")[:16], 16)
    random.Random(shuffle_seed).shuffle(shuffled)
    gold_index = next(i for i, (_, is_correct) in enumerate(shuffled) if is_correct)
    domain = casefold_field(item, "High-level domain", "Subdomain")
    return make_row(
        "gpqa",
        index,
        question_hash,
        question,
        [text for text, _ in shuffled],
        gold_index,
        {"domain": clean_text(domain) if domain is not None else "unknown"},
    )


NORMALIZERS = {
    "mmlu": normalize_mmlu,
    "arc_challenge": normalize_arc,
    "openbookqa": normalize_openbookqa,
    "hellaswag": normalize_hellaswag,
    "gpqa": normalize_gpqa,
}


def mmlu_category(subject: str) -> str:
    if subject in MMLU_STEM_SUBJECTS:
        return "stem"
    if subject in MMLU_HUMANITIES_SUBJECTS:
        return "humanities"
    if subject in MMLU_SOCIAL_SCIENCES_SUBJECTS:
        return "social_sciences"
    return "other"


def load_split(
    datasets: Any,
    spec: BenchmarkSpec,
    args: argparse.Namespace,
    split: str,
) -> list[dict[str, Any]]:
    load_kwargs: dict[str, Any] = {
        "split": split,
        "cache_dir": str(resolve_path(args.cache_dir)),
    }
    try:
        if args.dataset_config:
            loaded = datasets.load_dataset(
                args.dataset, args.dataset_config, **load_kwargs
            )
        else:
            loaded = datasets.load_dataset(args.dataset, **load_kwargs)
    except Exception as exc:
        gate_hint = (
            " GPQA is gated: accept its Hugging Face terms and run "
            "`hf auth login` first."
            if spec.key == "gpqa"
            else ""
        )
        config_text = f"/{args.dataset_config}" if args.dataset_config else ""
        raise SystemExit(
            f"Could not load {args.dataset}{config_text} split {split!r}: "
            f"{type(exc).__name__}: {exc}.{gate_hint}"
        ) from exc

    normalizer = NORMALIZERS[spec.key]
    rows: list[dict[str, Any]] = []
    for index, raw_item in enumerate(loaded):
        try:
            row = normalizer(dict(raw_item), index, args.seed)
            original_id = row["problem_id"].split("/", 1)[-1]
            row["problem_id"] = f"{spec.key}/{split}/{original_id}"
            rows.append(row)
        except (TypeError, ValueError, StopIteration) as exc:
            raise SystemExit(
                f"Invalid {spec.display_name} row {index} in split {split!r}: {exc}"
            ) from exc
    if not rows:
        raise SystemExit(
            f"No rows loaded from {args.dataset} split {split!r}."
        )
    return rows


def load_benchmark(
    datasets: Any, spec: BenchmarkSpec, args: argparse.Namespace
) -> list[dict[str, Any]]:
    rows = load_split(datasets, spec, args, args.split)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(
        f"Loaded {len(rows)} {spec.display_name} evaluation rows from "
        f"{args.split!r} for zero-shot evaluation.",
        flush=True,
    )
    return rows


def build_prompt(row: dict[str, Any]) -> str:
    lines = [f"Question: {row['question']}"]
    lines.extend(
        f"{choice['label']}. {choice['text']}" for choice in row["choices"]
    )
    lines.append(
        r"Solve the problem. Put only the final option letter inside \boxed{}."
    )
    return "\n".join(lines)


def build_messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
    """Build zero-shot chat messages without an example or a gold answer."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def boxed_contents(text: str) -> list[str]:
    # Accept standard \boxed and the closely related \fbox. A missing-backslash
    # variant is handled separately only at the response suffix, where it cannot
    # accidentally match prose in the reasoning body.
    commands = {
        match.end() - 1: match.start()
        for match in re.finditer(r"\\+(?:boxed|fbox)\s*\{", text, re.I)
    }
    # Scan braces once instead of rescanning the entire suffix for every box.
    # Runaway output containing thousands of unclosed boxes must not hang.
    stack: list[int] = []
    completed: list[tuple[int, str]] = []
    for index, char in enumerate(text):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            opening = stack.pop()
            if opening in commands and index - opening - 1 <= MAX_BOX_CONTENT_CHARS:
                completed.append((commands[opening], text[opening + 1:index].strip()))
    # Preserve command-start order, including inner boxes in nested macros.
    return [content for _, content in sorted(completed)]


def visible_answer_text(response: str) -> str:
    cleaned = unicodedata.normalize("NFKC", str(response)).replace("\x00", "")
    think_end = list(re.finditer(r"</(?:think|analysis)\s*>", cleaned, re.IGNORECASE))
    if think_end:
        suffix = cleaned[think_end[-1].end() :]
        if suffix.strip():
            cleaned = suffix
    if len(cleaned) > MAX_RESPONSE_CHARS:
        cleaned = cleaned[-MAX_RESPONSE_CHARS:]
    return cleaned.strip()


def normalized_label(value: str, valid_labels: set[str]) -> str | None:
    text = unicodedata.normalize("NFKC", clean_text(value))
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    text = re.sub(r"^\*+|\*+$", "", text).strip()
    text = text.strip("$`~")
    # Models often add harmless LaTeX wrappers or a short answer marker inside
    # the box. Unwrap only when the macro spans the whole candidate so text in
    # an explanation cannot be mistaken for the final label.
    for _ in range(3):
        latex_match = re.fullmatch(
            r"\\(?:text|textrm|textbf|textsf|mathrm|mathbf|mathsf|mathtt|"
            r"operatorname|rm)\s*\{\s*(.*?)\s*\}",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not latex_match:
            break
        text = latex_match.group(1).strip()
    text = re.sub(
        r"^(?:(?:final|correct)\s+)?(?:option|choice|answer)|^(?:最终答案|正确答案|答案|选项)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:is|为|是)\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("()[]{}<>.,;:：。！!").strip()
    label = text.upper()
    if label in valid_labels:
        return label
    if label.isdigit():
        index = int(label) - 1
        ordered = sorted(valid_labels)
        if 0 <= index < len(ordered):
            return ordered[index]
    return None


def strip_text_wrapper(value: str) -> str:
    """Remove presentation-only wrappers before exact option-text matching."""
    text = unicodedata.normalize("NFKC", clean_text(value))
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text).strip(" $`~")
    for _ in range(3):
        match = re.fullmatch(
            r"\\(?:text|textrm|textbf|textsf|mathrm|mathbf|mathsf|mathtt|"
            r"operatorname|rm)\s*\{\s*(.*?)\s*\}",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            break
        text = match.group(1).strip()
    return text


def normalized_option_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value)).casefold()
    text = re.sub(r"^[\s*`_#>-]+|[\s*`_#<.-]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.,;:：。!?！?()[]{}\"'")


def match_choice_text(line: str, choices: list[dict[str, str]]) -> str | None:
    candidate = normalized_option_text(line)
    if not candidate:
        return None
    matches = [
        choice["label"]
        for choice in choices
        if candidate == normalized_option_text(choice["text"])
    ]
    return matches[0] if len(matches) == 1 else None


def boxed_answer(
    content: str,
    choices: list[dict[str, str]],
    valid_labels: set[str],
) -> tuple[str | None, str]:
    label = normalized_label(content, valid_labels)
    if label is not None:
        return label, "boxed"

    unwrapped = strip_text_wrapper(content)
    text_label = match_choice_text(unwrapped, choices)
    if text_label is not None:
        return text_label, "boxed_choice_text"

    # Also accept "B. <option text>" inside a box, but only when both the label
    # and the accompanying text agree with the displayed option. This avoids
    # accepting an arbitrary first letter from a boxed explanation.
    labelled = re.fullmatch(
        r"\s*([A-Za-z])\s*[\).:：\-]\s*(.+?)\s*", unwrapped, re.DOTALL
    )
    if labelled:
        candidate = normalized_label(labelled.group(1), valid_labels)
        if candidate is not None:
            expected = next(
                choice["text"] for choice in choices if choice["label"] == candidate
            )
            if normalized_option_text(labelled.group(2)) == normalized_option_text(
                expected
            ):
                return candidate, "boxed_label_and_text"
    return None, "not_found"


def json_answer(
    response: str, valid_labels: set[str]
) -> str | None:
    candidate = response.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", candidate, re.IGNORECASE | re.DOTALL
    )
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError, RecursionError):
        return None
    if not isinstance(value, dict):
        return None
    normalized_keys = {
        str(key).casefold().replace("-", "_").replace(" ", "_"): item
        for key, item in value.items()
    }
    for key in (
        "final_answer", "answer", "choice", "option",
        "final_option", "correct_answer", "correct_option", "selected_option",
    ):
        item = normalized_keys.get(key)
        if isinstance(item, (str, int)) and not isinstance(item, bool):
            label = normalized_label(str(item), valid_labels)
            if label is not None:
                return label
    return None


def normalize_answer_presentation(text: str) -> str:
    """Remove presentation markup, never reasoning or answer-bearing words."""
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"(?<!\w)[*_]([A-Z1-9])[*_](?!\w)", r"\1", text, flags=re.I)
    for _ in range(4):
        previous = text
        text = re.sub(
            r"\\textcolor\s*\{[^{}]*\}\s*\{([^{}]*)\}",
            r"\1", text, flags=re.IGNORECASE,
        )
        text = re.sub(r"\\color\s*\{[^{}]*\}", "", text, flags=re.IGNORECASE)
        text = re.sub(
            r"\\(?:text|textrm|textbf|textsf|textit|mathrm|mathbf|mathsf|"
            r"mathtt|mathit|operatorname|underline|emph|mbox|rm)\s*\{([^{}]*)\}",
            r"\1", text, flags=re.IGNORECASE,
        )
        if text == previous:
            break
    return re.sub(r"\\([()\[\]])", r"\1", text)


def formatted_answer_fallback(
    visible: str, choices: list[dict[str, str]], valid_labels: set[str]
) -> tuple[str | None, str]:
    """Recover explicit formats only after the established parser has failed."""
    formatted = normalize_answer_presentation(visible)
    for content in reversed(boxed_contents(formatted)):
        answer, _ = boxed_answer(content.strip(" $\"'"), choices, valid_labels)
        if answer is not None:
            return answer, "formatted_boxed"

    alternate_boxes = list(re.finditer(
        r"\\+(?:boxed|fbox)\s*(?:\(\s*([A-Z]|\d{1,2})\s*\)|"
        r"\[\s*([A-Z]|\d{1,2})\s*\])",
        formatted, re.IGNORECASE,
    ))
    if alternate_boxes:
        selected = next(group for group in alternate_boxes[-1].groups() if group)
        answer = normalized_label(selected, valid_labels)
        if answer is not None:
            return answer, "alternative_box_delimiters"

    # Accept a final JSON object/fence after prose, not an example object buried
    # in reasoning. Limit scanning so pathological brace-heavy output is cheap.
    tail = visible[-4096:]
    fence = re.search(r"```(json|text|latex|math)?\s*(.*?)\s*```\s*$", tail, re.I | re.S)
    if fence:
        answer = json_answer(fence.group(2), valid_labels)
        if answer is not None:
            return answer, "trailing_structured_json"
        content = normalize_answer_presentation(fence.group(2)).strip(" $\"'")
        answer, _ = boxed_answer(content, choices, valid_labels)
        if answer is not None:
            return answer, "fenced_final_answer"
    decoder = json.JSONDecoder()
    starts = [match.start() for match in re.finditer(r"\{", tail)][-256:]
    for start in reversed(starts):
        try:
            _, end = decoder.raw_decode(tail[start:])
        except (ValueError, RecursionError):
            continue
        if tail[start + end:].strip():
            continue
        answer = json_answer(tail[start:start + end], valid_labels)
        if answer is not None:
            return answer, "trailing_structured_json"

    xml_answers = list(re.finditer(
        r"<(answer|final_answer|final)\s*>\s*([^<>]{1,1024}?)\s*</\1\s*>",
        formatted, re.IGNORECASE,
    ))
    if xml_answers:
        content = xml_answers[-1].group(2).strip(" $\"'")
        answer, _ = boxed_answer(content, choices, valid_labels)
        if answer is not None:
            return answer, "xml_final_answer"

    # Capture out-of-range labels too: an explicit final E must not fall back
    # to an earlier B. Boundaries use ASCII rather than \b so C项 is supported.
    value = r"\s*(?:option|choice|选项)?\s*[\(\[\{<\"']?\s*(?P<label>[A-Z]|\d{1,2})(?![A-Za-z0-9])"
    copula = r"(?:is|(?:should|would|must|will|might)\s+be|应(?:该)?为|应该是|为|是)"
    context = (
        r"(?:\s+(?:to\s+(?:this|the)\s+question|"
        r"from\s+(?:the\s+)?available\s+(?:choices|options)|"
        r"among\s+(?:the\s+)?(?:choices|options)(?:\s+provided)?))?"
    )
    marker = (
        r"(?<![A-Za-z])(?:"
        r"(?:final(?:ized)?|最终选择)\s*[:：=]|"
        r"(?:(?:(?:final|correct|best|closest|boxed\s+final)[\s_-]+"
        r"(?:answer|choice|option)|(?:the\s+)?answer|"
        r"最终答案|正确答案|答案))" + context +
        r"\s*(?:" + copula + r"\s*[:：=]?|[:：=])\s*(?:likely\s+)?"
        r")"
    )
    patterns = [
        (re.compile(marker + value, re.I), "formatted_final_marker"),
        (re.compile(r"^\s*(?:answer|答案)\s+" + value, re.I | re.M), "formatted_answer_line"),
        (re.compile(r"\bI\s+(?:choose|select|pick)" + value, re.I), "explicit_selection"),
        (re.compile(r"\\begin\s*\{(?:answer|finalanswer)\}" + value, re.I), "latex_answer_environment"),
        (re.compile(
            r"(?<![A-Za-z0-9])(?:option\s+|choice\s+)?"
            r"(?P<label>[A-Z]|\d{1,2})(?![A-Za-z0-9])\s*"
            r"is\s+(?:the\s+)?(?:correct|best|most\s+appropriate)\s+"
            r"(?:answer|choice|option)(?![A-Za-z])", re.I,
        ), "positive_answer_statement"),
    ]
    candidates: list[tuple[int, str | None, str]] = []
    for pattern, method in patterns:
        for match in pattern.finditer(formatted):
            # Do not promote hypothetical answers or instructions/examples.
            prefix = re.split(r"[\n.!?;]", formatted[max(0, match.start() - 100):match.start()])[-1]
            if re.search(
                r"\b(?:if|suppose|assuming|whether|example|print|output)\b|"
                r"\bnot\s+(?:the\s+)?$|(?:如果|假设|例如)", prefix, re.I,
            ):
                continue
            # A fabricated second problem may carry its own five-choice list.
            # Do not recover a label from that foreign option block when the
            # actual benchmark exposes a different set of labels.
            preceding_options = re.findall(
                r"(?m)^\s*([A-Z])\s*(?:[.):：]\s*|\s+)\S[^\n]*$",
                formatted[max(0, match.start() - 1024):match.start()],
            )
            if len(preceding_options) >= 3 and any(
                label not in valid_labels for label in preceding_options
            ):
                continue
            after = formatted[match.end():match.end() + 1024]
            ambiguous = re.match(
                r"\s*[)\]}>\"']?\s*(?:or|and|/|或者|或|与|以及|-)\s*"
                r"(?:option\s+|choice\s+)?[\[(]?\s*(?:[A-Z](?![A-Za-z])|\d)",
                after, re.I,
            )
            # An echoed answer/options block is not a selected final answer.
            option_lines = re.findall(r"(?m)^\s*[A-Z]\s*[.):：]\s*\S", after)
            if len(option_lines) >= 2:
                continue
            answer = None if ambiguous else normalized_label(match.group("label"), valid_labels)
            candidates.append((match.start(), answer, method))
    if candidates:
        _, answer, method = max(candidates, key=lambda candidate: candidate[0])
        if answer is not None:
            return answer, method
    return None, "not_found"


def extract_choice_answer(
    response: str, choices: list[dict[str, str]]
) -> tuple[str | None, str]:
    visible = visible_answer_text(response)
    # Markdown emphasis frequently wraps either the marker or the label. It is
    # formatting rather than answer content, so remove it only for marker regexes.
    marker_visible = visible.replace("**", "").replace("__", "").replace("`", "")
    valid_labels = {choice["label"].upper() for choice in choices}
    label_group = "".join(re.escape(label) for label in sorted(valid_labels))
    numeric_group = "".join(str(index + 1) for index in range(len(valid_labels)))
    answer_group = label_group + numeric_group

    for content in reversed(boxed_contents(visible)):
        label, method = boxed_answer(content, choices, valid_labels)
        if label is not None:
            return label, method

    unescaped_box = re.search(
        r"(?<![A-Za-z\\])(?:boxed|fbox)\s*\{\s*([^{}\r\n]{1,128})\s*\}"
        r"\s*[.!。！]?\s*$",
        visible,
        flags=re.IGNORECASE,
    )
    if unescaped_box:
        label, method = boxed_answer(
            unescaped_box.group(1), choices, valid_labels
        )
        if label is not None:
            return label, f"unescaped_{method}"

    bare_boxes = re.findall(
        rf"\\+(?:boxed|fbox)\s+"
        rf"(?:\\(?:text|textbf|mathrm|mathbf)\s*\{{\s*)?"
        rf"([{answer_group}])(?:\s*\}})?\b",
        visible,
        flags=re.IGNORECASE,
    )
    if bare_boxes:
        return normalized_label(bare_boxes[-1], valid_labels), "boxed_without_braces"

    # Recover a final box whose closing brace was lost to generation truncation.
    # This is deliberately suffix-anchored and accepts only a single label.
    truncated_box = re.search(
        rf"(?<![A-Za-z])\\*(?:boxed|fbox)\s*\{{\s*"
        rf"(?:\\(?:text|textbf|mathrm|mathbf)\s*\{{\s*)?"
        rf"([{answer_group}])\s*\}}?\s*$",
        visible,
        flags=re.IGNORECASE,
    )
    if truncated_box:
        return normalized_label(truncated_box.group(1), valid_labels), "truncated_box"

    structured = json_answer(visible, valid_labels)
    if structured is not None:
        return structured, "structured_json"

    strong_pattern = re.compile(
        rf"(?:final\s+(?:answer|choice)|correct\s+(?:answer|choice|option)|"
        rf"the\s+answer|最终答案|正确答案|答案)\s*"
        rf"(?:is|为|是)?\s*(?:option|choice|选项)?\s*[:：=\-]?\s*"
        rf"[\(\[\{{<]?\s*([{answer_group}])\s*[\)\]\}}>]?\b",
        re.IGNORECASE,
    )
    strong_matches = list(strong_pattern.finditer(marker_visible))
    if strong_matches:
        answer = normalized_label(strong_matches[-1].group(1), valid_labels)
        return answer, "explicit_final_marker"

    line_pattern = re.compile(
        rf"^\s*(?:[*_`#>-]+\s*)?(?:answer|choice|option|选择|选项)\s*"
        rf"(?:is|为|是)?\s*"
        rf"[:：=\-]\s*[\(\[\{{<]?\s*([{answer_group}])\s*[\)\]\}}>]?\b",
        re.IGNORECASE | re.MULTILINE,
    )
    line_matches = list(line_pattern.finditer(marker_visible))
    if line_matches:
        answer = normalized_label(line_matches[-1].group(1), valid_labels)
        return answer, "answer_line"

    inference_pattern = re.compile(
        rf"(?:therefore|thus|hence|所以|因此)\s*[,，:：]?\s*"
        rf"(?:the\s+answer\s+is\s+|(?:option|choice|选项)\s*)?"
        rf"[\(\[\{{<]?\s*([{answer_group}])\s*[\)\]\}}>]?"
        rf"\s*[.!。！]?\s*$",
        re.IGNORECASE,
    )
    inference_match = inference_pattern.search(marker_visible)
    if inference_match:
        answer = normalized_label(inference_match.group(1), valid_labels)
        return answer, "conclusion_suffix"

    selection_pattern = re.compile(
        rf"(?:I\s+)?(?:choose|select|pick)\s*(?:option|choice)?\s*"
        rf"[\(\[\{{<]?\s*([{answer_group}])\s*[\)\]\}}>]?\s*[.!。！]?\s*$|"
        rf"(?:故|所以|因此|综上(?:所述)?)?[，,:：\s]*"
        rf"(?:选择|选)(?:项)?\s*[\(\[\{{<]?\s*([{answer_group}])"
        rf"\s*[\)\]\}}>]?\s*[。.!！]?\s*$|"
        rf"[\(\[\{{<]?\s*([{answer_group}])\s*[\)\]\}}>]?\s+"
        rf"(?:is|为|是)\s+(?:the\s+)?(?:correct|best)\s+"
        rf"(?:answer|choice|option)\s*[.!。！]?\s*$",
        re.IGNORECASE,
    )
    selection_match = selection_pattern.search(marker_visible)
    if selection_match:
        selected = next(group for group in selection_match.groups() if group)
        return normalized_label(selected, valid_labels), "selection_suffix"

    lines = [line.strip() for line in visible.splitlines() if line.strip()]
    if lines:
        final_line = lines[-1]
        label = normalized_label(final_line, valid_labels)
        if label is not None:
            return label, "standalone_final_line"
        prefixed = re.fullmatch(
            rf"\s*[\(\[\{{<]?\s*([{answer_group}])\s*[\)\]\}}>]?"
            rf"\s*[.:：\-]\s*.+",
            final_line,
            flags=re.IGNORECASE,
        )
        is_option_list = sum(
            bool(re.match(r"^[A-Z]\s*[.):：]\s*\S", line, re.I))
            for line in lines[-len(choices):]
        ) >= 2
        if prefixed and len(final_line) <= 512 and not is_option_list:
            answer = normalized_label(prefixed.group(1), valid_labels)
            return answer, "labelled_final_line"
        text_label = match_choice_text(final_line, choices)
        if text_label is not None:
            return text_label, "exact_choice_text"

    marked_text_pattern = re.compile(
        r"(?:final\s+(?:answer|choice)|the\s+answer|answer|最终答案|答案)\s*"
        r"(?:is|为|是)?\s*[:：=\-]?\s*(.+)$",
        re.IGNORECASE | re.MULTILINE,
    )
    marked_text_matches = list(marked_text_pattern.finditer(marker_visible))
    if marked_text_matches:
        tail_line = marked_text_matches[-1].group(1).splitlines()[0].strip()
        text_label = match_choice_text(tail_line, choices)
        if text_label is not None:
            return text_label, "explicit_choice_text"

    suffix = re.search(
        rf"(?:\(\s*([{answer_group}])\s*\)|\[\s*([{answer_group}])\s*\]|"
        rf"\{{\s*([{answer_group}])\s*\}}|<\s*([{answer_group}])\s*>)"
        rf"\s*[.!。！]?\s*$",
        visible,
        flags=re.IGNORECASE,
    )
    if suffix:
        selected = next(group for group in suffix.groups() if group)
        answer = normalized_label(selected, valid_labels)
        return answer, "parenthesized_suffix"

    # Small or unstable models frequently emit a valid conclusion followed by
    # unrelated garbage instead of ending cleanly. Keep these looser patterns
    # after every conservative fallback above, so they can only recover a
    # response that would otherwise remain unparsed.
    conclusion_marker_pattern = re.compile(
        rf"(?:final(?:ized)?[\s_-]+(?:answer|choice)|"
        rf"expected[\s_-]*answer|best\s+(?:answer|choice)|"
        rf"(?:so|thus|hence|therefore)\s*[,，:]?\s*(?:the\s+)?answer)\s*"
        rf"(?:is|为|是)?\s*(?:option|choice|选项)?\s*[:：=\-]?\s*"
        rf"[\(\[\{{<\"']?\s*([{answer_group}])(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    conclusion_marker_matches = list(
        conclusion_marker_pattern.finditer(marker_visible)
    )
    if conclusion_marker_matches:
        answer = normalized_label(
            conclusion_marker_matches[-1].group(1), valid_labels
        )
        return answer, "explicit_conclusion_marker"

    braced_answer_pattern = re.compile(
        rf"(?<![A-Za-z])answer\s*[:：=\-]?\s*\{{\s*"
        rf"(?:\\color\s*\{{[^}}]+\}}\s*)?([{answer_group}])\s*\}}",
        re.IGNORECASE,
    )
    braced_answer_matches = list(braced_answer_pattern.finditer(marker_visible))
    if braced_answer_matches:
        answer = normalized_label(braced_answer_matches[-1].group(1), valid_labels)
        return answer, "braced_answer_marker"

    framework_answer_pattern = re.compile(
        rf"answer\s*:final\s*=>\s*[\"']?([{answer_group}])"
        rf"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    framework_matches = list(framework_answer_pattern.finditer(marker_visible))
    if framework_matches:
        answer = normalized_label(framework_matches[-1].group(1), valid_labels)
        return answer, "framework_answer_marker"

    correctness_pattern = re.compile(
        rf"(?:"
        rf"(?:answer|option|choice|选项)\s*[:：]?\s*"
        rf"[\(\[]?\s*([{answer_group}])\s*[\)\]]?\s*"
        rf"(?:is|为|是)?\s*(?:correct|正确)"
        rf"|"
        rf"(?:so|thus|hence|therefore)?\s*[,，]?\s*"
        rf"[\(\[]?\s*([{answer_group}])\s*[\)\]]?\s*"
        rf"(?:is|为|是)\s+(?:correct|正确)"
        rf"|"
        rf"(?:选项|选择|故选)\s*([{answer_group}])\s*(?:是)?正确"
        rf")",
        re.IGNORECASE,
    )
    correctness_matches = list(correctness_pattern.finditer(marker_visible))
    if correctness_matches:
        selected = next(
            group for group in correctness_matches[-1].groups() if group is not None
        )
        return normalized_label(selected, valid_labels), "correctness_statement"

    reasoning_selection_pattern = re.compile(
        rf"(?:so|thus|hence|therefore)\s*[,，:]?\s*"
        rf"(?:I\s+)?(?:choose|select|pick)\s*(?:option|choice)?\s*"
        rf"[\(\[]?\s*([{answer_group}])\s*[\)\]]?"
        rf"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    reasoning_selection_matches = list(
        reasoning_selection_pattern.finditer(marker_visible)
    )
    if reasoning_selection_matches:
        answer = normalized_label(
            reasoning_selection_matches[-1].group(1), valid_labels
        )
        return answer, "reasoning_selection"
    return formatted_answer_fallback(visible, choices, valid_labels)


def grade_response(response: str, row: dict[str, Any]) -> dict[str, Any]:
    parsed_answer, parse_method = extract_choice_answer(response, row["choices"])
    return {
        "parsed_answer": parsed_answer,
        "parse_method": parse_method,
        "correct": parsed_answer == row["gold_answer"],
    }


def build_client(args: argparse.Namespace) -> Any:
    openai = require_package("openai")
    return openai.OpenAI(
        api_key=args.api_key,
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout,
        max_retries=0,
    )


def request_completion(
    client: Any,
    row: dict[str, Any],
    prompt: str,
    system_prompt: str,
    args: argparse.Namespace,
) -> str:
    messages = build_messages(prompt, system_prompt)
    extra_body = {"repetition_penalty": args.repetition_penalty}
    # vLLM uses -1 rather than 0 for disabled top-k. Omitting the extension is
    # portable across OpenAI-compatible servers and lets --top-k 0 mean off.
    if args.top_k > 0:
        extra_body["top_k"] = args.top_k
    for attempt in range(1, args.retries + 1):
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=messages,
                n=1,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=request_seed(args.seed, row["problem_id"]),
                extra_body=extra_body,
            )
            if len(completion.choices) != 1:
                raise RuntimeError(
                    f"vLLM returned {len(completion.choices)} choices; expected 1."
                )
            return message_response_text(completion.choices[0].message)
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


def message_response_text(message: Any) -> str:
    """Read content blocks or reasoning text instead of silently saving blanks."""
    def field(key: str) -> Any:
        if isinstance(message, dict):
            return message.get(key)
        value = getattr(message, key, None)
        extra = getattr(message, "model_extra", None)
        return value if value is not None else (extra or {}).get(key)

    def text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            blocks = [block.get("text") if isinstance(block, dict)
                      else getattr(block, "text", None) for block in value]
            return "\n".join(block for block in blocks if isinstance(block, str))
        return ""

    for key in ("content", "reasoning_content", "reasoning", "analysis"):
        value = text(field(key))
        if value.strip():
            return value
    raise RuntimeError("vLLM returned no response text; refusing to record an empty answer.")


def regrade_result(result: dict[str, Any]) -> int:
    choices = result.get("choices")
    response = result.get("response")
    if not isinstance(choices, list) or not isinstance(response, str):
        raise ValueError("result is missing choices or response")
    result.update(grade_response(response, result))
    return 1


def score_row(
    row: dict[str, Any], response: str, prompt: str
) -> dict[str, Any]:
    return {
        **row,
        "prompt_sha256": stable_digest(prompt),
        "response": response,
        **grade_response(response, row),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_problems": 0,
            "num_correct": 0,
            "num_parsed": 0,
            "accuracy": None,
            "parse_rate": None,
            "parsed_accuracy": None,
            "parse_methods": {},
        }
    num_correct = sum(bool(row.get("correct")) for row in rows)
    num_parsed = sum(row.get("parsed_answer") is not None for row in rows)
    parse_methods: dict[str, int] = defaultdict(int)
    for row in rows:
        parse_methods[str(row.get("parse_method", "unknown"))] += 1
    return {
        "num_problems": len(rows),
        "num_correct": num_correct,
        "num_parsed": num_parsed,
        "num_unparsed": len(rows) - num_parsed,
        "accuracy": num_correct / len(rows),
        "parse_rate": num_parsed / len(rows),
        "parsed_accuracy": num_correct / num_parsed if num_parsed else None,
        "parse_methods": dict(sorted(parse_methods.items())),
    }


def grouped_metrics(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown"))].append(row)
    return {key: aggregate(groups[key]) for key in sorted(groups)}


def refresh_metrics(payload: dict[str, Any], spec: BenchmarkSpec) -> None:
    results = payload["benchmarks"][spec.key]["results"]
    metrics = aggregate(results)
    if spec.key == "mmlu":
        by_subject = grouped_metrics(results, "subject")
        by_category = grouped_metrics(results, "category")
        if by_subject:
            metrics["macro_subject_accuracy"] = sum(
                subject_metrics["accuracy"] for subject_metrics in by_subject.values()
            ) / len(by_subject)
        subject_categories = {
            str(row.get("subject", "unknown")): str(row.get("category", "other"))
            for row in results
        }
        for category, category_metrics in by_category.items():
            subject_scores = [
                subject_metrics["accuracy"]
                for subject, subject_metrics in by_subject.items()
                if subject_categories.get(subject) == category
            ]
            category_metrics["macro_subject_accuracy"] = (
                sum(subject_scores) / len(subject_scores) if subject_scores else None
            )
        stem_metrics = aggregate(
            [row for row in results if row.get("category") == "stem"]
        )
        stem_subject_scores = [
            subject_metrics["accuracy"]
            for subject, subject_metrics in by_subject.items()
            if subject_categories.get(subject) == "stem"
        ]
        stem_metrics["macro_subject_accuracy"] = (
            sum(stem_subject_scores) / len(stem_subject_scores)
            if stem_subject_scores
            else None
        )
        metrics["by_subject"] = by_subject
        metrics["by_category"] = by_category
        metrics["mmlu_stem"] = stem_metrics
    elif spec.key == "gpqa":
        metrics["by_domain"] = grouped_metrics(results, "domain")
    payload["benchmarks"][spec.key]["metrics"] = metrics
    payload["overall"] = {
        key: value
        for key, value in metrics.items()
        if key not in {"by_subject", "by_category", "by_domain", "mmlu_stem"}
    }


def write_output(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def checkpoint_path(output: Path) -> Path:
    resolved = resolve_path(output)
    return resolved.with_suffix(resolved.suffix + ".partial.jsonl")


def run_fingerprint(args: argparse.Namespace, spec: BenchmarkSpec) -> str:
    identity = {
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
        },
        "benchmark": spec.key,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": spec.system_prompt,
    }
    serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return stable_digest(serialized)


def append_checkpoint(
    output: Path, fingerprint: str, result: dict[str, Any]
) -> None:
    path = checkpoint_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"run_fingerprint": fingerprint, "result": result}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def merge_checkpoint(
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    payload: dict[str, Any],
) -> int:
    path = checkpoint_path(args.output)
    if not args.resume:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return 0
    if not path.exists():
        return 0

    expected_fingerprint = payload["run_fingerprint"]
    results = payload["benchmarks"][spec.key]["results"]
    by_id = {row["problem_id"]: row for row in results}
    merged = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"Could not read checkpoint {path}: {exc}") from exc
    last_nonempty_line = max(
        (index for index, line in enumerate(lines, start=1) if line.strip()),
        default=0,
    )
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_number == last_nonempty_line:
                print(
                    f"Warning: ignoring a truncated final checkpoint line in "
                    f"{path}; its request will be regenerated.",
                    flush=True,
                )
                break
            raise SystemExit(
                f"Invalid JSON in checkpoint {path} line {line_number}: {exc}"
            ) from exc
        if record.get("run_fingerprint") != expected_fingerprint:
            raise SystemExit(
                f"Checkpoint {path} line {line_number} belongs to a different "
                "model, dataset, prompt, or generation configuration. Remove it "
                "or run without --resume."
            )
        result = record.get("result")
        if not isinstance(result, dict) or not result.get("problem_id"):
            raise SystemExit(
                f"Checkpoint {path} line {line_number} has no valid result."
            )
        # Always apply the current parser to checkpointed generations. This also
        # covers checkpoints written by an older parser before the full JSON was
        # refreshed.
        try:
            regrade_result(result)
        except ValueError as exc:
            raise SystemExit(
                f"Checkpoint {path} line {line_number} is invalid: {exc}."
            ) from exc
        problem_id = result["problem_id"]
        if problem_id not in by_id:
            results.append(result)
            by_id[problem_id] = result
            merged += 1
    results.sort(key=lambda item: item["order"])
    if merged:
        refresh_metrics(payload, spec)
        write_output(args.output, payload)
    return merged


def new_payload(args: argparse.Namespace, spec: BenchmarkSpec) -> dict[str, Any]:
    return {
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "split": args.split,
        "parser_version": PARSER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompting": {
            "strategy": "zero_shot_solve",
        },
        "run_fingerprint": run_fingerprint(args, spec),
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
        },
        "benchmarks": {
            spec.key: {
                "display_name": spec.display_name,
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "results": [],
                "metrics": {},
            }
        },
        "overall": {},
    }


def regrade_payload(payload: dict[str, Any], spec: BenchmarkSpec) -> int:
    results = payload.get("benchmarks", {}).get(spec.key, {}).get("results", [])
    count = 0
    for row in results:
        try:
            count += regrade_result(row)
        except ValueError as exc:
            raise SystemExit(
                f"Cannot regrade resume output: {exc}."
            ) from exc
    refresh_metrics(payload, spec)
    return count


def load_or_create_payload(
    args: argparse.Namespace, spec: BenchmarkSpec
) -> dict[str, Any]:
    output = resolve_path(args.output)
    if not args.resume or not output.exists():
        return new_payload(args, spec)
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not resume from {output}: {exc}") from exc

    expected = new_payload(args, spec)
    for field in (
        "model",
        "base_url",
        "split",
        "generation",
        "run_fingerprint",
    ):
        if payload.get(field) != expected[field]:
            raise SystemExit(
                f"Resume output has {field}={payload.get(field)!r}, "
                f"but this run uses {expected[field]!r}."
            )
    actual_benchmark = payload.get("benchmarks", {}).get(spec.key, {})
    expected_benchmark = expected["benchmarks"][spec.key]
    for field in ("dataset", "dataset_config"):
        if actual_benchmark.get(field) != expected_benchmark[field]:
            raise SystemExit(
                f"Resume output uses {field}={actual_benchmark.get(field)!r}, "
                f"but this run uses {expected_benchmark[field]!r}."
            )
    if payload.get("parser_version") != PARSER_VERSION:
        count = regrade_payload(payload, spec)
        payload["parser_version"] = PARSER_VERSION
        write_output(args.output, payload)
        print(f"Regraded {count} cached responses with parser v{PARSER_VERSION}.")
    return payload


def validate_unique_ids(rows: list[dict[str, Any]], name: str) -> None:
    ids = [row["problem_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{name} contains duplicate normalized problem IDs.")


def print_metrics(spec: BenchmarkSpec, metrics: dict[str, Any]) -> None:
    if not metrics.get("num_problems"):
        return
    print(
        f"{spec.key}: problems={metrics['num_problems']}, "
        f"correct={metrics['num_correct']}, accuracy={metrics['accuracy']:.4f}, "
        f"parse_rate={metrics['parse_rate']:.4f}"
    )
    stem = metrics.get("mmlu_stem")
    if stem and stem.get("num_problems"):
        print(
            f"mmlu_stem: problems={stem['num_problems']}, "
            f"accuracy={stem['accuracy']:.4f}, parse_rate={stem['parse_rate']:.4f}"
        )


def run(spec: BenchmarkSpec) -> None:
    args = parse_args(spec)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")

    datasets = require_package("datasets")
    rows = load_benchmark(datasets, spec, args)
    validate_unique_ids(rows, f"{spec.display_name} evaluation split")

    client = build_client(args)
    check_server(client, args.model)
    payload = load_or_create_payload(args, spec)
    merged = merge_checkpoint(args, spec, payload)
    if not args.resume:
        # A non-resume run is an explicit fresh run. Replace a stale full output
        # before issuing requests so a later resume cannot mix old generations
        # with a new checkpoint if this process is interrupted early.
        refresh_metrics(payload, spec)
        write_output(args.output, payload)
    if merged:
        print(f"Recovered {merged} completed rows from the JSONL checkpoint.")
    completed_ids = {
        row["problem_id"]
        for row in payload["benchmarks"][spec.key]["results"]
    }
    requested_ids = {row["problem_id"] for row in rows}
    unexpected_ids = completed_ids - requested_ids
    if unexpected_ids:
        raise SystemExit(
            f"Resume output contains {len(unexpected_ids)} rows outside the current "
            "evaluation selection. Use the same or a larger --limit, or choose a "
            "new --output path."
        )
    pending = [row for row in rows if row["problem_id"] not in completed_ids]
    completed_now = len(rows) - len(pending)
    existing_results = payload["benchmarks"][spec.key]["results"]
    running_correct = sum(bool(row.get("correct")) for row in existing_results)
    running_parsed = sum(
        row.get("parsed_answer") is not None for row in existing_results
    )

    jobs: list[tuple[dict[str, Any], str]] = []
    for row in pending:
        prompt = build_prompt(row)
        jobs.append((row, prompt))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                request_completion,
                client,
                row,
                prompt,
                spec.system_prompt,
                args,
            ): (row, prompt)
            for row, prompt in jobs
        }
        try:
            for future in as_completed(futures):
                row, prompt = futures[future]
                response = future.result()
                result = score_row(row, response, prompt)
                append_checkpoint(
                    args.output,
                    payload["run_fingerprint"],
                    result,
                )
                results = payload["benchmarks"][spec.key]["results"]
                results.append(result)
                completed_now += 1
                running_correct += int(bool(result["correct"]))
                running_parsed += int(result["parsed_answer"] is not None)
                if completed_now % args.save_every == 0:
                    results.sort(key=lambda item: item["order"])
                    refresh_metrics(payload, spec)
                    write_output(args.output, payload)
                print(
                    f"[{completed_now}/{len(rows)}] {row['problem_id']}: "
                    f"parsed={result['parsed_answer']!r}, "
                    f"correct={result['correct']} | "
                    f"accuracy={running_correct / completed_now:.4f}, "
                    f"parse_rate={running_parsed / completed_now:.4f}",
                    flush=True,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    payload["benchmarks"][spec.key]["results"].sort(key=lambda item: item["order"])
    refresh_metrics(payload, spec)
    write_output(args.output, payload)
    print_metrics(spec, payload["benchmarks"][spec.key]["metrics"])
    print(f"Saved detailed results to {resolve_path(args.output)}")
