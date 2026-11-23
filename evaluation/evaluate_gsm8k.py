from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, DecimalException, InvalidOperation, localcontext
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache/huggingface"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation/results/gsm8k.json"
DEFAULT_DATASET = "openai/gsm8k"
DEFAULT_DATASET_CONFIG = "main"
MAX_GRADING_RESPONSE_CHARS = 262_144
SYSTEM_PROMPT = (
    "You are an expert mathematical reasoning assistant. Solve the problem "
    "carefully and show your reasoning. End your response with the final "
    "numeric answer in the form \\boxed{ANSWER}."
)
NUMBER_TOKEN = (
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)
NUMBER_PATTERN = re.compile(rf"(?<![\w.]){NUMBER_TOKEN}(?!\w|\.\d)")
FINAL_ANSWER_MARKER = re.compile(
    r"(?:final(?:\s+(?:answer|result))?\b|answer\s*(?:is|:|=)|therefore|thus|"
    r"最终答案\s*(?:是|为|:|：)?|答案\s*(?:是|为|:|：))",
    re.IGNORECASE,
)
GOLD_ANSWER_PATTERN = re.compile(r"####\s*([^\r\n]+)")
LATEX_FRACTION_PATTERN = re.compile(
    rf"\\(?:dfrac|tfrac|frac)\s*\{{\s*({NUMBER_TOKEN})\s*\}}"
    rf"\s*\{{\s*({NUMBER_TOKEN})\s*\}}"
)
PLAIN_FRACTION_PATTERN = re.compile(
    rf"(?<![\w.])({NUMBER_TOKEN})\s*/\s*({NUMBER_TOKEN})(?![\w.])"
)
LATEX_SCIENTIFIC_PATTERN = re.compile(
    rf"(?<![\w.])({NUMBER_TOKEN})\s*(?:\\times|\\cdot|×|·)\s*10"
    rf"\s*\^\s*(?:\{{\s*([-+]?\d+)\s*\}}|([-+]?\d+))(?![\w.])"
)
SIMPLE_EXPRESSION_PATTERN = re.compile(
    rf"(?<![\w.]){NUMBER_TOKEN}"
    rf"(?:\s*(?:\*\*|[+\-*/^×·]|\\times|\\cdot)\s*{NUMBER_TOKEN})+"
    rf"(?![\w.])"
)
UNICODE_MINUS_TRANSLATION = str.maketrans({"−": "-", "–": "-", "—": "-"})


@dataclass(frozen=True)
class MathVerifyBackend:
    parse: Any
    verify: Any
    prediction_config: tuple[Any, ...]
    gold_config: tuple[Any, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a model served by a vLLM OpenAI-compatible server on "
            "GSM8K with one generated solution per problem."
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
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key configured on the vLLM server (default: EMPTY).",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--hf-endpoint",
        help="Alternative Hugging Face endpoint, e.g. https://hf-mirror.com.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Timeout in seconds for each vLLM request.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum attempts for each vLLM request.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Initial retry delay in seconds; doubles after each failure.",
    )
    parser.add_argument(
        "--strict-boxed",
        action="store_true",
        help="Only accept answers written inside \\boxed{...}.",
    )
    parser.add_argument(
        "--grader",
        choices=("auto", "math_verify", "numeric"),
        default="auto",
        help=(
            "Answer grader. 'auto' uses math-verify when installed and otherwise "
            "uses the built-in numeric grader (default: auto)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from an existing output and skip completed problems.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate at most this many problems.",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
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


def require_package(package: str, install_name: str | None = None) -> Any:
    try:
        return __import__(package, fromlist=["*"])
    except ImportError as exc:
        name = install_name or package
        raise SystemExit(
            f"Missing dependency '{name}'. Install it with: pip install {name}"
        ) from exc


def resolve_math_verify_backend(
    requested: str,
) -> tuple[MathVerifyBackend | None, str]:
    if requested == "numeric":
        return None, "numeric"
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
            "math-verify is not installed; using the built-in numeric grader. "
            "Install it with: pip install math-verify",
            flush=True,
        )
        return None, "numeric"

    backend = MathVerifyBackend(
        parse=parse,
        verify=verify,
        prediction_config=(LatexExtractionConfig(), ExprExtractionConfig()),
        gold_config=(
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
        ),
    )
    return backend, "math_verify"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def canonical_decimal(number: Decimal) -> str | None:
    if not number.is_finite():
        return None
    if number.is_zero():
        return "0"
    # Scientific notation can describe an enormous expanded decimal with only
    # a few input characters. Bound formatting work for malformed output.
    if abs(number.adjusted()) > 100_000:
        return None
    if number == number.to_integral_value():
        # Avoid converting through ``int``: Python 3.11 limits decimal string
        # conversions to 4,300 digits, while malformed model output can easily
        # contain a much longer run of digits. Decimal can format it directly.
        return format(number.to_integral_value(), "f")
    return format(number.normalize(), "f")


def decimal_from_token(token: str) -> Decimal | None:
    token = token.translate(UNICODE_MINUS_TRANSLATION).replace(",", "")
    try:
        number = Decimal(token)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def normalize_number(value: Any) -> str | None:
    text = str(value).replace("$", "").translate(UNICODE_MINUS_TRANSLATION)
    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    number = decimal_from_token(matches[-1])
    return canonical_decimal(number) if number is not None else None


def evaluate_decimal_expression(expression: str) -> Decimal | None:
    expression = (
        expression.translate(UNICODE_MINUS_TRANSLATION)
        .replace("\\times", "*")
        .replace("\\cdot", "*")
        .replace("×", "*")
        .replace("·", "*")
        .replace("^", "**")
        .replace(",", "")
    )
    if len(expression) > 256 or not re.fullmatch(r"[\d.eE+\-*/()\s]+", expression):
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError):
        return None
    if sum(1 for _ in ast.walk(tree)) > 32:
        return None

    def visit(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                raise ValueError("booleans are not numeric answers")
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                if right != right.to_integral_value() or abs(right) > 10_000:
                    raise ValueError("unsupported exponent")
                return left ** int(right)
        raise ValueError("unsupported numeric expression")

    try:
        with localcontext() as context:
            context.prec = 50
            number = visit(tree)
    except (
        DecimalException,
        InvalidOperation,
        OverflowError,
        ValueError,
        ZeroDivisionError,
    ):
        return None
    return number if number.is_finite() else None


def parse_numeric_value(value: Any) -> str | None:
    text = str(value).strip().translate(UNICODE_MINUS_TRANSLATION)
    text = text.replace("\\$", "").replace("$", "").strip()

    fraction = LATEX_FRACTION_PATTERN.fullmatch(text)
    if fraction is None:
        fraction = PLAIN_FRACTION_PATTERN.fullmatch(text)
    if fraction is not None:
        numerator = decimal_from_token(fraction.group(1))
        denominator = decimal_from_token(fraction.group(2))
        if numerator is None or denominator is None or denominator.is_zero():
            return None
        try:
            with localcontext() as context:
                context.prec = 50
                return canonical_decimal(numerator / denominator)
        except DecimalException:
            return None

    scientific = LATEX_SCIENTIFIC_PATTERN.fullmatch(text)
    if scientific is not None:
        coefficient = decimal_from_token(scientific.group(1))
        exponent_text = scientific.group(2) or scientific.group(3)
        if coefficient is None:
            return None
        exponent = int(exponent_text)
        if abs(exponent) > 10_000:
            return None
        try:
            return canonical_decimal(coefficient * (Decimal(10) ** exponent))
        except DecimalException:
            return None

    direct = re.fullmatch(NUMBER_TOKEN, text)
    if direct is not None:
        number = decimal_from_token(direct.group(0))
        return canonical_decimal(number) if number is not None else None

    number = evaluate_decimal_expression(text)
    return canonical_decimal(number) if number is not None else None


def numeric_candidates(text: str) -> list[tuple[int, int, int, str]]:
    text = text.translate(UNICODE_MINUS_TRANSLATION)
    patterns = (
        (LATEX_FRACTION_PATTERN, 0),
        (LATEX_SCIENTIFIC_PATTERN, 0),
        (SIMPLE_EXPRESSION_PATTERN, 1),
        (PLAIN_FRACTION_PATTERN, 1),
        (NUMBER_PATTERN, 2),
    )
    candidates: list[tuple[int, int, int, str]] = []
    for pattern, priority in patterns:
        for match in pattern.finditer(text):
            answer = parse_numeric_value(match.group(0))
            if answer is not None:
                candidates.append((match.start(), match.end(), priority, answer))
    return candidates


def first_numeric_answer(text: str) -> str | None:
    candidates = numeric_candidates(text)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (item[0], item[2], -(item[1] - item[0])),
    )[3]


def last_numeric_answer(text: str) -> str | None:
    candidates = numeric_candidates(text)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (item[1], -item[2], item[1] - item[0]),
    )[3]


def extract_gold_answer(answer: Any) -> str | None:
    text = str(answer)
    matches = GOLD_ANSWER_PATTERN.findall(text)
    answer_text = matches[-1] if matches else text
    return parse_numeric_value(answer_text) or last_numeric_answer(answer_text)


def boxed_contents(text: str) -> list[str]:
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
                    contents.append(text[start:index])
                    break
    return contents


def visible_answer_text(text: str) -> str:
    # Ignore hidden reasoning, where a reasoning model may mention tentative
    # answers, and retain the end of pathological oversized responses because
    # that is where the requested final answer should appear.
    visible = str(text).replace("\x00", "").rsplit("</think>", 1)[-1]
    if len(visible) > MAX_GRADING_RESPONSE_CHARS:
        visible = visible[-MAX_GRADING_RESPONSE_CHARS:]
    return visible


def extract_gsm8k_answer(text: str, strict_boxed: bool) -> tuple[str | None, str]:
    visible_text = visible_answer_text(text)

    structured_box_failed = False
    for content in reversed(boxed_contents(visible_text)):
        answer = parse_numeric_value(content)
        if answer is not None:
            return answer, "boxed"
        if any(
            pattern.search(content)
            for pattern in (
                LATEX_FRACTION_PATTERN,
                LATEX_SCIENTIFIC_PATTERN,
                SIMPLE_EXPRESSION_PATTERN,
                PLAIN_FRACTION_PATTERN,
            )
        ):
            structured_box_failed = True
            continue
        answer = normalize_number(content)
        if answer is not None:
            return answer, "boxed"

    bare_boxed_markers = list(
        re.finditer(r"\\boxed\b(?!\s*\{)", visible_text, re.IGNORECASE)
    )
    for marker in reversed(bare_boxed_markers):
        answer = first_numeric_answer(visible_text[marker.end() : marker.end() + 160])
        if answer is not None:
            return answer, "boxed_without_braces"
    if strict_boxed or structured_box_failed:
        return None, "not_found"

    final_markers = list(FINAL_ANSWER_MARKER.finditer(visible_text))
    for marker in reversed(final_markers):
        answer = first_numeric_answer(visible_text[marker.end() : marker.end() + 160])
        if answer is not None:
            return answer, "final_answer_phrase"

    answer = last_numeric_answer(visible_text)
    if answer is not None:
        return answer, "last_number"
    return None, "not_found"


def parsed_representation(parsed: Any) -> str | None:
    if not parsed:
        return None
    try:
        candidates = reversed(parsed)
    except TypeError:
        candidates = (parsed,)
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            text = str(candidate).strip()
        except Exception:
            continue
        if text:
            return text
    return None


def math_verify_grade(
    prediction_text: str,
    gold_text: str,
    backend: MathVerifyBackend,
) -> tuple[str | None, bool | None, str]:
    try:
        gold_parsed = backend.parse(
            str(gold_text), extraction_config=backend.gold_config
        )
    except Exception as exc:
        return None, None, f"gold_parse_error:{type(exc).__name__}"
    try:
        prediction_parsed = backend.parse(
            prediction_text,
            extraction_config=backend.prediction_config,
        )
    except Exception as exc:
        return None, None, f"prediction_parse_error:{type(exc).__name__}"

    representation = parsed_representation(prediction_parsed)
    if not gold_parsed:
        return representation, None, "gold_not_found"
    if not prediction_parsed:
        return representation, None, "prediction_not_found"
    try:
        correct = bool(backend.verify(gold_parsed, prediction_parsed))
    except Exception as exc:
        return representation, None, f"verify_error:{type(exc).__name__}"
    return representation, correct, "verified"


def grade_completion(
    response: str,
    gold_answer: str,
    strict_boxed: bool,
    math_verify_backend: MathVerifyBackend | None,
) -> dict[str, Any]:
    parsed_answer, parse_method = extract_gsm8k_answer(response, strict_boxed)
    numeric_correct = parsed_answer == gold_answer
    result: dict[str, Any] = {
        "parsed_answer": parsed_answer,
        "parse_method": parse_method,
        "math_verify_answer": None,
        "math_verify_status": "disabled",
        "grading_method": "numeric",
        "correct": numeric_correct,
    }
    if math_verify_backend is None:
        return result

    visible_text = visible_answer_text(response)
    if strict_boxed and not (
        boxed_contents(visible_text)
        or re.search(r"\\boxed\b", visible_text, re.IGNORECASE)
    ):
        result["math_verify_status"] = "strict_boxed_missing"
        result["grading_method"] = "strict_boxed"
        result["correct"] = False
        return result
    math_answer, math_correct, status = math_verify_grade(
        visible_text,
        gold_answer,
        math_verify_backend,
    )
    result["math_verify_answer"] = math_answer
    result["math_verify_status"] = status
    if math_correct is None:
        result["grading_method"] = "numeric_fallback"
    else:
        result["grading_method"] = "math_verify"
        result["correct"] = math_correct
    return result


def load_benchmark(
    datasets: Any,
    dataset_name: str,
    dataset_config: str,
    split: str,
    cache_dir: Path,
    limit: int | None,
) -> list[dict[str, str]]:
    try:
        dataset = datasets.load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            cache_dir=str(resolve_path(cache_dir)),
        )
    except Exception as exc:
        raise SystemExit(
            f"Could not load {dataset_name}/{dataset_config} split {split!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    rows: list[dict[str, str]] = []
    for index, raw_item in enumerate(dataset):
        item = dict(raw_item)
        question = item.get("question")
        gold_answer = extract_gold_answer(item.get("answer", ""))
        raw_id = item.get("id", item.get("problem_id", index))
        if not isinstance(question, str) or not question.strip() or gold_answer is None:
            raise SystemExit(
                f"Invalid GSM8K row {index}: expected question and an answer "
                "containing a numeric result."
            )
        rows.append(
            {
                "problem_id": f"gsm8k/{raw_id}",
                "problem": question.strip(),
                "gold_answer": gold_answer,
            }
        )
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(
            f"No rows loaded from {dataset_name}/{dataset_config} split {split!r}."
        )
    print(
        f"Loaded {len(rows)} GSM8K problems from {dataset_name}/{dataset_config} "
        f"split {split!r}."
    )
    return rows


def build_client(args: argparse.Namespace) -> Any:
    openai = require_package("openai")
    return openai.OpenAI(
        api_key=args.api_key,
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout,
        max_retries=0,
    )


def check_server(client: Any, model_name: str) -> None:
    try:
        available = sorted(model.id for model in client.models.list().data)
    except Exception as exc:
        raise SystemExit(
            "Could not connect to the vLLM server. Start it before evaluation and "
            f"check --base-url. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    if model_name not in available:
        raise SystemExit(
            f"Model {model_name!r} is not served by vLLM. Available models: "
            + ", ".join(available)
        )
    print(f"Connected to vLLM at {client.base_url}; using model {model_name}.")


def request_seed(base_seed: int, problem_id: str) -> int:
    digest = hashlib.sha256(problem_id.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)


def request_completion(
    client: Any,
    row: dict[str, str],
    args: argparse.Namespace,
) -> str:
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
            return completion.choices[0].message.content or ""
        except Exception as exc:
            if attempt >= args.retries:
                raise RuntimeError(
                    f"{row['problem_id']} failed after {args.retries} attempts: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            delay = args.retry_delay * (2 ** (attempt - 1))
            print(
                f"{row['problem_id']} attempt {attempt}/{args.retries} failed: "
                f"{type(exc).__name__}: {exc}; retrying in {delay:g}s"
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def score_problem(
    row: dict[str, str],
    response: str,
    strict_boxed: bool,
    math_verify_backend: MathVerifyBackend | None = None,
) -> dict[str, Any]:
    return {
        **row,
        "response": response,
        **grade_completion(
            response,
            row["gold_answer"],
            strict_boxed,
            math_verify_backend,
        ),
    }


def benchmark_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"num_problems": 0, "num_correct": 0, "accuracy": None}
    num_correct = sum(bool(row.get("correct")) for row in results)
    return {
        "num_problems": len(results),
        "num_correct": num_correct,
        "accuracy": num_correct / len(results),
    }


def refresh_metrics(payload: dict[str, Any]) -> None:
    results = payload["benchmarks"]["gsm8k"]["results"]
    metrics = benchmark_metrics(results)
    payload["benchmarks"]["gsm8k"]["metrics"] = metrics
    payload["overall"] = dict(metrics)


def regrade_payload(
    payload: dict[str, Any],
    strict_boxed: bool,
    math_verify_backend: MathVerifyBackend | None,
) -> int:
    results = payload.get("benchmarks", {}).get("gsm8k", {}).get("results", [])
    regraded = 0
    for row in results:
        gold_answer = row.get("gold_answer")
        response = row.get("response")
        if not isinstance(gold_answer, str) or not isinstance(response, str):
            raise SystemExit(
                "Cannot regrade resume output: a GSM8K result is missing "
                "gold_answer or response."
            )
        row.update(
            grade_completion(
                response,
                gold_answer,
                strict_boxed,
                math_verify_backend,
            )
        )
        regraded += 1
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


def new_payload(
    args: argparse.Namespace,
    grading_backend: str,
) -> dict[str, Any]:
    return {
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "strict_boxed": args.strict_boxed,
        "split": args.split,
        "grading": {
            "backend": grading_backend,
            "math_verify_fallback": (
                "numeric" if grading_backend == "math_verify" else None
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
            "gsm8k": {
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
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
    existing_results = (
        payload.get("benchmarks", {}).get("gsm8k", {}).get("results", [])
    )
    if payload.get("num_samples", 1) != 1 or any(
        isinstance(row, dict) and "samples" in row for row in existing_results
    ):
        raise SystemExit(
            "Resume output was created by the old multi-sample GSM8K evaluator. "
            "Use a new --output path (or remove the old output and checkpoint) "
            "for one-completion-per-problem evaluation."
        )
    expected = new_payload(args, grading_backend)
    for field in (
        "model",
        "base_url",
        "strict_boxed",
        "split",
    ):
        if payload.get(field) != expected[field]:
            raise SystemExit(
                f"Resume output has {field}={payload.get(field)!r}, "
                f"but this run uses {expected[field]!r}."
            )
    if payload.get("generation") != expected["generation"]:
        raise SystemExit("Resume output uses different generation settings.")
    actual = payload.get("benchmarks", {}).get("gsm8k", {})
    expected_benchmark = expected["benchmarks"]["gsm8k"]
    for field in ("dataset", "dataset_config"):
        if actual.get(field) != expected_benchmark[field]:
            raise SystemExit(
                f"Resume output uses {field}={actual.get(field)!r}, "
                f"but this run uses {expected_benchmark[field]!r}."
            )
    existing_grading = payload.get("grading")
    if existing_grading != expected["grading"]:
        regraded = regrade_payload(
            payload,
            args.strict_boxed,
            math_verify_backend,
        )
        payload["grading"] = expected["grading"]
        write_output(args.output, payload)
        print(
            f"Regraded {regraded} cached GSM8K responses with "
            f"{grading_backend} grading.",
            flush=True,
        )
    return payload


def print_metrics(name: str, metrics: dict[str, Any]) -> None:
    if not metrics.get("num_problems"):
        return
    print(
        f"{name}: problems={metrics['num_problems']}, "
        f"correct={metrics['num_correct']}, accuracy={metrics['accuracy']:.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")

    math_verify_backend, grading_backend = resolve_math_verify_backend(args.grader)
    print(f"Using GSM8K grading backend: {grading_backend}.")
    datasets = require_package("datasets")
    client = build_client(args)
    check_server(client, args.model)
    rows = load_benchmark(
        datasets,
        args.dataset,
        args.dataset_config,
        args.split,
        args.cache_dir,
        args.limit,
    )
    payload = load_or_create_payload(
        args,
        grading_backend,
        math_verify_backend,
    )

    completed_ids = {
        row["problem_id"]
        for row in payload["benchmarks"]["gsm8k"]["results"]
    }
    pending = [row for row in rows if row["problem_id"] not in completed_ids]
    completed_now = len(rows) - len(pending)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(request_completion, client, row, args): row
            for row in pending
        }
        try:
            for future in as_completed(futures):
                row = futures[future]
                response = future.result()
                result = score_problem(
                    row,
                    response,
                    args.strict_boxed,
                    math_verify_backend,
                )
                payload["benchmarks"]["gsm8k"]["results"].append(result)
                completed_now += 1
                refresh_metrics(payload)
                write_output(args.output, payload)
                metrics = payload["benchmarks"]["gsm8k"]["metrics"]
                print(
                    f"[{completed_now}/{len(rows)}] {row['problem_id']}: "
                    f"parsed={result['parsed_answer']!r}, "
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
    print_metrics("gsm8k", payload["benchmarks"]["gsm8k"]["metrics"])
    print(f"Saved detailed results to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
