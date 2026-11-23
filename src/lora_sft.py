from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from .lora_dpo import (
        build_lora_config,
        filter_supported_kwargs,
        get_local_rank,
        load_config,
        require_package,
        resolve_path,
        resolve_resume_checkpoint,
        torch_dtype_from_string,
        validate_model_name_or_path,
    )
except ImportError:
    from lora_dpo import (  # type: ignore[no-redef]
        build_lora_config,
        filter_supported_kwargs,
        get_local_rank,
        load_config,
        require_package,
        resolve_path,
        resolve_resume_checkpoint,
        torch_dtype_from_string,
        validate_model_name_or_path,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs/lora_sft_config.json"
IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter on conversational SFT data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"SFT config JSON path (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument("--data-path", type=Path, help="Override SFT data JSON path.")
    parser.add_argument("--model", help="Override base model path/name.")
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument("--max-samples", type=int, help="Use at most this many rows.")
    parser.add_argument("--max-length", type=int, help="Override tokenized sequence length.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="latest",
        help=(
            "Resume from a checkpoint path. Without a value, use the latest "
            "checkpoint under training.output_dir."
        ),
    )
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank passed by a distributed launcher such as torchrun.",
    )
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    if args.data_path:
        config["data_path"] = str(args.data_path)
    if args.model:
        config["model_name_or_path"] = args.model
    if args.output_dir:
        training = dict(config.get("training", {}))
        training["output_dir"] = str(args.output_dir)
        config["training"] = training
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples
    if args.max_length is not None:
        config["max_length"] = args.max_length
    if args.seed is not None:
        config["seed"] = args.seed
        training = dict(config.get("training", {}))
        training["seed"] = args.seed
        config["training"] = training
    return config


def load_sft_rows(
    path: Path, max_samples: int | None, seed: int
) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read SFT data {path}: {exc}") from exc
    if not isinstance(data, list):
        raise SystemExit(f"SFT data {path} must contain a JSON list.")

    rows: list[dict[str, Any]] = []
    skipped = 0
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise SystemExit(f"{path}[{index}] must be a JSON object.")
        messages = item.get("messages")
        if not valid_messages(messages):
            skipped += 1
            continue
        rows.append(
            {
                "problem_id": str(item.get("problem_id") or index),
                "messages": messages,
            }
        )

    if max_samples is not None:
        if max_samples <= 0:
            raise SystemExit("--max-samples must be positive.")
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_samples]
    if not rows:
        raise SystemExit(f"No valid SFT rows found in {path}.")
    print(f"Loaded {len(rows)} valid SFT rows from {path}; skipped {skipped}.")
    return rows


def valid_messages(value: Any) -> bool:
    if not isinstance(value, list) or len(value) < 2:
        return False
    for message in value:
        if not isinstance(message, dict):
            return False
        if message.get("role") not in {"system", "user", "assistant"}:
            return False
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            return False
    return value[-1].get("role") == "assistant"


def tokenize_row(
    row: dict[str, Any], tokenizer: Any, max_length: int
) -> tuple[dict[str, list[int]] | None, bool]:
    messages = row["messages"]
    prompt_messages = messages[:-1]
    try:
        prompt_ids = chat_template_input_ids(
            tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        full_ids = chat_template_input_ids(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
    except (AttributeError, ValueError, TypeError) as exc:
        raise SystemExit(
            "The tokenizer must provide a working chat_template for messages data: "
            f"{exc}"
        ) from exc

    if full_ids[: len(prompt_ids)] != prompt_ids:
        # Some reasoning-model templates inject ``<think>`` for generation but
        # strip the reasoning section from completed assistant messages.  In
        # that case rendering the complete conversation both changes the
        # prefix and silently drops the SFT target.  Build the sequence from
        # the generation prefix and the original assistant content instead.
        full_ids = prompt_ids + assistant_completion_input_ids(
            tokenizer,
            prompt_messages,
            messages[-1]["content"],
        )

    truncated = len(full_ids) > max_length
    input_ids = full_ids[:max_length]
    prompt_length = len(prompt_ids)
    if prompt_length >= len(input_ids):
        return None, truncated
    labels = [IGNORE_INDEX] * prompt_length + input_ids[prompt_length:]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }, truncated


def assistant_completion_input_ids(
    tokenizer: Any,
    prompt_messages: list[dict[str, Any]],
    assistant_content: str,
) -> list[int]:
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    completion = remove_duplicate_think_prefix(str(prompt_text), assistant_content)
    completion_ids = chat_template_input_ids(
        tokenizer(completion, add_special_tokens=False)
    )

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int) and (
        not completion_ids or completion_ids[-1] != eos_token_id
    ):
        completion_ids.append(eos_token_id)
    return completion_ids


def remove_duplicate_think_prefix(prompt: str, completion: str) -> str:
    if not prompt.rstrip().endswith("<think>"):
        return completion

    stripped = completion.lstrip()
    if not stripped.startswith("<think>"):
        return completion
    stripped = stripped[len("<think>") :]
    if stripped.startswith("\r\n"):
        stripped = stripped[2:]
    elif stripped.startswith("\n"):
        stripped = stripped[1:]
    return stripped


def chat_template_input_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("chat template result does not contain input_ids")
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if (
        isinstance(encoded, (list, tuple))
        and len(encoded) == 1
        and isinstance(encoded[0], (list, tuple))
    ):
        encoded = encoded[0]
    if not isinstance(encoded, (list, tuple)) or not all(
        isinstance(token_id, int) for token_id in encoded
    ):
        raise ValueError(
            f"unexpected chat template token type: {type(encoded).__name__}"
        )
    return list(encoded)


def tokenize_rows(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int
) -> list[dict[str, list[int]]]:
    tokenized: list[dict[str, list[int]]] = []
    truncated = 0
    skipped = 0
    for row in rows:
        item, was_truncated = tokenize_row(row, tokenizer, max_length)
        truncated += int(was_truncated)
        if item is None:
            skipped += 1
            continue
        tokenized.append(item)
    if not tokenized:
        raise SystemExit("No trainable rows remain after tokenization.")
    print(
        f"Tokenized {len(tokenized)} rows; truncated {truncated}; "
        f"skipped {skipped} rows whose prompt filled max_length."
    )
    return tokenized


def make_datasets(
    rows: list[dict[str, list[int]]], config: dict[str, Any]
) -> tuple[Any, Any | None]:
    datasets = require_package("datasets")
    dataset = datasets.Dataset.from_list(rows)
    eval_ratio = float(config.get("eval_ratio", 0.0))
    if eval_ratio <= 0:
        return dataset, None
    if not 0 < eval_ratio < 1:
        raise SystemExit("eval_ratio must be between 0 and 1.")
    split = dataset.train_test_split(
        test_size=eval_ratio, seed=int(config.get("seed", 42))
    )
    return split["train"], split["test"]


def build_training_args(
    config: dict[str, Any], TrainingArguments: Any, has_eval_dataset: bool
) -> Any:
    training = dict(config.get("training", {}))
    if "output_dir" not in training:
        raise SystemExit("Config field training.output_dir is required.")
    training["output_dir"] = str(resolve_path(training["output_dir"]))
    training.pop("deepspeed", None)
    if not has_eval_dataset:
        training["eval_strategy"] = "no"
        training["evaluation_strategy"] = "no"
    training = filter_supported_kwargs(TrainingArguments.__init__, training)
    return TrainingArguments(**training)


def build_sft_model_kwargs(config: dict[str, Any], torch: Any) -> dict[str, Any]:
    model_kwargs = dict(config.get("model_kwargs", {}))
    if "torch_dtype" in model_kwargs:
        model_kwargs["torch_dtype"] = torch_dtype_from_string(
            torch, model_kwargs["torch_dtype"]
        )
    return model_kwargs


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    data_path_value = config.get("data_path")
    model_name = config.get("model_name_or_path")
    if not data_path_value:
        raise SystemExit("Config field data_path is required.")
    if not model_name:
        raise SystemExit("Config field model_name_or_path is required.")
    model_name = validate_model_name_or_path(str(model_name))
    max_length = int(config.get("max_length", 32768))
    if max_length <= 0:
        raise SystemExit("Config field max_length must be positive.")
    print(f"Using model: {model_name}")

    seed = int(config.get("seed", 42))
    rows = load_sft_rows(
        resolve_path(data_path_value), config.get("max_samples"), seed
    )

    torch = require_package("torch")
    transformers = require_package("transformers")
    peft = require_package("peft")

    local_rank = get_local_rank(args)
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = str(config.get("padding_side", "right"))

    tokenized_rows = tokenize_rows(rows, tokenizer, max_length)
    train_dataset, eval_dataset = make_datasets(tokenized_rows, config)

    model_kwargs = build_sft_model_kwargs(config, torch)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        **model_kwargs,
    )
    gradient_checkpointing = bool(config.get("gradient_checkpointing", True))
    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model = peft.get_peft_model(
        model, build_lora_config(config, peft.LoraConfig)
    )
    model.print_trainable_parameters()

    training_args = build_training_args(
        config, transformers.TrainingArguments, eval_dataset is not None
    )
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
    trainer_parameters = filter_supported_kwargs(
        transformers.Trainer.__init__,
        {"processing_class": tokenizer, "tokenizer": tokenizer},
    )
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = transformers.Trainer(**trainer_kwargs)
    resume_checkpoint = resolve_resume_checkpoint(
        args.resume_from_checkpoint, training_args.output_dir
    )
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Saved LoRA SFT adapter to {training_args.output_dir}")


if __name__ == "__main__":
    main()
