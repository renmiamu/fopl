from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

try:
    from .lora_dpo import (
        filter_supported_kwargs,
        get_local_rank,
        load_config,
        require_package,
        resolve_path,
        resolve_resume_checkpoint,
        validate_model_name_or_path,
    )
    from .lora_sft import (
        IGNORE_INDEX,
        build_sft_model_kwargs,
        make_datasets,
        tokenize_rows,
        valid_messages,
    )
except ImportError:
    from lora_dpo import (  # type: ignore[no-redef]
        filter_supported_kwargs,
        get_local_rank,
        load_config,
        require_package,
        resolve_path,
        resolve_resume_checkpoint,
        validate_model_name_or_path,
    )
    from lora_sft import (  # type: ignore[no-redef]
        IGNORE_INDEX,
        build_sft_model_kwargs,
        make_datasets,
        tokenize_rows,
        valid_messages,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs/full_sft_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-parameter fine-tuning on conversational SFT data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Full SFT config JSON path (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Override the conversational SFT JSON/JSONL path.",
    )
    parser.add_argument("--model", help="Override the base model path/name.")
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument("--max-samples", type=int, help="Use at most this many rows.")
    parser.add_argument("--max-length", type=int, help="Override sequence length.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument(
        "--deepspeed",
        help="Override DeepSpeed config path; use none/off/false/no to disable it.",
    )
    parser.add_argument(
        "--zero-stage",
        type=int,
        choices=(2, 3),
        help="Use the bundled DeepSpeed ZeRO stage 2 or 3 config.",
    )
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
        help="Local rank passed by DeepSpeed or torchrun.",
    )
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.max_length is not None and args.max_length <= 0:
        parser.error("--max-length must be positive")
    return args


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    if args.data_path:
        config["data_path"] = str(args.data_path)
    if args.model:
        config["model_name_or_path"] = args.model
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples
    if args.max_length is not None:
        config["max_length"] = args.max_length
    if args.seed is not None:
        config["seed"] = args.seed

    training = dict(config.get("training", {}))
    if args.output_dir:
        training["output_dir"] = str(args.output_dir)
    if args.seed is not None:
        training["seed"] = args.seed
    if args.zero_stage is not None:
        training["deepspeed"] = f"configs/ds_zero{args.zero_stage}.json"
    if args.deepspeed is not None:
        deepspeed = args.deepspeed.strip()
        if deepspeed.lower() in {"", "none", "off", "false", "no"}:
            training.pop("deepspeed", None)
        else:
            training["deepspeed"] = deepspeed
    config["training"] = training
    return config


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    line_number = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("row is not a JSON object")
                records.append(item)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(
            f"Could not read JSONL {path} at line {line_number}: {exc}"
        ) from exc
    return records


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"SFT data file does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise SystemExit(f"Could not read SFT data {path}: {exc}") from exc
    except json.JSONDecodeError:
        return load_jsonl(path)
    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise SystemExit(
            f"SFT data {path} must contain a JSON array, a JSON object, or JSONL."
        )
    return data


def load_sft_rows(
    path: Path, max_samples: int | None, seed: int
) -> list[dict[str, Any]]:
    data = load_json_or_jsonl(path)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for index, item in enumerate(data):
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
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_samples]
    if not rows:
        raise SystemExit(f"No valid SFT rows found in {path}.")
    print(f"Loaded {len(rows)} valid SFT rows from {path}; skipped {skipped}.")
    return rows


def build_training_args(
    config: dict[str, Any], TrainingArguments: Any, has_eval_dataset: bool
) -> Any:
    training = dict(config.get("training", {}))
    if "output_dir" not in training:
        raise SystemExit("Config field training.output_dir is required.")
    training["output_dir"] = str(resolve_path(training["output_dir"]))
    deepspeed = training.get("deepspeed")
    if isinstance(deepspeed, (str, Path)) and str(deepspeed).strip():
        training["deepspeed"] = str(resolve_path(str(deepspeed)))
    elif not deepspeed:
        training.pop("deepspeed", None)
    if not has_eval_dataset:
        training["eval_strategy"] = "no"
        training["evaluation_strategy"] = "no"
    training = filter_supported_kwargs(TrainingArguments.__init__, training)
    return TrainingArguments(**training)


def enable_gradient_checkpointing(model: Any, config: dict[str, Any]) -> None:
    if not bool(config.get("gradient_checkpointing", True)):
        return
    model.config.use_cache = False
    training = config.get("training", {})
    kwargs = config.get("gradient_checkpointing_kwargs")
    if kwargs is None and isinstance(training, dict):
        kwargs = training.get("gradient_checkpointing_kwargs")
    if isinstance(kwargs, dict) and kwargs:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=kwargs
            )
            return
        except TypeError:
            pass
    model.gradient_checkpointing_enable()


def print_parameter_counts(model: Any) -> None:
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    percentage = 100.0 * trainable / total if total else 0.0
    print(
        f"Trainable parameters: {trainable:,}/{total:,} ({percentage:.2f}%)."
    )
    if trainable != total:
        raise SystemExit(
            "Full SFT requires every model parameter to be trainable, but some "
            "parameters are frozen."
        )


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
    training_args = build_training_args(
        config, transformers.TrainingArguments, eval_dataset is not None
    )

    model_kwargs = build_sft_model_kwargs(config, torch)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        **model_kwargs,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    enable_gradient_checkpointing(model, config)
    print_parameter_counts(model)

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
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
    print(f"Saved full SFT model to {training_args.output_dir}")


if __name__ == "__main__":
    main()
