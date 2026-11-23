from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any

try:
    from .lora_dpo import (
        build_dpo_config,
        get_local_rank,
        load_config,
        load_dpo_rows,
        make_dataset,
        require_package,
        resolve_path,
        resolve_resume_checkpoint,
        torch_dtype_from_string,
        validate_model_name_or_path,
    )
except ImportError:
    from lora_dpo import (  # type: ignore[no-redef]
        build_dpo_config,
        get_local_rank,
        load_config,
        load_dpo_rows,
        make_dataset,
        require_package,
        resolve_path,
        resolve_resume_checkpoint,
        torch_dtype_from_string,
        validate_model_name_or_path,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs/full_dpo_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-parameter Direct Preference Optimization (DPO)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Full DPO config JSON path (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument("--data-path", type=Path, help="Override DPO data JSON path.")
    parser.add_argument("--model", help="Override base model path/name.")
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument("--max-samples", type=int, help="Use at most this many rows.")
    parser.add_argument(
        "--max-length",
        type=int,
        help="Override DPO sequence length and overlong-sample filter threshold.",
    )
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


def apply_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    config = dict(config)
    if args.data_path:
        config["data_path"] = str(args.data_path)
    if args.model:
        config["model_name_or_path"] = args.model
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples
    if args.seed is not None:
        config["seed"] = args.seed

    training = dict(config.get("training", {}))
    if args.output_dir:
        training["output_dir"] = str(args.output_dir)
    if args.max_length is not None:
        training["max_length"] = args.max_length
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


def build_full_model_kwargs(config: dict[str, Any], torch: Any) -> dict[str, Any]:
    model_kwargs = dict(config.get("model_kwargs", {}))
    if "torch_dtype" in model_kwargs:
        model_kwargs["torch_dtype"] = torch_dtype_from_string(
            torch, model_kwargs["torch_dtype"]
        )

    quantization = dict(config.get("quantization", {}))
    if (
        quantization.get("load_in_4bit")
        or quantization.get("load_in_8bit")
        or model_kwargs.get("load_in_4bit")
        or model_kwargs.get("load_in_8bit")
        or "quantization_config" in model_kwargs
    ):
        raise SystemExit(
            "Full DPO cannot train a bitsandbytes-quantized base model. Remove "
            "the quantization settings or use src/lora_dpo.py for QLoRA DPO."
        )
    return model_kwargs


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


def verify_full_parameter_training(model: Any) -> None:
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    percentage = 100.0 * trainable / total if total else 0.0
    print(f"Trainable parameters: {trainable:,}/{total:,} ({percentage:.2f}%).")
    if trainable != total:
        raise SystemExit(
            "Full DPO requires every policy-model parameter to be trainable, "
            "but some parameters are frozen."
        )


def filter_overlong_rows(
    rows: list[dict[str, str]],
    tokenizer: Any,
    max_length: int,
    batch_size: int = 32,
) -> list[dict[str, str]]:
    if max_length <= 0:
        raise SystemExit("training.max_length must be positive.")
    if batch_size <= 0:
        raise SystemExit("length_filter_batch_size must be positive.")
    eos_token = tokenizer.eos_token
    if not isinstance(eos_token, str) or not eos_token:
        raise SystemExit(
            "drop_overlong_samples requires a tokenizer with a non-empty eos_token."
        )

    kept: list[dict[str, str]] = []
    dropped = 0
    longest_kept = 0
    shortest_dropped: int | None = None
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [row["prompt"] for row in batch]
        chosens = [
            row["chosen"]
            if row["chosen"].endswith(eos_token)
            else row["chosen"] + eos_token
            for row in batch
        ]
        rejecteds = [
            row["rejected"]
            if row["rejected"].endswith(eos_token)
            else row["rejected"] + eos_token
            for row in batch
        ]
        chosen_ids = tokenizer(
            [prompt + chosen for prompt, chosen in zip(prompts, chosens)],
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        rejected_ids = tokenizer(
            [prompt + rejected for prompt, rejected in zip(prompts, rejecteds)],
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]

        for row, chosen, rejected in zip(batch, chosen_ids, rejected_ids):
            pair_length = max(len(chosen), len(rejected))
            if pair_length <= max_length:
                kept.append(row)
                longest_kept = max(longest_kept, pair_length)
            else:
                dropped += 1
                if shortest_dropped is None:
                    shortest_dropped = pair_length
                else:
                    shortest_dropped = min(shortest_dropped, pair_length)

    if not kept:
        raise SystemExit(
            f"All {len(rows)} DPO rows exceed training.max_length={max_length}."
        )
    print(
        f"Length filter max_length={max_length}: kept {len(kept)}/{len(rows)} "
        f"rows; dropped {dropped} ({dropped / len(rows):.1%})."
    )
    print(
        f"Longest kept pair: {longest_kept} tokens; "
        f"shortest dropped pair: {shortest_dropped or 'n/a'} tokens."
    )
    return kept


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
    print(f"Using model: {model_name}")

    seed = int(config.get("seed", 42))
    rows = load_dpo_rows(
        resolve_path(data_path_value),
        max_samples=config.get("max_samples"),
        seed=seed,
    )

    torch = require_package("torch")
    transformers = require_package("transformers")
    trl = require_package("trl")

    local_rank = get_local_rank(args)
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = str(config.get("padding_side", "left"))

    if bool(config.get("drop_overlong_samples", False)):
        training = config.get("training", {})
        if not isinstance(training, dict) or training.get("max_length") is None:
            raise SystemExit(
                "drop_overlong_samples requires training.max_length in the config."
            )
        rows = filter_overlong_rows(
            rows,
            tokenizer,
            max_length=int(training["max_length"]),
            batch_size=int(config.get("length_filter_batch_size", 32)),
        )
    train_dataset, eval_dataset = make_dataset(rows, config)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        **build_full_model_kwargs(config, torch),
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    enable_gradient_checkpointing(model, config)
    verify_full_parameter_training(model)

    dpo_args = build_dpo_config(config, trl.DPOConfig, eval_dataset is not None)
    trainer_kwargs = {
        "model": model,
        "args": dpo_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    trainer_signature = inspect.signature(trl.DPOTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    # With no PEFT config and no explicit reference model, DPOTrainer creates a
    # frozen copy of the initial policy and uses it as the DPO reference model.
    trainer = trl.DPOTrainer(**trainer_kwargs)
    resume_checkpoint = resolve_resume_checkpoint(
        args.resume_from_checkpoint, dpo_args.output_dir
    )
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(dpo_args.output_dir)
    print(f"Saved full DPO model to {dpo_args.output_dir}")


if __name__ == "__main__":
    main()
