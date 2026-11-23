from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import random
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cot_prompt import upgrade_legacy_cot_prompt


DEFAULT_CONFIG = PROJECT_ROOT / "configs/lora_dpo_config.json"
REQUIRED_FIELDS = ("prompt", "chosen", "rejected")
PROMPT_RESPONSE_SEPARATOR = "\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter with DPO.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"DPO LoRA config JSON path (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument("--data-path", type=Path, help="Override DPO data JSON path.")
    parser.add_argument("--model", help="Override base model path/name.")
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument("--max-samples", type=int, help="Use at most this many rows.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument(
        "--deepspeed",
        help="Override DeepSpeed config path; use none/off/false/no to disable it.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="latest",
        help=(
            "Resume training from a checkpoint path. If passed without a value, "
            "the latest checkpoint in training.output_dir is used."
        ),
    )
    parser.add_argument(
        "--zero-stage",
        type=int,
        choices=(2, 3),
        help="Use the bundled DeepSpeed ZeRO stage config.",
    )
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank passed by DeepSpeed/torchrun launchers.",
    )
    return parser.parse_args()


def require_package(package: str, install_name: str | None = None) -> Any:
    try:
        # Import only the package itself.  Using ``fromlist=["*"]`` forces lazy
        # packages such as TRL to resolve every public trainer, including the
        # optional GRPO/vLLM stack that is not needed for DPO training.
        return importlib.import_module(package)
    except ImportError as exc:
        name = install_name or package
        raise SystemExit(
            f"Missing dependency '{name}'. Install it with: pip install {name}"
        ) from exc


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"{path} must contain a JSON object.")
    return config


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
    if args.seed is not None:
        config["seed"] = args.seed
        training = dict(config.get("training", {}))
        training["seed"] = args.seed
        config["training"] = training
    if args.zero_stage is not None:
        training = dict(config.get("training", {}))
        training["deepspeed"] = f"configs/ds_zero{args.zero_stage}.json"
        config["training"] = training
    if args.deepspeed is not None:
        training = dict(config.get("training", {}))
        deepspeed = args.deepspeed.strip()
        if deepspeed.lower() in {"", "none", "off", "false", "no"}:
            training.pop("deepspeed", None)
        else:
            training["deepspeed"] = deepspeed
        config["training"] = training
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_resume_checkpoint(value: str | None, output_dir: str | Path) -> str | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    output_path = Path(output_dir)
    if normalized in {"", "latest", "auto", "true", "yes"}:
        checkpoints = [
            path
            for path in output_path.glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
        ]
        if not checkpoints:
            raise SystemExit(f"No checkpoints found in {output_path}.")
        return str(max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[1])))

    checkpoint_path = Path(value)
    candidates = []
    if checkpoint_path.is_absolute():
        candidates.append(checkpoint_path)
    else:
        candidates.extend((resolve_path(checkpoint_path), output_path / checkpoint_path))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise SystemExit(f"Checkpoint does not exist: {value}")


def validate_model_name_or_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() and not path.exists():
        raise SystemExit(
            f"Configured local model path does not exist: {path}. "
            "Check model_name_or_path or the --model override."
        )
    return value


def load_dpo_rows(path: Path, max_samples: int | None, seed: int) -> list[dict[str, str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read DPO data {path}: {exc}") from exc
    if not isinstance(data, list):
        raise SystemExit(f"DPO data {path} must contain a JSON list.")

    rows: list[dict[str, str]] = []
    skipped = 0
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise SystemExit(f"{path}[{index}] must be a JSON object.")
        row = {field: str(item.get(field) or "").strip() for field in REQUIRED_FIELDS}
        if not all(row.values()):
            skipped += 1
            continue
        row["prompt"] = (
            upgrade_legacy_cot_prompt(row["prompt"]) + PROMPT_RESPONSE_SEPARATOR
        )
        if item.get("problem_id") is not None:
            row["problem_id"] = str(item["problem_id"])
        rows.append(row)

    if max_samples is not None:
        if max_samples <= 0:
            raise SystemExit("--max-samples must be positive.")
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_samples]

    if not rows:
        raise SystemExit(f"No valid DPO rows found in {path}.")
    print(f"Loaded {len(rows)} valid DPO rows from {path}; skipped {skipped}.")
    return rows


def make_dataset(rows: list[dict[str, str]], config: dict[str, Any]) -> tuple[Any, Any | None]:
    datasets = require_package("datasets")
    dataset = datasets.Dataset.from_list(rows)
    eval_ratio = float(config.get("eval_ratio", 0.0))
    seed = int(config.get("seed", 42))

    if eval_ratio <= 0:
        return dataset, None
    if not 0 < eval_ratio < 1:
        raise SystemExit("eval_ratio must be between 0 and 1.")
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    return split["train"], split["test"]


def torch_dtype_from_string(torch: Any, value: str | None) -> Any:
    if not value or value == "auto":
        return "auto"
    normalized = value.lower()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    dtype_name = aliases.get(normalized)
    if not dtype_name or not hasattr(torch, dtype_name):
        raise SystemExit(f"Unsupported torch_dtype: {value}")
    return getattr(torch, dtype_name)


def build_model_kwargs(config: dict[str, Any], torch: Any, transformers: Any) -> dict[str, Any]:
    model_kwargs = dict(config.get("model_kwargs", {}))
    if "torch_dtype" in model_kwargs:
        model_kwargs["torch_dtype"] = torch_dtype_from_string(
            torch, model_kwargs["torch_dtype"]
        )

    quantization = dict(config.get("quantization", {}))
    if quantization.get("load_in_4bit"):
        bnb_kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": bool(
                quantization.get("bnb_4bit_use_double_quant", True)
            ),
            "bnb_4bit_quant_type": quantization.get("bnb_4bit_quant_type", "nf4"),
        }
        compute_dtype = torch_dtype_from_string(
            torch, quantization.get("bnb_4bit_compute_dtype", "bfloat16")
        )
        if compute_dtype != "auto":
            bnb_kwargs["bnb_4bit_compute_dtype"] = compute_dtype
        model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            **bnb_kwargs
        )
    return model_kwargs


def filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def build_dpo_config(config: dict[str, Any], DPOConfig: Any, has_eval_dataset: bool) -> Any:
    training = dict(config.get("training", {}))
    if "output_dir" not in training:
        raise SystemExit("Config field training.output_dir is required.")
    training["output_dir"] = str(resolve_path(training["output_dir"]))
    deepspeed = training.get("deepspeed")
    if isinstance(deepspeed, str) and deepspeed:
        training["deepspeed"] = str(resolve_path(deepspeed))
    elif not deepspeed:
        training.pop("deepspeed", None)
    if not has_eval_dataset:
        training["eval_strategy"] = "no"
        training["evaluation_strategy"] = "no"
    training = filter_supported_kwargs(DPOConfig.__init__, training)
    return DPOConfig(**training)


def build_lora_config(config: dict[str, Any], LoraConfig: Any) -> Any:
    lora = dict(config.get("lora", {}))
    if not lora:
        raise SystemExit("Config field lora is required.")
    return LoraConfig(**lora)


def uses_4bit_quantization(config: dict[str, Any]) -> bool:
    return bool(dict(config.get("quantization", {})).get("load_in_4bit"))


def get_local_rank(args: argparse.Namespace) -> int:
    if args.local_rank >= 0:
        return args.local_rank
    try:
        return int(os.environ.get("LOCAL_RANK", "-1"))
    except ValueError:
        return -1


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
    train_dataset, eval_dataset = make_dataset(rows, config)

    torch = require_package("torch")
    transformers = require_package("transformers")
    trl = require_package("trl")
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
    tokenizer.padding_side = str(config.get("padding_side", "left"))

    model_kwargs = build_model_kwargs(config, torch, transformers)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        **model_kwargs,
    )
    if bool(config.get("gradient_checkpointing", True)):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    if uses_4bit_quantization(config):
        model = peft.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        )

    dpo_args = build_dpo_config(config, trl.DPOConfig, eval_dataset is not None)
    peft_config = build_lora_config(config, peft.LoraConfig)

    trainer_kwargs = {
        "model": model,
        "args": dpo_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
    }
    trainer_signature = inspect.signature(trl.DPOTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = trl.DPOTrainer(**trainer_kwargs)
    resume_checkpoint = resolve_resume_checkpoint(
        args.resume_from_checkpoint, dpo_args.output_dir
    )
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(dpo_args.output_dir)
    print(f"Saved LoRA DPO adapter to {dpo_args.output_dir}")


if __name__ == "__main__":
    main()
