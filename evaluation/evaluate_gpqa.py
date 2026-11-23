from __future__ import annotations

try:
    from .evaluate_multiple_choice import BenchmarkSpec, PROJECT_ROOT, run
except ImportError:
    from evaluate_multiple_choice import BenchmarkSpec, PROJECT_ROOT, run


SPEC = BenchmarkSpec(
    key="gpqa",
    display_name="GPQA Diamond",
    dataset="Idavidrein/gpqa",
    dataset_config="gpqa_diamond",
    split="train",
    output=PROJECT_ROOT / "evaluation/results/gpqa_diamond.json",
    system_prompt="Solve the multiple-choice problem carefully.",
    default_max_tokens=16384,
)


if __name__ == "__main__":
    run(SPEC)
