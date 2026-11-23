from __future__ import annotations

try:
    from .evaluate_multiple_choice import BenchmarkSpec, PROJECT_ROOT, run
except ImportError:
    from evaluate_multiple_choice import BenchmarkSpec, PROJECT_ROOT, run


SPEC = BenchmarkSpec(
    key="openbookqa",
    display_name="OpenBookQA",
    dataset="allenai/openbookqa",
    dataset_config="main",
    split="test",
    output=PROJECT_ROOT / "evaluation/results/openbookqa.json",
    system_prompt="Solve the multiple-choice problem carefully.",
    default_max_tokens=8192,
)


if __name__ == "__main__":
    run(SPEC)
