"""Evaluate cross-model similarity of reconstructed reasoning traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
JUDGE_SCRIPT = PROJECT_ROOT / "evaluation/evaluate_cot_llm_judge.py"
DEFAULT_PROFILES = (
    "deepseek_v4_pro",
    "kimi_k2p5",
    "glm_5_2",
    "qwen3_235b_a22b_thinking_2507",
    "xiaomi_mimo_v2_5_pro",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "generate", "judge"), default="all")
    parser.add_argument("--pipeline-dir", type=Path, default=Path("evaluation/llm_judge/cross_model_similarity"))
    parser.add_argument("--config", type=Path, default=Path("configs/model_configs.json"))
    parser.add_argument("--target-profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--reconstruction-model", help="Served name of the DPO policy")
    parser.add_argument("--reconstruction-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--judge-profile", default="gpt_5_6_luna")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--generation-workers", type=int, default=4)
    args = parser.parse_args()
    if len(args.target_profiles) < 2 or len(set(args.target_profiles)) != len(args.target_profiles):
        parser.error("--target-profiles must contain at least two unique names")
    if args.stage != "judge" and not args.reconstruction_model:
        parser.error("--reconstruction-model is required for generation")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.workers <= 0 or args.generation_workers <= 0:
        parser.error("worker counts must be positive")
    return args


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def call_judge(args: list[str]) -> None:
    subprocess.run([sys.executable, str(JUDGE_SCRIPT), *args], cwd=PROJECT_ROOT, check=True)


def tally(row: dict) -> dict[str, int]:
    return {key: int(row[key]) for key in ("wins", "losses", "ties")}


def with_rate(counts: dict[str, int]) -> dict[str, int | float | None]:
    total = sum(counts.values())
    return {**counts, "comparisons": total,
            "preference_rate": (counts["wins"] + 0.5 * counts["ties"]) / total if total else None}


def main() -> None:
    args = parse_args()
    pipeline_dir = absolute(args.pipeline_dir)
    common = ["--config", str(absolute(args.config))]
    if args.stage in {"all", "generate"}:
        command = [
            "--cross-reconstruction", "--pipeline-stage", "generate",
            "--local-only-generation", "--pipeline-dir", str(pipeline_dir),
            "--target-profiles", *args.target_profiles,
            "--reconstruction-model", args.reconstruction_model,
            "--reconstruction-base-url", args.reconstruction_base_url,
            "--generation-workers", str(args.generation_workers),
            "--workers", str(args.workers), *common,
        ]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        call_judge(command)
    if args.stage == "generate":
        return

    matrix: dict[str, dict[str, dict]] = {}
    for reference in args.target_profiles:
        reference_file = pipeline_dir / "targets" / reference / "reference.json"
        matching_file = pipeline_dir / "targets" / reference / "candidates/local_reconstruction.json"
        if not reference_file.is_file() or not matching_file.is_file():
            raise SystemExit(f"Missing generated data for {reference}; run --stage generate first")
        aggregate = {"wins": 0, "losses": 0, "ties": 0}
        row: dict[str, dict] = {}
        for alternative in args.target_profiles:
            if alternative == reference:
                continue
            other_file = pipeline_dir / "targets" / alternative / "candidates/local_reconstruction.json"
            if not other_file.is_file():
                raise SystemExit(f"Missing generated data: {other_file}")
            output = pipeline_dir / "judgments" / f"{reference}_vs_{alternative}.json"
            command = [
                str(matching_file), str(other_file),
                "--reference", str(reference_file),
                "--candidate-labels", reference, alternative,
                "--anchor-label", reference,
                "--reference-label", reference,
                "--min-boxed-answer-agreement", "2",
                "--profile", args.judge_profile,
                "--workers", str(args.workers),
                "--output", str(output), "--resume", *common,
            ]
            if args.limit is not None:
                command += ["--limit", str(args.limit)]
            call_judge(command)
            summary_path = output.with_suffix(".summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            ranking = {entry["label"]: entry for entry in summary["ranking"]}
            matched = tally(ranking[reference])
            for key in aggregate:
                aggregate[key] += matched[key]
            row[alternative] = with_rate(tally(ranking[alternative]))
        row[reference] = with_rate(aggregate)
        matrix[reference] = {name: row[name] for name in args.target_profiles}

    result = {
        "metric": "(wins + 0.5 * ties) / (wins + losses + ties)",
        "answer_filter": "reference and both response sources have the same normalized boxed answer",
        "judge_profile": args.judge_profile,
        "profiles": args.target_profiles,
        "matrix": matrix,
    }
    output = pipeline_dir / "cross_model_similarity_matrix.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
