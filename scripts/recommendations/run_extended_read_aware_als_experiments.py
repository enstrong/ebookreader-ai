#!/usr/bin/env python3
"""Run the extended read-aware ALS experiments requested after the first grid."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
INTERACTIONS = REPO_ROOT / "data" / "recommendations" / "interactions_with_reads.csv"
EXPERIMENT_ROOT = REPO_ROOT / "data" / "recommendations" / "experiments"
STRICT_5K_SIMILARITIES = EXPERIMENT_ROOT / "strict_rating5_top5000.similar.csv"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "recommendations" / "train_als.py"
TRAIN_FROM_ARTIFACTS_SCRIPT = REPO_ROOT / "scripts" / "recommendations" / "train_als_from_artifacts.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "recommendations" / "evaluate_als_holdout.py"
SOURCE_20K_ARTIFACT_DIR = REPO_ROOT / "data" / "recommendations" / "als_reads_20k"


@dataclass(frozen=True)
class Experiment:
    purpose: str
    top_books: int
    factors: int
    regularization: float
    iterations: int = 20

    @property
    def slug(self) -> str:
        lambda_slug = str(self.regularization).replace(".", "p")
        return f"als_reads_{self.top_books // 1000}k_f{self.factors}_i{self.iterations}_lam{lambda_slug}"

    @property
    def output_dir(self) -> Path:
        return EXPERIMENT_ROOT / self.slug


HIGH_FEATURE_20K = [
    Experiment("high_feature_20k", 20_000, 192, 0.1),
    Experiment("high_feature_20k", 20_000, 192, 0.2),
    Experiment("high_feature_20k", 20_000, 256, 0.1),
    Experiment("high_feature_20k", 20_000, 256, 0.2),
]


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def run_command(command: list[str], log_path: Path, dry_run: bool) -> tuple[str, float]:
    start = time.monotonic()
    printable = " ".join(command)
    if dry_run:
        print(f"[dry-run] {printable}", flush=True)
        return "dry_run", 0.0

    print(f"[run] {printable}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {printable}\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - start
    status = "success" if completed.returncode == 0 else f"failed:{completed.returncode}"
    print(f"[{status}] {log_path} in {format_seconds(elapsed)}", flush=True)
    return status, elapsed


def append_run_log(
    run_log_path: Path,
    experiment: Experiment,
    step: str,
    status: str,
    elapsed: float,
    output_path: Path,
    log_path: Path,
) -> None:
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not run_log_path.exists()
    with run_log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model_name",
                "purpose",
                "step",
                "top_books",
                "factors",
                "lambda",
                "iterations",
                "status",
                "runtime_seconds",
                "runtime_hhmmss",
                "output_path",
                "log_path",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "model_name": experiment.slug,
                "purpose": experiment.purpose,
                "step": step,
                "top_books": experiment.top_books,
                "factors": experiment.factors,
                "lambda": experiment.regularization,
                "iterations": experiment.iterations,
                "status": status,
                "runtime_seconds": round(elapsed, 3),
                "runtime_hhmmss": format_seconds(elapsed),
                "output_path": str(output_path.relative_to(REPO_ROOT)),
                "log_path": str(log_path.relative_to(REPO_ROOT)),
            }
        )


def completed_training(experiment: Experiment) -> bool:
    return all(
        (experiment.output_dir / name).exists()
        for name in ["metadata.json", "als_model.pkl", "mappings.pkl", "user_items.npz"]
    )


def train_from_artifacts_command(experiment: Experiment, source_dir: Path) -> list[str]:
    return [
        str(PYTHON),
        str(TRAIN_FROM_ARTIFACTS_SCRIPT),
        "--source-dir",
        str(source_dir.relative_to(REPO_ROOT)),
        "--output-dir",
        str(experiment.output_dir.relative_to(REPO_ROOT)),
        "--factors",
        str(experiment.factors),
        "--iterations",
        str(experiment.iterations),
        "--regularization",
        str(experiment.regularization),
    ]


def train_from_interactions_command(experiment: Experiment) -> list[str]:
    return [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--interactions",
        str(INTERACTIONS.relative_to(REPO_ROOT)),
        "--output-dir",
        str(experiment.output_dir.relative_to(REPO_ROOT)),
        "--signal-mode",
        "mean-centered",
        "--top-books",
        str(experiment.top_books),
        "--factors",
        str(experiment.factors),
        "--iterations",
        str(experiment.iterations),
        "--regularization",
        str(experiment.regularization),
        "--min-user-likes",
        "5",
        "--mean-shrinkage",
        "5",
    ]


def eval_command(experiment: Experiment, eval_name: str) -> list[str]:
    output_name = "evaluation_holdout_level2_5k.json" if eval_name == "level2_5k" else f"evaluation_holdout_{eval_name}.json"
    command = [
        str(PYTHON),
        str(EVAL_SCRIPT),
        "--model-dir",
        str(experiment.output_dir.relative_to(REPO_ROOT)),
        "--interactions",
        str(INTERACTIONS.relative_to(REPO_ROOT)),
        "--output",
        str((experiment.output_dir / output_name).relative_to(REPO_ROOT)),
        "--max-users",
        "10000",
        "--min-likes",
        "5",
        "--min-like-rating",
        "5",
    ]
    if eval_name == "level2_5k":
        command.extend(["--allowed-similarities", str(STRICT_5K_SIMILARITIES.relative_to(REPO_ROOT))])
    return command


def run_training(
    experiment: Experiment,
    command: list[str],
    run_log_path: Path,
    dry_run: bool,
) -> None:
    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    train_log = experiment.output_dir / "train.log"
    if completed_training(experiment):
        print(f"[skip] training artifacts already exist: {experiment.slug}", flush=True)
        append_run_log(run_log_path, experiment, "train", "skipped_existing", 0.0, experiment.output_dir, train_log)
        return
    status, elapsed = run_command(command, train_log, dry_run)
    append_run_log(run_log_path, experiment, "train", status, elapsed, experiment.output_dir, train_log)
    if status.startswith("failed"):
        raise SystemExit(f"Training failed for {experiment.slug}; see {train_log}")


def run_evaluations(experiment: Experiment, run_log_path: Path, dry_run: bool) -> None:
    for eval_name in [f"{experiment.top_books // 1000}k", "level2_5k"]:
        output_name = "evaluation_holdout_level2_5k.json" if eval_name == "level2_5k" else f"evaluation_holdout_{eval_name}.json"
        output_path = experiment.output_dir / output_name
        eval_log = experiment.output_dir / f"evaluate_{eval_name}.log"
        if output_path.exists():
            print(f"[skip] evaluation exists: {output_path}", flush=True)
            append_run_log(run_log_path, experiment, f"evaluate_{eval_name}", "skipped_existing", 0.0, output_path, eval_log)
            continue
        status, elapsed = run_command(eval_command(experiment, eval_name), eval_log, dry_run)
        append_run_log(run_log_path, experiment, f"evaluate_{eval_name}", status, elapsed, output_path, eval_log)
        if status.startswith("failed"):
            raise SystemExit(f"Evaluation failed for {experiment.slug} {eval_name}; see {eval_log}")


def read_mrr(experiment: Experiment) -> float:
    path = experiment.output_dir / "evaluation_holdout_20k.json"
    with path.open(encoding="utf-8") as file:
        return float(json.load(file)["mean_reciprocal_rank"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-log",
        type=Path,
        default=EXPERIMENT_ROOT / "extended_read_aware_als_run_log.csv",
    )
    args = parser.parse_args()

    for required in [PYTHON, INTERACTIONS, STRICT_5K_SIMILARITIES, SOURCE_20K_ARTIFACT_DIR / "user_items.npz"]:
        if not required.exists():
            raise SystemExit(f"Required artifact not found: {required}")

    for index, experiment in enumerate(HIGH_FEATURE_20K, start=1):
        print(f"\n=== 20k high-feature {index}/{len(HIGH_FEATURE_20K)} {experiment.slug} ===", flush=True)
        run_training(
            experiment,
            train_from_artifacts_command(experiment, SOURCE_20K_ARTIFACT_DIR),
            args.run_log,
            args.dry_run,
        )
        run_evaluations(experiment, args.run_log, args.dry_run)

    if args.dry_run:
        best_20k = HIGH_FEATURE_20K[-1]
    else:
        best_20k = max(HIGH_FEATURE_20K, key=read_mrr)
    print(f"\nBest 20k high-feature model by MRR: {best_20k.slug}", flush=True)

    best_50k = Experiment("best_high_feature_50k", 50_000, best_20k.factors, best_20k.regularization)
    comparison_50k = Experiment("comparison_50k", 50_000, 128, 0.1)
    fifty_k_experiments = [best_50k]
    if comparison_50k.slug != best_50k.slug:
        fifty_k_experiments.append(comparison_50k)

    print(f"\n=== 50k model 1/{len(fifty_k_experiments)} {best_50k.slug} ===", flush=True)
    run_training(best_50k, train_from_interactions_command(best_50k), args.run_log, args.dry_run)
    run_evaluations(best_50k, args.run_log, args.dry_run)

    if comparison_50k.slug != best_50k.slug:
        print(f"\n=== 50k model 2/{len(fifty_k_experiments)} {comparison_50k.slug} ===", flush=True)
        run_training(
            comparison_50k,
            train_from_artifacts_command(comparison_50k, best_50k.output_dir),
            args.run_log,
            args.dry_run,
        )
        run_evaluations(comparison_50k, args.run_log, args.dry_run)


if __name__ == "__main__":
    main()
