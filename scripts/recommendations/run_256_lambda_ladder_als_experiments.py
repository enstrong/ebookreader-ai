#!/usr/bin/env python3
"""Run the 256-feature lambda ladder for read-aware ALS."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
INTERACTIONS = REPO_ROOT / "data" / "recommendations" / "interactions_with_reads.csv"
EXPERIMENT_ROOT = REPO_ROOT / "data" / "recommendations" / "experiments"
STRICT_5K_SIMILARITIES = EXPERIMENT_ROOT / "strict_rating5_top5000.similar.csv"
TRAIN_FROM_ARTIFACTS_SCRIPT = REPO_ROOT / "scripts" / "recommendations" / "train_als_from_artifacts.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "recommendations" / "evaluate_als_holdout.py"
SOURCE_20K_ARTIFACT_DIR = REPO_ROOT / "data" / "recommendations" / "als_reads_20k"


@dataclass(frozen=True)
class Experiment:
    factors: int
    regularization: float
    iterations: int = 20
    top_books: int = 20_000

    @property
    def slug(self) -> str:
        lambda_slug = str(self.regularization).replace(".", "p")
        return f"als_reads_20k_f{self.factors}_i{self.iterations}_lam{lambda_slug}"

    @property
    def output_dir(self) -> Path:
        return EXPERIMENT_ROOT / self.slug


EXPERIMENTS = [
    Experiment(256, 0.01),
    Experiment(256, 0.5),
    Experiment(256, 0.8),
    Experiment(256, 1.0),
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
                "purpose": "lambda_ladder_256f",
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


def train_command(experiment: Experiment) -> list[str]:
    return [
        str(PYTHON),
        str(TRAIN_FROM_ARTIFACTS_SCRIPT),
        "--source-dir",
        str(SOURCE_20K_ARTIFACT_DIR.relative_to(REPO_ROOT)),
        "--output-dir",
        str(experiment.output_dir.relative_to(REPO_ROOT)),
        "--factors",
        str(experiment.factors),
        "--iterations",
        str(experiment.iterations),
        "--regularization",
        str(experiment.regularization),
    ]


def eval_command(experiment: Experiment, eval_name: str) -> list[str]:
    output_name = "evaluation_holdout_level2_5k.json" if eval_name == "level2_5k" else "evaluation_holdout_20k.json"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-log",
        type=Path,
        default=EXPERIMENT_ROOT / "lambda_ladder_256f_run_log.csv",
    )
    args = parser.parse_args()

    for required in [PYTHON, INTERACTIONS, STRICT_5K_SIMILARITIES, SOURCE_20K_ARTIFACT_DIR / "user_items.npz"]:
        if not required.exists():
            raise SystemExit(f"Required artifact not found: {required}")

    for index, experiment in enumerate(EXPERIMENTS, start=1):
        print(f"\n=== lambda ladder {index}/{len(EXPERIMENTS)} {experiment.slug} ===", flush=True)
        experiment.output_dir.mkdir(parents=True, exist_ok=True)

        train_log = experiment.output_dir / "train.log"
        if completed_training(experiment):
            print(f"[skip] training artifacts already exist: {experiment.slug}", flush=True)
            append_run_log(args.run_log, experiment, "train", "skipped_existing", 0.0, experiment.output_dir, train_log)
        else:
            status, elapsed = run_command(train_command(experiment), train_log, args.dry_run)
            append_run_log(args.run_log, experiment, "train", status, elapsed, experiment.output_dir, train_log)
            if status.startswith("failed"):
                raise SystemExit(f"Training failed for {experiment.slug}; see {train_log}")

        for eval_name in ["20k", "level2_5k"]:
            output_name = "evaluation_holdout_level2_5k.json" if eval_name == "level2_5k" else "evaluation_holdout_20k.json"
            output_path = experiment.output_dir / output_name
            eval_log = experiment.output_dir / f"evaluate_{eval_name}.log"
            if output_path.exists():
                print(f"[skip] evaluation exists: {output_path}", flush=True)
                append_run_log(args.run_log, experiment, f"evaluate_{eval_name}", "skipped_existing", 0.0, output_path, eval_log)
                continue
            status, elapsed = run_command(eval_command(experiment, eval_name), eval_log, args.dry_run)
            append_run_log(args.run_log, experiment, f"evaluate_{eval_name}", status, elapsed, output_path, eval_log)
            if status.startswith("failed"):
                raise SystemExit(f"Evaluation failed for {experiment.slug} {eval_name}; see {eval_log}")


if __name__ == "__main__":
    main()
