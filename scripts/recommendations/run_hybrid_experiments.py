#!/usr/bin/env python3
"""Run the first ALS/content hybrid validation experiment grid."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

from evaluate_hybrid_validation_split import EXPERIMENTS


DEFAULT_EXPERIMENTS = [
    "als_baseline",
    "balanced_metadata",
    "author_50_baseline",
    "author_50_content_heavy",
    "author_50_collaborative_heavy",
    "genre_heavy",
    "author_genre_only",
    "rating_popularity_boosted",
    "language_aware",
    "dynamic_alpha_balanced",
    "dynamic_alpha_author_50",
]


def run_command(command: list[str], log_path: Path) -> tuple[str, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    runtime = time.time() - start
    return ("success" if process.returncode == 0 else f"failed:{process.returncode}"), runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/recommendations/experiments/als_reads_20k_f256_i10_lam1p0_validation_split"),
    )
    parser.add_argument("--validation", type=Path, default=Path("data/recommendations/interactions_with_reads_validation_10k.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/recommendations/hybrid/book_metadata_20k.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/recommendations/hybrid/experiments"))
    parser.add_argument("--run-log", type=Path, default=Path("data/recommendations/hybrid/hybrid_experiment_run_log.csv"))
    parser.add_argument("--candidate-pool", type=int, default=500)
    parser.add_argument("--max-profile-books", type=int, default=50)
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS), default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for experiment in args.experiments:
        output = args.output_dir / f"{experiment}.json"
        log = args.output_dir / f"{experiment}.log"
        if args.skip_existing and output.exists():
            status = "skipped_existing"
            runtime = 0.0
        else:
            command = [
                ".venv/bin/python",
                "scripts/recommendations/evaluate_hybrid_validation_split.py",
                "--model-dir",
                str(args.model_dir),
                "--validation",
                str(args.validation),
                "--metadata",
                str(args.metadata),
                "--experiment",
                experiment,
                "--candidate-pool",
                str(args.candidate_pool),
                "--max-profile-books",
                str(args.max_profile_books),
                "--output",
                str(output),
            ]
            print(f"Running {experiment}", flush=True)
            status, runtime = run_command(command, log)
            if not status.startswith("success"):
                print(f"{experiment} failed; see {log}", flush=True)
                raise SystemExit(1)

        rows.append(
            {
                "experiment": experiment,
                "candidate_pool": args.candidate_pool,
                "max_profile_books": args.max_profile_books,
                "status": status,
                "runtime_seconds": f"{runtime:.3f}",
                "output": str(output),
                "log": str(log),
            }
        )

    args.run_log.parent.mkdir(parents=True, exist_ok=True)
    with args.run_log.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "candidate_pool",
                "max_profile_books",
                "status",
                "runtime_seconds",
                "output",
                "log",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote run log: {args.run_log}", flush=True)


if __name__ == "__main__":
    main()
