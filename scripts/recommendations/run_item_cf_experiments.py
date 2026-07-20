#!/usr/bin/env python3
"""Run repeatable Level 2 item-CF build/evaluation experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_EXPERIMENTS = [
    {
        "name": "baseline_rating4_top5000",
        "min_like_rating": 4,
        "top_books": 5000,
        "neighbors_per_book": 25,
        "min_co_likes": 5,
    },
    {
        "name": "strict_rating5_top5000",
        "min_like_rating": 5,
        "top_books": 5000,
        "neighbors_per_book": 25,
        "min_co_likes": 5,
    },
]


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, default=Path("data/recommendations/interactions_filtered.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/recommendations/experiments"))
    parser.add_argument("--max-users", type=int, default=10_000)
    parser.add_argument("--rebuild-existing", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Optional experiment names to run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        experiment
        for experiment in DEFAULT_EXPERIMENTS
        if args.only is None or experiment["name"] in args.only
    ]
    if not selected:
        raise SystemExit("No experiments selected.")

    summaries = []
    for experiment in selected:
        name = experiment["name"]
        similarities = args.output_dir / f"{name}.similar.csv"
        metadata = args.output_dir / f"{name}.metadata.json"
        evaluation = args.output_dir / f"{name}.evaluation.json"

        if args.rebuild_existing or not similarities.exists():
            run([
                sys.executable,
                "scripts/recommendations/build_item_cf.py",
                "--interactions", str(args.interactions),
                "--output", str(similarities),
                "--metadata", str(metadata),
                "--min-like-rating", str(experiment["min_like_rating"]),
                "--top-books", str(experiment["top_books"]),
                "--neighbors-per-book", str(experiment["neighbors_per_book"]),
                "--min-co-likes", str(experiment["min_co_likes"]),
            ])
        else:
            print(f"Skipping build for {name}; found {similarities}", flush=True)

        run([
            sys.executable,
            "scripts/recommendations/evaluate_item_cf.py",
            "--interactions", str(args.interactions),
            "--similarities", str(similarities),
            "--output", str(evaluation),
            "--max-users", str(args.max_users),
            "--min-like-rating", str(experiment["min_like_rating"]),
        ])

        results = load_json(evaluation)
        summaries.append({
            "name": name,
            "users_evaluated": results["users_evaluated"],
            "hit_rate": results["hit_rate"],
            "mean_reciprocal_rank": results["mean_reciprocal_rank"],
            "similarities": str(similarities),
            "evaluation": str(evaluation),
        })

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print("\nExperiment summary", flush=True)
    for summary in summaries:
        print(
            f"{summary['name']}: "
            f"HR@10={summary['hit_rate']['@10']:.4f}, "
            f"HR@50={summary['hit_rate']['@50']:.4f}, "
            f"MRR={summary['mean_reciprocal_rank']:.4f}",
            flush=True,
        )
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
