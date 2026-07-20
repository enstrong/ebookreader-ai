#!/usr/bin/env python3
"""Evaluate thresholded ALS/content hybrid experiments in one validation pass."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_hybrid_grid_validation_split import (
    DEFAULT_EXPERIMENTS,
    blocked_items,
    build_metadata_indexes,
    component_scores,
    empty_stats,
    finalize_results,
    rough_content_candidates,
    weighted_content_scores,
)
from evaluate_hybrid_validation_split import (
    EXPERIMENTS,
    dynamic_alpha,
    load_artifacts,
    load_metadata,
    minmax,
    profile_sources,
    reciprocal_rank,
)


def threshold_label(value: float) -> str:
    return f"threshold_{value:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/recommendations/experiments/als_reads_20k_f256_i10_lam1p0_validation_split"),
    )
    parser.add_argument("--validation", type=Path, default=Path("data/recommendations/interactions_with_reads_validation_10k.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/recommendations/hybrid/book_metadata_20k.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("data/recommendations/hybrid/strategy_experiments"))
    parser.add_argument("--candidate-pool", type=int, default=1000)
    parser.add_argument("--content-candidate-pool", type=int, default=500)
    parser.add_argument("--content-min-scores", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    parser.add_argument("--content-prepool", type=int, default=3000)
    parser.add_argument("--max-profile-books", type=int, default=50)
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS), default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 20, 50])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cutoffs = sorted(args.cutoffs)
    model, mappings, user_items = load_artifacts(args.model_dir)
    validation = pd.read_csv(args.validation)
    metadata, metadata_stats = load_metadata(args.metadata)

    user_to_idx = mappings["user_to_idx"]
    book_to_idx = mappings["book_to_idx"]
    idx_to_book = mappings["idx_to_book"]
    metadata_coverage = sum(1 for book_id in book_to_idx if int(book_id) in metadata) / len(book_to_idx)
    author_index, genre_index, popular_items = build_metadata_indexes(metadata, book_to_idx)
    configs = [EXPERIMENTS[name] for name in args.experiments]
    thresholds = sorted(set(float(value) for value in args.content_min_scores))
    total_candidate_pool = args.candidate_pool + args.content_candidate_pool
    max_k = max(cutoffs)

    stats_by_threshold = {
        threshold: {config.name: empty_stats(config.name, cutoffs) for config in configs}
        for threshold in thresholds
    }
    skipped_unknown_user = 0
    skipped_unknown_book = 0
    evaluated = 0
    start = time.time()

    for row in validation.itertuples(index=False):
        user_id = int(row.user_id)
        hidden_book = int(row.goodreads_book_id)
        if user_id not in user_to_idx:
            skipped_unknown_user += 1
            continue
        if hidden_book not in book_to_idx:
            skipped_unknown_book += 1
            continue

        hidden_item_idx = int(book_to_idx[hidden_book])
        user_idx = int(user_to_idx[user_id])
        item_indices, als_scores = model.recommend(
            0,
            user_items[user_idx],
            N=args.candidate_pool,
            filter_already_liked_items=True,
            recalculate_user=True,
        )
        item_indices = np.asarray(item_indices, dtype=np.int32)
        als_scores = np.asarray(als_scores, dtype=np.float32)
        als_score_by_item = {int(item_idx): float(score) for item_idx, score in zip(item_indices, als_scores)}
        min_als_score = float(np.min(als_scores)) if len(als_scores) else 0.0
        als_top = [int(item_idx) for item_idx in item_indices[:max_k]]
        als_rr = reciprocal_rank(als_top, hidden_item_idx)

        sources = profile_sources(user_items[user_idx], idx_to_book, metadata, args.max_profile_books)
        blocked = blocked_items(user_items[user_idx])
        content_seed_indices = rough_content_candidates(
            sources,
            metadata,
            book_to_idx,
            author_index,
            genre_index,
            popular_items,
            blocked,
            args.content_prepool,
        )
        content_seed_components = component_scores(content_seed_indices, sources, idx_to_book, metadata)
        evaluated += 1

        for config in configs:
            alpha = dynamic_alpha(len(sources)) if config.alpha_policy == "dynamic" else float(config.alpha)
            content_seed_scores = weighted_content_scores(content_seed_components, config.weights)
            positive_scores = [
                (int(item_idx), float(score))
                for item_idx, score in zip(content_seed_indices, content_seed_scores)
                if score > 0
            ]

            for threshold in thresholds:
                stats = stats_by_threshold[threshold][config.name]
                if args.content_candidate_pool > 0 and positive_scores:
                    content_score_by_item = {
                        item_idx: score
                        for item_idx, score in positive_scores
                        if score >= threshold
                    }
                    content_top_items = [
                        item_idx
                        for item_idx, _ in sorted(content_score_by_item.items(), key=lambda pair: pair[1], reverse=True)
                        if item_idx not in als_score_by_item
                    ][: args.content_candidate_pool]
                else:
                    content_score_by_item = {}
                    content_top_items = []

                union_items = list(dict.fromkeys([int(item_idx) for item_idx in item_indices] + content_top_items))
                union_indices = np.asarray(union_items, dtype=np.int32)
                als_union_scores = np.asarray(
                    [als_score_by_item.get(item_idx, min_als_score) for item_idx in union_items],
                    dtype=np.float32,
                )
                content_union_scores = np.asarray(
                    [content_score_by_item.get(item_idx, 0.0) for item_idx in union_items],
                    dtype=np.float32,
                )
                final_scores = alpha * minmax(als_union_scores) + (1.0 - alpha) * minmax(content_union_scores)
                order = np.argsort(-final_scores, kind="stable")
                recommendations = [int(union_indices[index]) for index in order[:max_k]]
                if hidden_item_idx in union_items:
                    stats["hidden_in_candidate_pool"] += 1

                for k in cutoffs:
                    if hidden_item_idx in recommendations[:k]:
                        stats["hits"][k] += 1

                hybrid_rr = reciprocal_rank(recommendations, hidden_item_idx)
                stats["reciprocal_ranks"].append(hybrid_rr)
                stats["alpha_values"].append(alpha)
                stats["profile_counts"].append(len(sources))

                hybrid_rank = int(1 / hybrid_rr) if hybrid_rr else None
                als_rank = int(1 / als_rr) if als_rr else None
                if hybrid_rank != als_rank:
                    stats["changed_rank_count"] += 1
                    if len(stats["examples"]) < 12:
                        stats["examples"].append(
                            {
                                "user_id": user_id,
                                "hidden_goodreads_book_id": hidden_book,
                                "als_rank": als_rank,
                                "hybrid_rank": hybrid_rank,
                                "alpha": alpha,
                                "positive_profile_count": len(sources),
                                "als_top_recommendations": [int(idx_to_book[item_idx]) for item_idx in als_top[:10]],
                                "hybrid_top_recommendations": [
                                    int(idx_to_book[item_idx]) for item_idx in recommendations[:10]
                                ],
                            }
                        )

        if evaluated % 500 == 0:
            print(f"Threshold grid evaluated {evaluated:,} validation rows", flush=True)

    runtime = time.time() - start
    args.output_root.mkdir(parents=True, exist_ok=True)
    for threshold in thresholds:
        label = threshold_label(threshold)
        output_dir = args.output_root / label
        output_dir.mkdir(parents=True, exist_ok=True)
        log_rows = []

        for config in configs:
            result = finalize_results(
                stats_by_threshold[threshold][config.name],
                config,
                len(validation),
                evaluated,
                skipped_unknown_user,
                skipped_unknown_book,
                stats_by_threshold[threshold][config.name]["hidden_in_candidate_pool"],
                metadata_coverage,
                metadata_stats,
                total_candidate_pool,
                args.max_profile_books,
                cutoffs,
            )
            result.update(
                {
                    "method": "als_content_hybrid_threshold_grid",
                    "candidate_pool": args.candidate_pool if not config.weights else total_candidate_pool,
                    "candidate_strategy": "als_only" if not config.weights else "als_plus_thresholded_content",
                    "als_candidate_pool": args.candidate_pool,
                    "content_candidate_pool": 0 if not config.weights else args.content_candidate_pool,
                    "content_min_score": 0.0 if not config.weights else threshold,
                    "content_prepool": args.content_prepool,
                    "model_dir": str(args.model_dir),
                    "validation": str(args.validation),
                    "metadata": str(args.metadata),
                    "cutoffs": cutoffs,
                }
            )
            output = output_dir / f"{config.name}.json"
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            log_rows.append(
                {
                    "experiment": config.name,
                    "candidate_pool": total_candidate_pool,
                    "als_candidate_pool": args.candidate_pool,
                    "content_candidate_pool": args.content_candidate_pool,
                    "content_min_score": threshold,
                    "max_profile_books": args.max_profile_books,
                    "status": "success",
                    "runtime_seconds": f"{runtime:.3f}",
                    "output": str(output),
                    "log": "threshold_grid",
                }
            )

        run_log = args.output_root / f"{label}_run_log.csv"
        with run_log.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "experiment",
                    "candidate_pool",
                    "als_candidate_pool",
                    "content_candidate_pool",
                    "content_min_score",
                    "max_profile_books",
                    "status",
                    "runtime_seconds",
                    "output",
                    "log",
                ],
            )
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"Wrote {len(configs)} experiment JSON files to {output_dir}", flush=True)
        print(f"Wrote run log: {run_log}", flush=True)


if __name__ == "__main__":
    main()
