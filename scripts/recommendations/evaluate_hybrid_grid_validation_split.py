#!/usr/bin/env python3
"""Evaluate all ALS/content hybrid reranking experiments in a single validation pass."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_hybrid_validation_split import (
    EXPERIMENTS,
    author_similarity,
    dynamic_alpha,
    jaccard,
    language_similarity,
    load_artifacts,
    load_metadata,
    minmax,
    page_similarity,
    profile_sources,
    rating_similarity,
    reciprocal_rank,
)


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
    "author_60_genre_40_a06",
    "author_60_genre_40_a07",
    "author_70_genre_30_a06",
    "author_70_genre_30_a07",
    "author_70_genre_30_a08",
    "author_80_genre_20_a06",
    "author_80_with_rest_a06",
    "author_90_genre_10_a06",
    "genre_60_author_40_a06",
    "genre_70_author_30_a06",
    "genre_80_author_20_a06",
    "genre_80_with_rest_a06",
    "genre_90_author_10_a06",
    "pure_author_a06",
    "pure_genre_a06",
]


def empty_stats(config_name: str, cutoffs: list[int]) -> dict:
    return {
        "experiment_name": config_name,
        "hits": {k: 0 for k in cutoffs},
        "reciprocal_ranks": [],
        "examples": [],
        "changed_rank_count": 0,
        "hidden_in_candidate_pool": 0,
        "alpha_values": [],
        "profile_counts": [],
    }


def component_scores(candidate_indices, sources, idx_to_book, metadata):
    length = len(candidate_indices)
    components = {
        "genre": np.zeros(length, dtype=np.float32),
        "author": np.zeros(length, dtype=np.float32),
        "rating": np.zeros(length, dtype=np.float32),
        "page": np.zeros(length, dtype=np.float32),
        "popularity": np.zeros(length, dtype=np.float32),
        "language": np.zeros(length, dtype=np.float32),
    }
    if not sources:
        return components

    source_meta = [(metadata[book_id], max(weight, 0.0)) for book_id, weight in sources if book_id in metadata]
    denominator = sum(weight for _, weight in source_meta)
    if denominator <= 0:
        return components

    for offset, item_idx in enumerate(candidate_indices):
        candidate_book = int(idx_to_book[int(item_idx)])
        candidate_meta = metadata.get(candidate_book)
        if candidate_meta is None:
            continue
        components["popularity"][offset] = candidate_meta.popularity_norm
        genre_total = 0.0
        author_total = 0.0
        rating_total = 0.0
        page_total = 0.0
        language_total = 0.0
        for source, source_weight in source_meta:
            genre_total += source_weight * jaccard(candidate_meta.genres, source.genres)
            author_total += source_weight * author_similarity(candidate_meta, source)
            rating_total += source_weight * rating_similarity(candidate_meta, source)
            page_total += source_weight * page_similarity(candidate_meta, source)
            language_total += source_weight * language_similarity(candidate_meta, source)
        components["genre"][offset] = genre_total / denominator
        components["author"][offset] = author_total / denominator
        components["rating"][offset] = rating_total / denominator
        components["page"][offset] = page_total / denominator
        components["language"][offset] = language_total / denominator
    return components


def weighted_content_scores(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    if not weights:
        first = next(iter(components.values()))
        return np.zeros(len(first), dtype=np.float32)
    scores = np.zeros(len(next(iter(components.values()))), dtype=np.float32)
    for key, weight in weights.items():
        scores += float(weight) * components[key]
    return scores


def build_metadata_indexes(metadata: dict, book_to_idx: dict[int, int]) -> tuple[dict[str, list[int]], dict[str, list[int]], list[int]]:
    author_index: dict[str, list[int]] = {}
    genre_index: dict[str, list[int]] = {}
    popular_items = []

    for book_id, item in metadata.items():
        if book_id not in book_to_idx:
            continue
        item_idx = int(book_to_idx[book_id])
        for author in item.author_key.split("|"):
            if author:
                author_index.setdefault(author, []).append(item_idx)
        for genre in item.genres:
            genre_index.setdefault(genre, []).append(item_idx)
        popular_items.append((item.popularity_norm, item_idx))

    return (
        author_index,
        genre_index,
        [item_idx for _, item_idx in sorted(popular_items, reverse=True)],
    )


def blocked_items(row) -> set[int]:
    start = row.indptr[0]
    end = row.indptr[1]
    return {int(item_idx) for item_idx in row.indices[start:end]}


def rough_content_candidates(
    sources,
    metadata,
    book_to_idx,
    author_index,
    genre_index,
    popular_items,
    blocked,
    prepool_limit,
) -> np.ndarray:
    scores: dict[int, float] = {}
    for book_id, source_weight in sources:
        source = metadata.get(book_id)
        if source is None:
            continue
        for author in source.author_key.split("|"):
            if not author:
                continue
            for item_idx in author_index.get(author, []):
                scores[item_idx] = scores.get(item_idx, 0.0) + 3.0 * source_weight
        genre_denominator = max(len(source.genres), 1)
        for genre in source.genres:
            for item_idx in genre_index.get(genre, []):
                scores[item_idx] = scores.get(item_idx, 0.0) + source_weight / genre_denominator

    for rank, item_idx in enumerate(popular_items[:1000]):
        scores[item_idx] = scores.get(item_idx, 0.0) + 0.01 * (1000 - rank) / 1000

    ranked = [
        item_idx
        for item_idx, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        if item_idx not in blocked
    ][:prepool_limit]
    return np.asarray(ranked, dtype=np.int32)


def finalize_results(
    stats: dict,
    config,
    validation_rows: int,
    evaluated: int,
    skipped_unknown_user: int,
    skipped_unknown_book: int,
    hidden_in_candidate_pool: int,
    metadata_coverage: float,
    metadata_stats: dict,
    candidate_pool: int,
    max_profile_books: int,
    cutoffs: list[int],
) -> dict:
    return {
        "experiment_name": config.name,
        "candidate_pool": candidate_pool,
        "max_profile_books": max_profile_books,
        "alpha_policy": config.alpha_policy,
        "alpha": config.alpha,
        "weights": config.weights,
        "validation_rows": validation_rows,
        "users_evaluated": evaluated,
        "skipped_unknown_user": skipped_unknown_user,
        "skipped_unknown_book": skipped_unknown_book,
        "hidden_in_candidate_pool": hidden_in_candidate_pool,
        "candidate_pool_recall": hidden_in_candidate_pool / evaluated if evaluated else 0.0,
        "changed_rank_count": stats["changed_rank_count"],
        "changed_rank_rate": stats["changed_rank_count"] / evaluated if evaluated else 0.0,
        "mean_alpha": sum(stats["alpha_values"]) / len(stats["alpha_values"]) if stats["alpha_values"] else 0.0,
        "mean_positive_profile_count": (
            sum(stats["profile_counts"]) / len(stats["profile_counts"]) if stats["profile_counts"] else 0.0
        ),
        "metadata_coverage": metadata_coverage,
        "metadata_stats": metadata_stats,
        "hit_rate": {f"@{k}": stats["hits"][k] / evaluated if evaluated else 0.0 for k in cutoffs},
        "mean_reciprocal_rank": sum(stats["reciprocal_ranks"]) / evaluated if evaluated else 0.0,
        "examples": stats["examples"],
    }


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
    parser.add_argument("--als-candidate-pool", type=int, default=None)
    parser.add_argument("--content-candidate-pool", type=int, default=0)
    parser.add_argument("--content-min-score", type=float, default=0.0)
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
    als_candidate_pool = args.als_candidate_pool if args.als_candidate_pool is not None else args.candidate_pool
    total_candidate_pool = als_candidate_pool + args.content_candidate_pool

    configs = [EXPERIMENTS[name] for name in args.experiments]
    stats_by_name = {config.name: empty_stats(config.name, cutoffs) for config in configs}
    max_k = max(cutoffs)
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
            N=als_candidate_pool,
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
            stats = stats_by_name[config.name]
            alpha = dynamic_alpha(len(sources)) if config.alpha_policy == "dynamic" else float(config.alpha)
            content_seed_scores = weighted_content_scores(content_seed_components, config.weights)
            content_score_by_item = {
                int(item_idx): float(score)
                for item_idx, score in zip(content_seed_indices, content_seed_scores)
                if score > 0 and score >= args.content_min_score
            }
            if args.content_candidate_pool > 0 and len(content_score_by_item) > 0:
                content_top_items = [
                    item_idx
                    for item_idx, _ in sorted(content_score_by_item.items(), key=lambda pair: pair[1], reverse=True)
                    if item_idx not in als_score_by_item
                ][: args.content_candidate_pool]
            else:
                content_top_items = []

            union_items = list(dict.fromkeys([int(item_idx) for item_idx in item_indices] + content_top_items))
            union_indices = np.asarray(union_items, dtype=np.int32)
            als_union_scores = np.asarray([als_score_by_item.get(item_idx, min_als_score) for item_idx in union_items], dtype=np.float32)
            content_union_scores = np.asarray([content_score_by_item.get(item_idx, 0.0) for item_idx in union_items], dtype=np.float32)
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
            print(f"Grid evaluated {evaluated:,} validation rows", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    runtime = time.time() - start
    for config in configs:
        result = finalize_results(
            stats_by_name[config.name],
            config,
            len(validation),
            evaluated,
            skipped_unknown_user,
            skipped_unknown_book,
            stats_by_name[config.name]["hidden_in_candidate_pool"],
            metadata_coverage,
            metadata_stats,
            total_candidate_pool,
            args.max_profile_books,
            cutoffs,
        )
        result.update(
            {
                "method": "als_content_hybrid_validation_rerank_grid",
                "candidate_pool": als_candidate_pool if not config.weights else total_candidate_pool,
                "candidate_strategy": (
                    "als_only"
                    if not config.weights
                    else "als_plus_content"
                    if args.content_candidate_pool and args.content_min_score <= 0
                    else "als_plus_thresholded_content"
                    if args.content_candidate_pool
                    else "als_rerank_only"
                ),
                "als_candidate_pool": als_candidate_pool,
                "content_candidate_pool": 0 if not config.weights else args.content_candidate_pool,
                "content_min_score": 0.0 if not config.weights else args.content_min_score,
                "content_prepool": args.content_prepool,
                "model_dir": str(args.model_dir),
                "validation": str(args.validation),
                "metadata": str(args.metadata),
                "cutoffs": cutoffs,
            }
        )
        output = args.output_dir / f"{config.name}.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        log_rows.append(
            {
                "experiment": config.name,
                "candidate_pool": total_candidate_pool,
                "als_candidate_pool": als_candidate_pool,
                "content_candidate_pool": args.content_candidate_pool,
                "content_min_score": args.content_min_score,
                "max_profile_books": args.max_profile_books,
                "status": "success",
                "runtime_seconds": f"{runtime:.3f}",
                "output": str(output),
                "log": "grid",
            }
        )

    args.run_log.parent.mkdir(parents=True, exist_ok=True)
    with args.run_log.open("w", newline="", encoding="utf-8") as file:
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

    print(f"Wrote {len(configs)} experiment JSON files to {args.output_dir}", flush=True)
    print(f"Wrote run log: {args.run_log}", flush=True)


if __name__ == "__main__":
    main()
