#!/usr/bin/env python3
"""Evaluate focused content-candidate strategies for ALS/content hybrid experiments."""

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
    weighted_content_scores,
)
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


BALANCED_SOURCE_WEIGHTS = {
    "genre": 0.50,
    "author": 0.25,
    "rating": 0.10,
    "page": 0.05,
    "popularity": 0.10,
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
    parser.add_argument("--output-root", type=Path, default=Path("data/recommendations/hybrid/candidate_strategy_experiments"))
    parser.add_argument("--candidate-pool", type=int, default=1000)
    parser.add_argument("--author-pool", type=int, default=100)
    parser.add_argument("--genre-pool", type=int, default=100)
    parser.add_argument("--genre-threshold", type=float, default=0.35)
    parser.add_argument("--source-top-books", type=int, default=10)
    parser.add_argument("--source-neighbors-per-book", type=int, default=50)
    parser.add_argument("--max-profile-books", type=int, default=50)
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS), default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 20, 50])
    return parser.parse_args()


def source_similarity(candidate, source) -> float:
    return (
        BALANCED_SOURCE_WEIGHTS["genre"] * jaccard(candidate.genres, source.genres)
        + BALANCED_SOURCE_WEIGHTS["author"] * author_similarity(candidate, source)
        + BALANCED_SOURCE_WEIGHTS["rating"] * rating_similarity(candidate, source)
        + BALANCED_SOURCE_WEIGHTS["page"] * page_similarity(candidate, source)
        + BALANCED_SOURCE_WEIGHTS["popularity"] * candidate.popularity_norm
    )


def rank_author_exact_candidates(sources, metadata, idx_to_book, author_index, blocked, limit) -> np.ndarray:
    scores: dict[int, float] = {}
    for book_id, source_weight in sources:
        source = metadata.get(book_id)
        if source is None:
            continue
        for author in source.author_key.split("|"):
            if not author:
                continue
            for item_idx in author_index.get(author, []):
                if item_idx in blocked:
                    continue
                candidate_meta = metadata.get(int(idx_to_book[int(item_idx)]))
                if candidate_meta is None:
                    continue
                scores[item_idx] = scores.get(item_idx, 0.0) + float(source_weight) + 0.05 * candidate_meta.popularity_norm
    ranked = [item_idx for item_idx, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit]]
    return np.asarray(ranked, dtype=np.int32)


def rank_genre_high_confidence_candidates(
    sources,
    metadata,
    idx_to_book,
    genre_index,
    blocked,
    limit,
    threshold,
) -> np.ndarray:
    source_meta = [(metadata[book_id], max(float(weight), 0.0)) for book_id, weight in sources if book_id in metadata]
    denominator = sum(weight for _, weight in source_meta)
    if denominator <= 0:
        return np.asarray([], dtype=np.int32)

    candidates: set[int] = set()
    for source, _ in source_meta:
        for genre in source.genres:
            candidates.update(genre_index.get(genre, []))

    scores: list[tuple[int, float]] = []
    for item_idx in candidates:
        if item_idx in blocked:
            continue
        candidate_meta = metadata.get(int(idx_to_book[int(item_idx)]))
        if candidate_meta is None:
            continue
        score = sum(weight * jaccard(candidate_meta.genres, source.genres) for source, weight in source_meta) / denominator
        if score >= threshold:
            scores.append((int(item_idx), score + 0.02 * candidate_meta.popularity_norm))

    ranked = [item_idx for item_idx, _ in sorted(scores, key=lambda pair: pair[1], reverse=True)[:limit]]
    return np.asarray(ranked, dtype=np.int32)


def rank_source_neighbor_candidates(
    sources,
    metadata,
    idx_to_book,
    author_index,
    genre_index,
    blocked,
    top_sources,
    neighbors_per_source,
) -> np.ndarray:
    candidate_scores: dict[int, float] = {}
    for book_id, source_weight in sources[:top_sources]:
        source = metadata.get(book_id)
        if source is None:
            continue

        candidate_set: set[int] = set()
        for author in source.author_key.split("|"):
            if author:
                candidate_set.update(author_index.get(author, []))
        for genre in source.genres:
            candidate_set.update(genre_index.get(genre, []))

        per_source_scores: list[tuple[int, float]] = []
        for item_idx in candidate_set:
            if item_idx in blocked:
                continue
            candidate_meta = metadata.get(int(idx_to_book[int(item_idx)]))
            if candidate_meta is None:
                continue
            score = source_similarity(candidate_meta, source)
            if score > 0:
                per_source_scores.append((int(item_idx), float(source_weight) * score))

        for item_idx, score in sorted(per_source_scores, key=lambda pair: pair[1], reverse=True)[:neighbors_per_source]:
            candidate_scores[item_idx] = max(candidate_scores.get(item_idx, 0.0), score)

    ranked = [item_idx for item_idx, _ in sorted(candidate_scores.items(), key=lambda pair: pair[1], reverse=True)]
    return np.asarray(ranked, dtype=np.int32)


def run_strategy_recommendation(
    config,
    strategy_indices,
    strategy_components,
    alpha,
    item_indices,
    als_score_by_item,
    min_als_score,
    max_k,
) -> tuple[list[int], list[int], dict[int, float]]:
    content_scores = weighted_content_scores(strategy_components, config.weights)
    content_score_by_item = {
        int(item_idx): float(score)
        for item_idx, score in zip(strategy_indices, content_scores)
        if score > 0
    }
    content_top_items = [int(item_idx) for item_idx in strategy_indices if int(item_idx) not in als_score_by_item]
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
    return recommendations, union_items, content_score_by_item


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
    author_index, genre_index, _popular_items = build_metadata_indexes(metadata, book_to_idx)
    configs = [EXPERIMENTS[name] for name in args.experiments]
    strategy_names = ["author_exact_top100", "genre_high_conf_top100", "source_neighbors_top50x10"]
    stats_by_strategy = {
        strategy: {config.name: empty_stats(config.name, cutoffs) for config in configs}
        for strategy in strategy_names
    }

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
        strategy_indices = {
            "author_exact_top100": rank_author_exact_candidates(
                sources,
                metadata,
                idx_to_book,
                author_index,
                blocked,
                args.author_pool,
            ),
            "genre_high_conf_top100": rank_genre_high_confidence_candidates(
                sources,
                metadata,
                idx_to_book,
                genre_index,
                blocked,
                args.genre_pool,
                args.genre_threshold,
            ),
            "source_neighbors_top50x10": rank_source_neighbor_candidates(
                sources,
                metadata,
                idx_to_book,
                author_index,
                genre_index,
                blocked,
                args.source_top_books,
                args.source_neighbors_per_book,
            ),
        }
        strategy_components = {
            strategy: component_scores(indices, sources, idx_to_book, metadata)
            for strategy, indices in strategy_indices.items()
        }
        evaluated += 1

        for strategy in strategy_names:
            for config in configs:
                stats = stats_by_strategy[strategy][config.name]
                alpha = dynamic_alpha(len(sources)) if config.alpha_policy == "dynamic" else float(config.alpha)
                recommendations, union_items, _content_score_by_item = run_strategy_recommendation(
                    config,
                    strategy_indices[strategy],
                    strategy_components[strategy],
                    alpha,
                    item_indices,
                    als_score_by_item,
                    min_als_score,
                    max_k,
                )
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
            print(f"Candidate strategy grid evaluated {evaluated:,} validation rows", flush=True)

    runtime = time.time() - start
    args.output_root.mkdir(parents=True, exist_ok=True)
    for strategy in strategy_names:
        output_dir = args.output_root / strategy
        output_dir.mkdir(parents=True, exist_ok=True)
        content_pool = (
            args.author_pool
            if strategy == "author_exact_top100"
            else args.genre_pool
            if strategy == "genre_high_conf_top100"
            else args.source_top_books * args.source_neighbors_per_book
        )
        candidate_pool = args.candidate_pool + content_pool
        log_rows = []

        for config in configs:
            result = finalize_results(
                stats_by_strategy[strategy][config.name],
                config,
                len(validation),
                evaluated,
                skipped_unknown_user,
                skipped_unknown_book,
                stats_by_strategy[strategy][config.name]["hidden_in_candidate_pool"],
                metadata_coverage,
                metadata_stats,
                candidate_pool,
                args.max_profile_books,
                cutoffs,
            )
            result.update(
                {
                    "method": "als_content_hybrid_candidate_strategy_grid",
                    "candidate_strategy": "als_only" if not config.weights else strategy,
                    "candidate_pool": args.candidate_pool if not config.weights else candidate_pool,
                    "als_candidate_pool": args.candidate_pool,
                    "content_candidate_pool": 0 if not config.weights else content_pool,
                    "author_pool": args.author_pool,
                    "genre_pool": args.genre_pool,
                    "genre_threshold": args.genre_threshold,
                    "source_top_books": args.source_top_books,
                    "source_neighbors_per_book": args.source_neighbors_per_book,
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
                    "strategy": strategy,
                    "candidate_pool": candidate_pool,
                    "als_candidate_pool": args.candidate_pool,
                    "content_candidate_pool": content_pool,
                    "status": "success",
                    "runtime_seconds": f"{runtime:.3f}",
                    "output": str(output),
                    "log": "candidate_strategy_grid",
                }
            )

        run_log = args.output_root / f"{strategy}_run_log.csv"
        with run_log.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "experiment",
                    "strategy",
                    "candidate_pool",
                    "als_candidate_pool",
                    "content_candidate_pool",
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
