#!/usr/bin/env python3
"""Evaluate ALS/content hybrid reranking against a true validation holdout."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sparse


@dataclass(frozen=True)
class BookMeta:
    goodreads_book_id: int
    author_key: str
    genres: frozenset[str]
    average_rating: float
    ratings_count: int
    page_count: int
    language: str
    popularity_norm: float


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    alpha_policy: str
    alpha: float | None
    weights: dict[str, float]


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "als_baseline": ExperimentConfig("als_baseline", "fixed", 1.0, {}),
    "balanced_metadata": ExperimentConfig(
        "balanced_metadata",
        "fixed",
        0.80,
        {"genre": 0.50, "author": 0.25, "rating": 0.10, "page": 0.05, "popularity": 0.10},
    ),
    "author_50_baseline": ExperimentConfig(
        "author_50_baseline",
        "fixed",
        0.80,
        {"author": 0.50, "genre": 0.30, "rating": 0.10, "page": 0.05, "popularity": 0.05},
    ),
    "author_50_content_heavy": ExperimentConfig(
        "author_50_content_heavy",
        "fixed",
        0.60,
        {"author": 0.50, "genre": 0.30, "rating": 0.10, "page": 0.05, "popularity": 0.05},
    ),
    "author_50_collaborative_heavy": ExperimentConfig(
        "author_50_collaborative_heavy",
        "fixed",
        0.90,
        {"author": 0.50, "genre": 0.30, "rating": 0.10, "page": 0.05, "popularity": 0.05},
    ),
    "genre_heavy": ExperimentConfig(
        "genre_heavy",
        "fixed",
        0.80,
        {"genre": 0.70, "author": 0.15, "rating": 0.05, "page": 0.05, "popularity": 0.05},
    ),
    "author_genre_only": ExperimentConfig(
        "author_genre_only",
        "fixed",
        0.80,
        {"author": 0.50, "genre": 0.50},
    ),
    "rating_popularity_boosted": ExperimentConfig(
        "rating_popularity_boosted",
        "fixed",
        0.80,
        {"genre": 0.35, "author": 0.25, "rating": 0.20, "popularity": 0.15, "page": 0.05},
    ),
    "language_aware": ExperimentConfig(
        "language_aware",
        "fixed",
        0.80,
        {"genre": 0.45, "author": 0.25, "rating": 0.10, "page": 0.05, "popularity": 0.05, "language": 0.10},
    ),
    "dynamic_alpha_balanced": ExperimentConfig(
        "dynamic_alpha_balanced",
        "dynamic",
        None,
        {"genre": 0.50, "author": 0.25, "rating": 0.10, "page": 0.05, "popularity": 0.10},
    ),
    "dynamic_alpha_author_50": ExperimentConfig(
        "dynamic_alpha_author_50",
        "dynamic",
        None,
        {"author": 0.50, "genre": 0.30, "rating": 0.10, "page": 0.05, "popularity": 0.05},
    ),
    "author_60_genre_40_a06": ExperimentConfig(
        "author_60_genre_40_a06",
        "fixed",
        0.60,
        {"author": 0.60, "genre": 0.40},
    ),
    "author_60_genre_40_a07": ExperimentConfig(
        "author_60_genre_40_a07",
        "fixed",
        0.70,
        {"author": 0.60, "genre": 0.40},
    ),
    "author_70_genre_30_a06": ExperimentConfig(
        "author_70_genre_30_a06",
        "fixed",
        0.60,
        {"author": 0.70, "genre": 0.30},
    ),
    "author_70_genre_30_a07": ExperimentConfig(
        "author_70_genre_30_a07",
        "fixed",
        0.70,
        {"author": 0.70, "genre": 0.30},
    ),
    "author_70_genre_30_a08": ExperimentConfig(
        "author_70_genre_30_a08",
        "fixed",
        0.80,
        {"author": 0.70, "genre": 0.30},
    ),
    "author_80_genre_20_a06": ExperimentConfig(
        "author_80_genre_20_a06",
        "fixed",
        0.60,
        {"author": 0.80, "genre": 0.20},
    ),
    "author_80_with_rest_a06": ExperimentConfig(
        "author_80_with_rest_a06",
        "fixed",
        0.60,
        {"author": 0.80, "genre": 0.10, "rating": 0.05, "popularity": 0.05},
    ),
    "author_90_genre_10_a06": ExperimentConfig(
        "author_90_genre_10_a06",
        "fixed",
        0.60,
        {"author": 0.90, "genre": 0.10},
    ),
    "genre_60_author_40_a06": ExperimentConfig(
        "genre_60_author_40_a06",
        "fixed",
        0.60,
        {"genre": 0.60, "author": 0.40},
    ),
    "genre_70_author_30_a06": ExperimentConfig(
        "genre_70_author_30_a06",
        "fixed",
        0.60,
        {"genre": 0.70, "author": 0.30},
    ),
    "genre_80_author_20_a06": ExperimentConfig(
        "genre_80_author_20_a06",
        "fixed",
        0.60,
        {"genre": 0.80, "author": 0.20},
    ),
    "genre_80_with_rest_a06": ExperimentConfig(
        "genre_80_with_rest_a06",
        "fixed",
        0.60,
        {"genre": 0.80, "author": 0.10, "rating": 0.05, "popularity": 0.05},
    ),
    "genre_90_author_10_a06": ExperimentConfig(
        "genre_90_author_10_a06",
        "fixed",
        0.60,
        {"genre": 0.90, "author": 0.10},
    ),
    "pure_author_a06": ExperimentConfig(
        "pure_author_a06",
        "fixed",
        0.60,
        {"author": 1.00},
    ),
    "pure_genre_a06": ExperimentConfig(
        "pure_genre_a06",
        "fixed",
        0.60,
        {"genre": 1.00},
    ),
}


def load_artifacts(model_dir: Path):
    with (model_dir / "als_model.pkl").open("rb") as file:
        model = pickle.load(file)
    with (model_dir / "mappings.pkl").open("rb") as file:
        mappings = pickle.load(file)
    user_items = sparse.load_npz(model_dir / "user_items.npz").tocsr()
    return model, mappings, user_items


def parse_genres(value: Any) -> frozenset[str]:
    if value is None or pd.isna(value):
        return frozenset()
    return frozenset(part.strip().lower() for part in str(value).split(";") if part.strip())


def load_metadata(path: Path) -> tuple[dict[int, BookMeta], dict[str, float]]:
    frame = pd.read_csv(path)
    ratings_counts = frame["ratings_count"].fillna(0).astype(float).to_numpy()
    log_popularity = np.log1p(ratings_counts)
    pop_min = float(log_popularity.min()) if len(log_popularity) else 0.0
    pop_max = float(log_popularity.max()) if len(log_popularity) else 0.0
    pop_range = pop_max - pop_min

    metadata: dict[int, BookMeta] = {}
    for offset, row in enumerate(frame.itertuples(index=False)):
        book_id = int(row.goodreads_book_id)
        popularity_norm = float((log_popularity[offset] - pop_min) / pop_range) if pop_range > 0 else 0.0
        metadata[book_id] = BookMeta(
            goodreads_book_id=book_id,
            author_key="" if pd.isna(row.author_key) else str(row.author_key),
            genres=parse_genres(row.genres),
            average_rating=0.0 if pd.isna(row.average_rating) else float(row.average_rating),
            ratings_count=0 if pd.isna(row.ratings_count) else int(row.ratings_count),
            page_count=0 if pd.isna(row.page_count) else int(row.page_count),
            language="" if pd.isna(row.language) else str(row.language).lower(),
            popularity_norm=popularity_norm,
        )

    coverage = {
        "metadata_rows": float(len(metadata)),
        "genre_coverage": sum(1 for item in metadata.values() if item.genres) / len(metadata) if metadata else 0.0,
        "author_coverage": sum(1 for item in metadata.values() if item.author_key) / len(metadata) if metadata else 0.0,
        "rating_coverage": sum(1 for item in metadata.values() if item.average_rating > 0) / len(metadata) if metadata else 0.0,
        "page_coverage": sum(1 for item in metadata.values() if item.page_count > 0) / len(metadata) if metadata else 0.0,
    }
    return metadata, coverage


def minmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float32)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    value_range = max_value - min_value
    if value_range <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / value_range).astype(np.float32)


def reciprocal_rank(recommendations: list[int], hidden_item_idx: int) -> float:
    for index, item_idx in enumerate(recommendations, start=1):
        if item_idx == hidden_item_idx:
            return 1.0 / index
    return 0.0


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def rating_similarity(left: BookMeta, right: BookMeta) -> float:
    if left.average_rating <= 0 or right.average_rating <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(left.average_rating - right.average_rating) / 4.0)


def page_similarity(left: BookMeta, right: BookMeta) -> float:
    if left.page_count <= 0 or right.page_count <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(left.page_count - right.page_count) / 1000.0)


def language_similarity(left: BookMeta, right: BookMeta) -> float:
    if not left.language or not right.language:
        return 0.0
    return 1.0 if left.language == right.language else 0.0


def author_similarity(left: BookMeta, right: BookMeta) -> float:
    if not left.author_key or not right.author_key:
        return 0.0
    left_authors = set(left.author_key.split("|"))
    right_authors = set(right.author_key.split("|"))
    return 1.0 if left_authors & right_authors else 0.0


def similarity(candidate: BookMeta, source: BookMeta, weights: dict[str, float]) -> float:
    if not weights:
        return 0.0
    score = 0.0
    score += weights.get("genre", 0.0) * jaccard(candidate.genres, source.genres)
    score += weights.get("author", 0.0) * author_similarity(candidate, source)
    score += weights.get("rating", 0.0) * rating_similarity(candidate, source)
    score += weights.get("page", 0.0) * page_similarity(candidate, source)
    score += weights.get("popularity", 0.0) * candidate.popularity_norm
    score += weights.get("language", 0.0) * language_similarity(candidate, source)
    return score


def dynamic_alpha(positive_profile_count: int) -> float:
    return min(0.90, max(0.55, 0.55 + 0.35 * positive_profile_count / 50.0))


def profile_sources(
    row: sparse.csr_matrix,
    idx_to_book: dict[int, int],
    metadata: dict[int, BookMeta],
    max_profile_books: int,
) -> list[tuple[int, float]]:
    start = row.indptr[0]
    end = row.indptr[1]
    indices = row.indices[start:end]
    values = row.data[start:end]
    positives = [(int(item_idx), float(value)) for item_idx, value in zip(indices, values) if value > 0]
    positives.sort(key=lambda pair: pair[1], reverse=True)

    sources: list[tuple[int, float]] = []
    for item_idx, value in positives:
        book_id = int(idx_to_book[item_idx])
        if book_id in metadata:
            sources.append((book_id, value))
        if len(sources) >= max_profile_books:
            break
    return sources


def content_scores_for_candidates(
    candidate_indices: np.ndarray,
    sources: list[tuple[int, float]],
    idx_to_book: dict[int, int],
    metadata: dict[int, BookMeta],
    weights: dict[str, float],
) -> np.ndarray:
    if not sources or not weights:
        return np.zeros(len(candidate_indices), dtype=np.float32)

    source_meta = [(metadata[book_id], max(weight, 0.0)) for book_id, weight in sources if book_id in metadata]
    denominator = sum(weight for _, weight in source_meta)
    if denominator <= 0:
        return np.zeros(len(candidate_indices), dtype=np.float32)

    scores = np.zeros(len(candidate_indices), dtype=np.float32)
    for offset, item_idx in enumerate(candidate_indices):
        candidate_book = int(idx_to_book[int(item_idx)])
        candidate_meta = metadata.get(candidate_book)
        if candidate_meta is None:
            continue
        total = 0.0
        for source, source_weight in source_meta:
            total += source_weight * similarity(candidate_meta, source, weights)
        scores[offset] = total / denominator
    return scores


def evaluate(
    model,
    mappings: dict,
    user_items: sparse.csr_matrix,
    validation: pd.DataFrame,
    metadata: dict[int, BookMeta],
    metadata_stats: dict[str, float],
    config: ExperimentConfig,
    candidate_pool: int,
    max_profile_books: int,
    cutoffs: list[int],
) -> dict:
    max_k = max(cutoffs)
    hits = {k: 0 for k in cutoffs}
    reciprocal_ranks = []
    examples = []
    skipped_unknown_user = 0
    skipped_unknown_book = 0

    user_to_idx = mappings["user_to_idx"]
    book_to_idx = mappings["book_to_idx"]
    idx_to_book = mappings["idx_to_book"]
    metadata_model_coverage = sum(1 for book_id in book_to_idx if int(book_id) in metadata) / len(book_to_idx)

    evaluated = 0
    hidden_in_candidate_pool = 0
    changed_rank_count = 0
    alpha_values: list[float] = []
    profile_counts: list[int] = []

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
            N=candidate_pool,
            filter_already_liked_items=True,
            recalculate_user=True,
        )
        item_indices = np.asarray(item_indices, dtype=np.int32)
        als_scores = np.asarray(als_scores, dtype=np.float32)

        sources = profile_sources(user_items[user_idx], idx_to_book, metadata, max_profile_books)
        alpha = dynamic_alpha(len(sources)) if config.alpha_policy == "dynamic" else float(config.alpha)
        content_scores = content_scores_for_candidates(item_indices, sources, idx_to_book, metadata, config.weights)
        final_scores = alpha * minmax(als_scores) + (1.0 - alpha) * minmax(content_scores)
        order = np.argsort(-final_scores, kind="stable")
        recommendations = [int(item_indices[index]) for index in order[:max_k]]
        als_top = [int(item_idx) for item_idx in item_indices[:max_k]]

        evaluated += 1
        alpha_values.append(alpha)
        profile_counts.append(len(sources))
        if hidden_item_idx in item_indices:
            hidden_in_candidate_pool += 1

        for k in cutoffs:
            if hidden_item_idx in recommendations[:k]:
                hits[k] += 1

        hybrid_rr = reciprocal_rank(recommendations, hidden_item_idx)
        als_rr = reciprocal_rank(als_top, hidden_item_idx)
        reciprocal_ranks.append(hybrid_rr)

        hybrid_rank = int(1 / hybrid_rr) if hybrid_rr else None
        als_rank = int(1 / als_rr) if als_rr else None
        if hybrid_rank != als_rank:
            changed_rank_count += 1
            if len(examples) < 12:
                examples.append(
                    {
                        "user_id": user_id,
                        "hidden_goodreads_book_id": hidden_book,
                        "als_rank": als_rank,
                        "hybrid_rank": hybrid_rank,
                        "alpha": alpha,
                        "positive_profile_count": len(sources),
                        "als_top_recommendations": [int(idx_to_book[item_idx]) for item_idx in als_top[:10]],
                        "hybrid_top_recommendations": [int(idx_to_book[item_idx]) for item_idx in recommendations[:10]],
                    }
                )

        if evaluated % 500 == 0:
            print(f"{config.name}: evaluated {evaluated:,} validation rows", flush=True)

    return {
        "experiment_name": config.name,
        "candidate_pool": candidate_pool,
        "max_profile_books": max_profile_books,
        "alpha_policy": config.alpha_policy,
        "alpha": config.alpha,
        "weights": config.weights,
        "validation_rows": len(validation),
        "users_evaluated": evaluated,
        "skipped_unknown_user": skipped_unknown_user,
        "skipped_unknown_book": skipped_unknown_book,
        "hidden_in_candidate_pool": hidden_in_candidate_pool,
        "candidate_pool_recall": hidden_in_candidate_pool / evaluated if evaluated else 0.0,
        "changed_rank_count": changed_rank_count,
        "changed_rank_rate": changed_rank_count / evaluated if evaluated else 0.0,
        "mean_alpha": sum(alpha_values) / len(alpha_values) if alpha_values else 0.0,
        "mean_positive_profile_count": sum(profile_counts) / len(profile_counts) if profile_counts else 0.0,
        "metadata_coverage": metadata_model_coverage,
        "metadata_stats": metadata_stats,
        "hit_rate": {f"@{k}": hits[k] / evaluated if evaluated else 0.0 for k in cutoffs},
        "mean_reciprocal_rank": sum(reciprocal_ranks) / evaluated if evaluated else 0.0,
        "examples": examples,
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
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--candidate-pool", type=int, default=500)
    parser.add_argument("--max-profile-books", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 20, 50])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, mappings, user_items = load_artifacts(args.model_dir)
    validation = pd.read_csv(args.validation)
    metadata, metadata_stats = load_metadata(args.metadata)
    config = EXPERIMENTS[args.experiment]
    results = evaluate(
        model,
        mappings,
        user_items,
        validation,
        metadata,
        metadata_stats,
        config,
        args.candidate_pool,
        args.max_profile_books,
        sorted(args.cutoffs),
    )
    results.update(
        {
            "method": "als_content_hybrid_validation_rerank",
            "model_dir": str(args.model_dir),
            "validation": str(args.validation),
            "metadata": str(args.metadata),
            "cutoffs": sorted(args.cutoffs),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"Wrote evaluation: {args.output}", flush=True)


if __name__ == "__main__":
    main()
