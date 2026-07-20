#!/usr/bin/env python3
"""Evaluate a NumPy ALS model with a 5-star interaction holdout."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_model(model_dir: Path):
    user_factors = np.load(model_dir / "user_factors.npy")
    item_factors = np.load(model_dir / "item_factors.npy")
    with (model_dir / "mappings.pkl").open("rb") as file:
        mappings = pickle.load(file)
    user_rows = np.load(model_dir / "user_rows.npz")
    return user_factors, item_factors, mappings, user_rows


def load_similarity_universe(path: Path | None) -> tuple[set[int] | None, set[int] | None]:
    if path is None:
        return None, None

    frame = pd.read_csv(path, usecols=["goodreads_book_id", "similar_goodreads_book_id"])
    hideable_books = set(frame["goodreads_book_id"].astype(int).tolist())
    recommendable_books = hideable_books.union(frame["similar_goodreads_book_id"].astype(int).tolist())
    return hideable_books, recommendable_books


def load_user_likes(
    interactions_path: Path,
    candidate_books: set[int],
    model_users: set[int],
    min_like_rating: int,
    chunk_size: int,
    max_users: int,
    min_likes: int,
) -> dict[int, set[int]]:
    user_likes: dict[int, set[int]] = defaultdict(set)
    dtypes = {"user_id": "int64", "goodreads_book_id": "int64", "rating": "int8"}
    columns = ["user_id", "goodreads_book_id", "rating"]

    for chunk_number, chunk in enumerate(
        pd.read_csv(interactions_path, usecols=columns, dtype=dtypes, chunksize=chunk_size),
        start=1,
    ):
        liked = chunk[
            (chunk["rating"] >= min_like_rating)
            & (chunk["goodreads_book_id"].isin(candidate_books))
            & (chunk["user_id"].isin(model_users))
        ]
        for row in liked.itertuples(index=False):
            user_likes[int(row.user_id)].add(int(row.goodreads_book_id))

        eligible_count = sum(1 for likes in user_likes.values() if len(likes) >= min_likes)
        print(
            f"Loaded chunk {chunk_number}: users={len(user_likes):,}, eligible={eligible_count:,}",
            flush=True,
        )
        if eligible_count >= max_users:
            break

    eligible = {
        user_id: likes
        for user_id, likes in user_likes.items()
        if len(likes) >= min_likes
    }
    return dict(list(eligible.items())[:max_users])


def user_row_without_item(user_rows, user_idx: int, hidden_item_idx: int):
    start = int(user_rows["indptr"][user_idx])
    end = int(user_rows["indptr"][user_idx + 1])
    indices = user_rows["indices"][start:end]
    confidences = user_rows["confidences"][start:end]
    preferences = user_rows["preferences"][start:end]
    keep = indices != hidden_item_idx
    return indices[keep], confidences[keep], preferences[keep]


def recalculate_user_vector(
    item_factors: np.ndarray,
    item_indices: np.ndarray,
    confidences: np.ndarray,
    preferences: np.ndarray,
    regularization: float,
) -> np.ndarray:
    factors = item_factors.shape[1]
    gram = item_factors.T @ item_factors
    identity = np.eye(factors, dtype=np.float32)
    related_factors = item_factors[item_indices]
    weighted_factors = related_factors.T * (confidences - 1.0)
    a_matrix = gram + weighted_factors @ related_factors + regularization * identity
    b_vector = related_factors.T @ (confidences * preferences)
    return np.linalg.solve(a_matrix, b_vector).astype(np.float32)


def rank_top_items(scores: np.ndarray, blocked: set[int], allowed_items: np.ndarray | None, limit: int) -> list[int]:
    if allowed_items is None:
        ranked = np.argsort(-scores)
    else:
        ranked_positions = np.argsort(-scores[allowed_items])
        ranked = allowed_items[ranked_positions]

    results = []
    for item_idx in ranked:
        item_idx = int(item_idx)
        if item_idx in blocked:
            continue
        results.append(item_idx)
        if len(results) >= limit:
            break
    return results


def reciprocal_rank(recommendations: list[int], hidden_item_idx: int) -> float:
    for index, item_idx in enumerate(recommendations, start=1):
        if item_idx == hidden_item_idx:
            return 1.0 / index
    return 0.0


def evaluate(
    item_factors: np.ndarray,
    mappings: dict,
    user_rows,
    user_likes: dict[int, set[int]],
    recommendable_book_ids: set[int] | None,
    regularization: float,
    cutoffs: list[int],
    rng: random.Random,
) -> dict:
    user_to_idx = mappings["user_to_idx"]
    book_to_idx = mappings["book_to_idx"]
    idx_to_book = mappings["idx_to_book"]
    allowed_items = None
    if recommendable_book_ids is not None:
        allowed_items = np.array(
            sorted(book_to_idx[book_id] for book_id in recommendable_book_ids if book_id in book_to_idx),
            dtype=np.int32,
        )

    max_k = max(cutoffs)
    hits = {k: 0 for k in cutoffs}
    reciprocal_ranks = []
    examples = []

    for evaluated, (user_id, likes) in enumerate(user_likes.items(), start=1):
        hidden_book = rng.choice(sorted(likes))
        hidden_item_idx = int(book_to_idx[hidden_book])
        user_idx = int(user_to_idx[user_id])
        item_indices, confidences, preferences = user_row_without_item(user_rows, user_idx, hidden_item_idx)
        user_vector = recalculate_user_vector(
            item_factors,
            item_indices,
            confidences,
            preferences,
            regularization,
        )

        scores = item_factors @ user_vector
        blocked = {int(item_idx) for item_idx in item_indices}
        recommendations = rank_top_items(scores, blocked, allowed_items, max_k)

        for k in cutoffs:
            if hidden_item_idx in recommendations[:k]:
                hits[k] += 1

        rr = reciprocal_rank(recommendations, hidden_item_idx)
        reciprocal_ranks.append(rr)

        if len(examples) < 10:
            examples.append(
                {
                    "user_id": user_id,
                    "hidden_goodreads_book_id": hidden_book,
                    "source_likes_count": len(likes) - 1,
                    "hit_rank": int(1 / rr) if rr else None,
                    "top_recommendations": [
                        int(idx_to_book[item_idx]) for item_idx in recommendations[:10]
                    ],
                }
            )

        if evaluated % 500 == 0:
            print(f"Evaluated {evaluated:,}/{len(user_likes):,} users", flush=True)

    evaluated_count = len(user_likes)
    return {
        "users_evaluated": evaluated_count,
        "hit_rate": {
            f"@{k}": hits[k] / evaluated_count if evaluated_count else 0.0
            for k in cutoffs
        },
        "mean_reciprocal_rank": sum(reciprocal_ranks) / evaluated_count if evaluated_count else 0.0,
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/recommendations/als_numpy_sample"))
    parser.add_argument("--interactions", type=Path, default=Path("data/recommendations/interactions_filtered_sample.csv"))
    parser.add_argument("--allowed-similarities", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/recommendations/als_numpy_sample/evaluation.json"))
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--max-users", type=int, default=1_000)
    parser.add_argument("--min-likes", type=int, default=5)
    parser.add_argument("--min-like-rating", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cutoffs = sorted(args.cutoffs)
    _, item_factors, mappings, user_rows = load_model(args.model_dir)
    hideable_books, recommendable_books = load_similarity_universe(args.allowed_similarities)
    model_books = set(mappings["book_to_idx"])
    if hideable_books is None:
        hideable_books = model_books
    else:
        hideable_books = hideable_books.intersection(model_books)

    print(f"NumPy ALS model dir: {args.model_dir}", flush=True)
    print(f"Holdout candidate books: {len(hideable_books):,}", flush=True)
    user_likes = load_user_likes(
        args.interactions,
        hideable_books,
        set(mappings["user_to_idx"]),
        args.min_like_rating,
        args.chunk_size,
        args.max_users,
        args.min_likes,
    )
    print(f"Eligible users selected: {len(user_likes):,}", flush=True)

    results = evaluate(
        item_factors,
        mappings,
        user_rows,
        user_likes,
        recommendable_books,
        args.regularization,
        cutoffs,
        random.Random(args.seed),
    )
    results.update(
        {
            "method": "numpy_als_interaction_holdout_recalculate_user",
            "model_dir": str(args.model_dir),
            "interactions": str(args.interactions),
            "allowed_similarities": str(args.allowed_similarities) if args.allowed_similarities else None,
            "max_users": args.max_users,
            "min_likes": args.min_likes,
            "min_like_rating": args.min_like_rating,
            "regularization": args.regularization,
            "cutoffs": cutoffs,
            "seed": args.seed,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"Wrote evaluation: {args.output}", flush=True)


if __name__ == "__main__":
    main()
