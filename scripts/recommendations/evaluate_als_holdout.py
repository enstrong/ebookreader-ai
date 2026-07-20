#!/usr/bin/env python3
"""Evaluate ALS with the same interaction holdout style as Level 2 item-item CF."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse


def load_artifacts(model_dir: Path):
    with (model_dir / "als_model.pkl").open("rb") as file:
        model = pickle.load(file)
    with (model_dir / "mappings.pkl").open("rb") as file:
        mappings = pickle.load(file)
    user_items = sparse.load_npz(model_dir / "user_items.npz")
    return model, mappings, user_items.tocsr()


def load_similarity_universe(path: Path | None) -> tuple[set[int] | None, set[int] | None]:
    if path is None:
        return None, None

    frame = pd.read_csv(path, usecols=["goodreads_book_id", "similar_goodreads_book_id"])
    hideable_books = set(frame["goodreads_book_id"].astype(int).tolist())
    recommendable_books = hideable_books.union(frame["similar_goodreads_book_id"].astype(int).tolist())
    return hideable_books, recommendable_books


def book_ids_to_item_indices(book_ids: set[int] | None, book_to_idx: dict[int, int]) -> set[int] | None:
    if book_ids is None:
        return None
    return {
        int(book_to_idx[book_id])
        for book_id in book_ids
        if book_id in book_to_idx
    }


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


def row_without_item(row: sparse.csr_matrix, hidden_item_idx: int, width: int) -> sparse.csr_matrix:
    mask = row.indices != hidden_item_idx
    return sparse.csr_matrix(
        (row.data[mask], row.indices[mask], [0, int(mask.sum())]),
        shape=(1, width),
        dtype=row.dtype,
    )


def reciprocal_rank(recommendations: list[int], hidden_item_idx: int) -> float:
    for index, item_idx in enumerate(recommendations, start=1):
        if item_idx == hidden_item_idx:
            return 1.0 / index
    return 0.0


def evaluate(
    model,
    mappings: dict,
    user_items: sparse.csr_matrix,
    user_likes: dict[int, set[int]],
    recommendable_item_indices: set[int] | None,
    cutoffs: list[int],
    rng: random.Random,
) -> dict:
    max_k = max(cutoffs)
    hits = {k: 0 for k in cutoffs}
    reciprocal_ranks = []
    examples = []

    user_to_idx = mappings["user_to_idx"]
    book_to_idx = mappings["book_to_idx"]
    idx_to_book = mappings["idx_to_book"]
    recommendable_items = (
        np.array(sorted(recommendable_item_indices), dtype=np.int32)
        if recommendable_item_indices is not None
        else None
    )

    evaluated = 0
    skipped = 0
    for user_id, likes in user_likes.items():
        hideable_books = [book_id for book_id in sorted(likes) if book_id in book_to_idx]
        if not hideable_books:
            skipped += 1
            continue

        hidden_book = rng.choice(hideable_books)
        hidden_item_idx = int(book_to_idx[hidden_book])
        user_idx = int(user_to_idx[user_id])
        row = user_items[user_idx]
        temporary_user_items = row_without_item(row, hidden_item_idx, user_items.shape[1])

        item_indices, _ = model.recommend(
            0,
            temporary_user_items,
            N=max_k,
            filter_already_liked_items=True,
            recalculate_user=True,
            items=recommendable_items,
        )
        recommendations = [int(item_idx) for item_idx in item_indices]
        evaluated += 1

        for k in cutoffs:
            if hidden_item_idx in recommendations[:k]:
                hits[k] += 1

        rr = reciprocal_rank(recommendations, hidden_item_idx)
        reciprocal_ranks.append(rr)

        if len(examples) < 10:
            examples.append(
                {
                    "user_id": user_id,
                    "hidden_goodreads_book_id": int(hidden_book),
                    "source_likes_count": len(likes) - 1,
                    "hit_rank": int(1 / rr) if rr else None,
                    "top_recommendations": [
                        int(idx_to_book[item_idx]) for item_idx in recommendations[:10]
                    ],
                }
            )

        if evaluated % 500 == 0:
            print(f"Evaluated {evaluated:,}/{len(user_likes):,} users", flush=True)

    return {
        "users_evaluated": evaluated,
        "users_skipped": skipped,
        "hit_rate": {
            f"@{k}": hits[k] / evaluated if evaluated else 0.0
            for k in cutoffs
        },
        "mean_reciprocal_rank": sum(reciprocal_ranks) / evaluated if evaluated else 0.0,
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/recommendations/als"))
    parser.add_argument("--interactions", type=Path, default=Path("data/recommendations/interactions_filtered.csv"))
    parser.add_argument("--allowed-similarities", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/recommendations/als/evaluation_holdout.json"))
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--max-users", type=int, default=10_000)
    parser.add_argument("--min-likes", type=int, default=5)
    parser.add_argument("--min-like-rating", type=int, default=5)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cutoffs = sorted(args.cutoffs)
    model, mappings, user_items = load_artifacts(args.model_dir)
    hideable_books, recommendable_books = load_similarity_universe(args.allowed_similarities)

    model_books = set(mappings["book_to_idx"])
    if hideable_books is None:
        hideable_books = model_books
    else:
        hideable_books = hideable_books.intersection(model_books)
    recommendable_item_indices = book_ids_to_item_indices(recommendable_books, mappings["book_to_idx"])

    print(f"ALS model dir: {args.model_dir}", flush=True)
    print(f"User-item matrix: shape={user_items.shape}, nonzeros={user_items.nnz:,}", flush=True)
    print(f"Holdout candidate books: {len(hideable_books):,}", flush=True)
    print("Loading eligible user likes", flush=True)
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
        model,
        mappings,
        user_items,
        user_likes,
        recommendable_item_indices,
        cutoffs,
        random.Random(args.seed),
    )
    results.update(
        {
            "method": "implicit_als_interaction_holdout_recalculate_user",
            "model_dir": str(args.model_dir),
            "interactions": str(args.interactions),
            "allowed_similarities": str(args.allowed_similarities) if args.allowed_similarities else None,
            "max_users": args.max_users,
            "min_likes": args.min_likes,
            "min_like_rating": args.min_like_rating,
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
