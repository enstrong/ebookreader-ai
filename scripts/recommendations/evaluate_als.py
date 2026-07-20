#!/usr/bin/env python3
"""Evaluate an implicit ALS model with a leave-one-out ranking test.

For each eligible user:
1. choose one liked book to hide
2. rebuild that user's vector from the remaining liked books
3. recommend books from the trained item factors
4. check whether the hidden book appears in the top K results
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
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


def load_allowed_books(similarities_path: Path | None) -> tuple[set[int] | None, set[int] | None]:
    if similarities_path is None:
        return None, None

    frame = pd.read_csv(similarities_path, usecols=["goodreads_book_id", "similar_goodreads_book_id"])
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


def eligible_user_indices(
    user_items: sparse.csr_matrix,
    min_likes: int,
    max_users: int,
    hideable_item_indices: set[int] | None,
) -> list[int]:
    eligible = []
    for index in range(user_items.shape[0]):
        start = user_items.indptr[index]
        end = user_items.indptr[index + 1]
        row_indices = user_items.indices[start:end]
        row_data = user_items.data[start:end]
        positive_count = 0
        for item_idx, value in zip(row_indices, row_data):
            if value <= 0:
                continue
            if hideable_item_indices is not None and int(item_idx) not in hideable_item_indices:
                continue
            positive_count += 1
        if positive_count >= min_likes:
            eligible.append(index)
            if len(eligible) >= max_users:
                break
    return eligible


def positive_item_indices(
    row: sparse.csr_matrix,
    hideable_item_indices: set[int] | None,
) -> list[int]:
    return [
        int(item_idx)
        for item_idx, value in zip(row.indices, row.data)
        if value > 0 and (hideable_item_indices is None or int(item_idx) in hideable_item_indices)
    ]


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
    user_indices: list[int],
    cutoffs: list[int],
    rng: random.Random,
    hideable_item_indices: set[int] | None,
    recommendable_item_indices: set[int] | None,
) -> dict:
    max_k = max(cutoffs)
    hits = {k: 0 for k in cutoffs}
    reciprocal_ranks = []
    examples = []

    idx_to_book = mappings["idx_to_book"]
    idx_to_user = {index: user_id for user_id, index in mappings["user_to_idx"].items()}
    recommendable_items = (
        np.array(sorted(recommendable_item_indices), dtype=np.int32)
        if recommendable_item_indices is not None
        else None
    )

    for evaluated, user_idx in enumerate(user_indices, start=1):
        row = user_items[user_idx]
        positive_items = positive_item_indices(row, hideable_item_indices)
        hidden_item_idx = int(rng.choice(positive_items))
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

        for k in cutoffs:
            if hidden_item_idx in recommendations[:k]:
                hits[k] += 1

        rr = reciprocal_rank(recommendations, hidden_item_idx)
        reciprocal_ranks.append(rr)

        if len(examples) < 10:
            examples.append(
                {
                    "user_id": int(idx_to_user[user_idx]),
                    "hidden_goodreads_book_id": int(idx_to_book[hidden_item_idx]),
                    "source_books_count": int(row.nnz - 1),
                    "positive_source_books_count": int(len(positive_items) - 1),
                    "hit_rank": int(1 / rr) if rr else None,
                    "top_recommendations": [
                        int(idx_to_book[item_idx]) for item_idx in recommendations[:10]
                    ],
                }
            )

        if evaluated % 500 == 0:
            print(f"Evaluated {evaluated:,}/{len(user_indices):,} users", flush=True)

    evaluated_count = len(user_indices)
    return {
        "users_evaluated": evaluated_count,
        "hit_rate": {
            f"@{k}": hits[k] / evaluated_count if evaluated_count else 0.0
            for k in cutoffs
        },
        "mean_reciprocal_rank": (
            sum(reciprocal_ranks) / evaluated_count if evaluated_count else 0.0
        ),
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/recommendations/als"))
    parser.add_argument("--output", type=Path, default=Path("data/recommendations/als/evaluation.json"))
    parser.add_argument("--max-users", type=int, default=10_000)
    parser.add_argument("--min-likes", type=int, default=5)
    parser.add_argument(
        "--allowed-similarities",
        type=Path,
        default=None,
        help="Restrict hidden/recommended books to the candidate universe from a Level 2 similarity CSV.",
    )
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cutoffs = sorted(args.cutoffs)
    model, mappings, user_items = load_artifacts(args.model_dir)
    hideable_books, recommendable_books = load_allowed_books(args.allowed_similarities)
    hideable_item_indices = book_ids_to_item_indices(hideable_books, mappings["book_to_idx"])
    recommendable_item_indices = book_ids_to_item_indices(recommendable_books, mappings["book_to_idx"])
    user_indices = eligible_user_indices(
        user_items,
        args.min_likes,
        args.max_users,
        hideable_item_indices,
    )

    print(f"ALS model dir: {args.model_dir}", flush=True)
    print(f"User-item matrix: shape={user_items.shape}, nonzeros={user_items.nnz:,}", flush=True)
    if args.allowed_similarities is not None:
        print(
            f"Allowed similarity universe: hideable={len(hideable_item_indices or [])}, "
            f"recommendable={len(recommendable_item_indices or [])}",
            flush=True,
        )
    print(f"Eligible users selected: {len(user_indices):,}", flush=True)

    results = evaluate(
        model,
        mappings,
        user_items,
        user_indices,
        cutoffs,
        random.Random(args.seed),
        hideable_item_indices,
        recommendable_item_indices,
    )
    results.update(
        {
            "method": "implicit_als_leave_one_out_recalculate_user",
            "model_dir": str(args.model_dir),
            "max_users": args.max_users,
            "min_likes": args.min_likes,
            "allowed_similarities": str(args.allowed_similarities) if args.allowed_similarities else None,
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
