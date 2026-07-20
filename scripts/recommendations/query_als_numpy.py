#!/usr/bin/env python3
"""Query a NumPy-trained ALS model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def load_model(model_dir: Path):
    user_factors = np.load(model_dir / "user_factors.npy")
    item_factors = np.load(model_dir / "item_factors.npy")
    with (model_dir / "mappings.pkl").open("rb") as file:
        mappings = pickle.load(file)
    user_rows = np.load(model_dir / "user_rows.npz")
    return user_factors, item_factors, mappings, user_rows


def rank_top_items(scores: np.ndarray, blocked: set[int], limit: int) -> list[int]:
    ranked = np.argsort(-scores)
    results = []
    for item_idx in ranked:
        item_idx = int(item_idx)
        if item_idx in blocked:
            continue
        results.append(item_idx)
        if len(results) >= limit:
            break
    return results


def user_blocked_items(user_rows, user_idx: int) -> set[int]:
    start = int(user_rows["indptr"][user_idx])
    end = int(user_rows["indptr"][user_idx + 1])
    return {int(item_idx) for item_idx in user_rows["indices"][start:end]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/recommendations/als_numpy_sample"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--user-id", type=int, help="UCSD compact user ID")
    mode.add_argument("--similar-goodreads-book-id", type=int, help="Goodreads book ID")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    user_factors, item_factors, mappings, user_rows = load_model(args.model_dir)
    idx_to_book = mappings["idx_to_book"]

    if args.user_id is not None:
        user_to_idx = mappings["user_to_idx"]
        if args.user_id not in user_to_idx:
            raise SystemExit(f"user_id={args.user_id} is not in the NumPy ALS model.")

        user_idx = int(user_to_idx[args.user_id])
        scores = item_factors @ user_factors[user_idx]
        blocked = user_blocked_items(user_rows, user_idx)
        recommendations = rank_top_items(scores, blocked, args.limit)
        for item_idx in recommendations:
            print(f"{idx_to_book[item_idx]}\t{float(scores[item_idx]):.6f}")
    else:
        book_to_idx = mappings["book_to_idx"]
        if args.similar_goodreads_book_id not in book_to_idx:
            raise SystemExit(
                f"goodreads_book_id={args.similar_goodreads_book_id} is not in the NumPy ALS model."
            )

        item_idx = int(book_to_idx[args.similar_goodreads_book_id])
        norms = np.linalg.norm(item_factors, axis=1)
        denominator = norms * norms[item_idx]
        similarities = np.divide(
            item_factors @ item_factors[item_idx],
            denominator,
            out=np.zeros_like(norms),
            where=denominator > 0,
        )
        recommendations = rank_top_items(similarities, {item_idx}, args.limit)
        for similar_idx in recommendations:
            print(f"{idx_to_book[similar_idx]}\t{float(similarities[similar_idx]):.6f}")


if __name__ == "__main__":
    main()
