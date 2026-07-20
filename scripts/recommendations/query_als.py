#!/usr/bin/env python3
"""Query a trained implicit ALS model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import scipy.sparse as sparse


def load_artifacts(model_dir: Path):
    with (model_dir / "als_model.pkl").open("rb") as file:
        model = pickle.load(file)
    with (model_dir / "mappings.pkl").open("rb") as file:
        mappings = pickle.load(file)
    user_items = sparse.load_npz(model_dir / "user_items.npz")
    return model, mappings, user_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/recommendations/als"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--user-id", type=int, help="UCSD compact user ID")
    mode.add_argument("--similar-goodreads-book-id", type=int, help="Goodreads book ID")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, mappings, user_items = load_artifacts(args.model_dir)

    if args.user_id is not None:
        user_to_idx = mappings["user_to_idx"]
        idx_to_book = mappings["idx_to_book"]
        if args.user_id not in user_to_idx:
            raise SystemExit(f"user_id={args.user_id} is not in the ALS model.")
        user_idx = user_to_idx[args.user_id]
        item_indices, scores = model.recommend(
            user_idx,
            user_items[user_idx],
            N=args.limit,
            filter_already_liked_items=True,
        )
        for item_idx, score in zip(item_indices, scores):
            print(f"{idx_to_book[int(item_idx)]}\t{float(score):.6f}")
    else:
        book_to_idx = mappings["book_to_idx"]
        idx_to_book = mappings["idx_to_book"]
        if args.similar_goodreads_book_id not in book_to_idx:
            raise SystemExit(f"goodreads_book_id={args.similar_goodreads_book_id} is not in the ALS model.")
        item_idx = book_to_idx[args.similar_goodreads_book_id]
        item_indices, scores = model.similar_items(item_idx, N=args.limit + 1)
        for similar_idx, score in zip(item_indices, scores):
            book_id = idx_to_book[int(similar_idx)]
            if book_id == args.similar_goodreads_book_id:
                continue
            print(f"{book_id}\t{float(score):.6f}")


if __name__ == "__main__":
    main()
