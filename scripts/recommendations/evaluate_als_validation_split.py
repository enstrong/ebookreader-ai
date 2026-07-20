#!/usr/bin/env python3
"""Evaluate an ALS model against a true pre-training validation holdout file."""

from __future__ import annotations

import argparse
import json
import pickle
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
    return {int(book_to_idx[book_id]) for book_id in book_ids if book_id in book_to_idx}


def reciprocal_rank(recommendations: list[int], hidden_item_idx: int) -> float:
    for index, item_idx in enumerate(recommendations, start=1):
        if item_idx == hidden_item_idx:
            return 1.0 / index
    return 0.0


def evaluate(
    model,
    mappings: dict,
    user_items: sparse.csr_matrix,
    validation: pd.DataFrame,
    recommendable_item_indices: set[int] | None,
    cutoffs: list[int],
) -> dict:
    max_k = max(cutoffs)
    hits = {k: 0 for k in cutoffs}
    reciprocal_ranks = []
    examples = []
    skipped_unknown_user = 0
    skipped_unknown_book = 0
    skipped_not_recommendable = 0

    user_to_idx = mappings["user_to_idx"]
    book_to_idx = mappings["book_to_idx"]
    idx_to_book = mappings["idx_to_book"]
    recommendable_items = (
        np.array(sorted(recommendable_item_indices), dtype=np.int32)
        if recommendable_item_indices is not None
        else None
    )

    evaluated = 0
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
        if recommendable_item_indices is not None and hidden_item_idx not in recommendable_item_indices:
            skipped_not_recommendable += 1
            continue

        user_idx = int(user_to_idx[user_id])
        item_indices, _ = model.recommend(
            0,
            user_items[user_idx],
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
                    "hidden_goodreads_book_id": hidden_book,
                    "hit_rank": int(1 / rr) if rr else None,
                    "top_recommendations": [int(idx_to_book[item_idx]) for item_idx in recommendations[:10]],
                }
            )

        if evaluated % 500 == 0:
            print(f"Evaluated {evaluated:,} validation rows", flush=True)

    return {
        "validation_rows": len(validation),
        "users_evaluated": evaluated,
        "skipped_unknown_user": skipped_unknown_user,
        "skipped_unknown_book": skipped_unknown_book,
        "skipped_not_recommendable": skipped_not_recommendable,
        "hit_rate": {f"@{k}": hits[k] / evaluated if evaluated else 0.0 for k in cutoffs},
        "mean_reciprocal_rank": sum(reciprocal_ranks) / evaluated if evaluated else 0.0,
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--allowed-similarities", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 20, 50])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cutoffs = sorted(args.cutoffs)
    model, mappings, user_items = load_artifacts(args.model_dir)
    validation = pd.read_csv(args.validation)
    hideable_books, recommendable_books = load_similarity_universe(args.allowed_similarities)

    model_books = set(mappings["book_to_idx"])
    if hideable_books is not None:
        validation = validation[validation["goodreads_book_id"].isin(hideable_books.intersection(model_books))]
    recommendable_item_indices = book_ids_to_item_indices(recommendable_books, mappings["book_to_idx"])

    results = evaluate(model, mappings, user_items, validation, recommendable_item_indices, cutoffs)
    results.update(
        {
            "method": "implicit_als_pre_training_validation_holdout",
            "model_dir": str(args.model_dir),
            "validation": str(args.validation),
            "allowed_similarities": str(args.allowed_similarities) if args.allowed_similarities else None,
            "cutoffs": cutoffs,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"Wrote evaluation: {args.output}", flush=True)


if __name__ == "__main__":
    main()
