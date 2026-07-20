#!/usr/bin/env python3
"""Recommend books with the Level 2 item-item collaborative filtering model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_user_history(
    interactions_path: Path,
    user_id: int,
    min_like_rating: int,
    chunk_size: int,
) -> tuple[set[int], dict[int, float]]:
    read_books: set[int] = set()
    liked_books: dict[int, float] = {}
    columns = ["user_id", "goodreads_book_id", "rating"]
    dtypes = {"user_id": "int64", "goodreads_book_id": "int64", "rating": "int8"}

    for chunk in pd.read_csv(interactions_path, usecols=columns, dtype=dtypes, chunksize=chunk_size):
        rows = chunk[chunk["user_id"] == user_id]
        if rows.empty:
            continue
        for row in rows.itertuples(index=False):
            book_id = int(row.goodreads_book_id)
            rating = float(row.rating)
            read_books.add(book_id)
            if rating >= min_like_rating:
                liked_books[book_id] = rating

    return read_books, liked_books


def parse_liked_books(raw_values: list[str]) -> dict[int, float]:
    liked: dict[int, float] = {}
    for raw in raw_values:
        if ":" in raw:
            book_id, rating = raw.split(":", 1)
            liked[int(book_id)] = float(rating)
        else:
            liked[int(raw)] = 5.0
    return liked


def recommend(
    similarities_path: Path,
    liked_books: dict[int, float],
    read_books: set[int],
    limit: int,
) -> pd.DataFrame:
    similarities = pd.read_csv(similarities_path)
    liked_ids = set(liked_books)
    rows = similarities[similarities["goodreads_book_id"].isin(liked_ids)]
    if rows.empty:
        return pd.DataFrame()

    scores: dict[int, float] = defaultdict(float)
    evidence: dict[int, int] = defaultdict(int)
    best_co_likes: dict[int, int] = defaultdict(int)

    for row in rows.itertuples(index=False):
        candidate_id = int(row.similar_goodreads_book_id)
        if candidate_id in read_books or candidate_id in liked_ids:
            continue
        source_rating = liked_books[int(row.goodreads_book_id)]
        rating_weight = max(source_rating - 3.0, 0.5)
        scores[candidate_id] += float(row.score) * rating_weight
        evidence[candidate_id] += 1
        best_co_likes[candidate_id] = max(best_co_likes[candidate_id], int(row.co_likes))

    ranked = sorted(scores, key=lambda book_id: (scores[book_id], evidence[book_id]), reverse=True)
    data = [
        {
            "goodreads_book_id": book_id,
            "score": scores[book_id],
            "evidence_books": evidence[book_id],
            "best_co_likes": best_co_likes[book_id],
        }
        for book_id in ranked[:limit]
    ]
    return pd.DataFrame(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--similarities", type=Path, default=Path("data/recommendations/item_cf_similar.csv"))
    parser.add_argument("--interactions", type=Path, default=Path("data/recommendations/interactions_filtered.csv"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--user-id", type=int, help="Known UCSD compact user ID from the filtered interactions")
    source.add_argument(
        "--liked-goodreads-book-id",
        action="append",
        default=[],
        help="Liked Goodreads book ID. Optionally include rating like 21:5. Repeatable.",
    )
    parser.add_argument("--min-like-rating", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.user_id is not None:
        read_books, liked_books = load_user_history(
            args.interactions,
            args.user_id,
            args.min_like_rating,
            args.chunk_size,
        )
        if not liked_books:
            raise SystemExit(f"No liked books found for user_id={args.user_id}.")
    else:
        liked_books = parse_liked_books(args.liked_goodreads_book_id)
        read_books = set(liked_books)

    recommendations = recommend(args.similarities, liked_books, read_books, args.limit)
    if recommendations.empty:
        raise SystemExit("No recommendations found. The liked books may be outside the Level 2 candidate set.")

    print(recommendations.to_string(index=False, formatters={"score": "{:.6f}".format}))


if __name__ == "__main__":
    main()
