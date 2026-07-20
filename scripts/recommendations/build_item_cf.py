#!/usr/bin/env python3
"""Build a Level 2 item-item collaborative filtering model from Goodreads ratings.

This is memory-based collaborative filtering. We do not learn latent vectors.
Instead, we count how often users like pairs of books together, then normalize
those co-like counts into an item-item similarity score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd


RATING_COLUMNS = ["user_id", "book_id", "goodreads_book_id", "rating"]


def read_chunks(path: Path, chunk_size: int):
    dtypes = {
        "user_id": "int64",
        "book_id": "int64",
        "goodreads_book_id": "int64",
        "rating": "int8",
    }
    return pd.read_csv(path, usecols=RATING_COLUMNS, dtype=dtypes, chunksize=chunk_size)


def count_likes(
    interactions_path: Path,
    min_like_rating: int,
    chunk_size: int,
    max_rows: int | None,
) -> tuple[Counter, dict[int, int], dict]:
    like_counts: Counter[int] = Counter()
    book_to_goodreads: dict[int, int] = {}
    stats = {"rows_scanned": 0, "liked_rows": 0}

    for chunk_number, chunk in enumerate(read_chunks(interactions_path, chunk_size), start=1):
        if max_rows is not None:
            remaining = max_rows - stats["rows_scanned"]
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        stats["rows_scanned"] += len(chunk)
        liked = chunk[chunk["rating"] >= min_like_rating]
        stats["liked_rows"] += len(liked)
        like_counts.update(liked["book_id"].tolist())

        ids = liked[["book_id", "goodreads_book_id"]].drop_duplicates()
        book_to_goodreads.update(zip(ids["book_id"].tolist(), ids["goodreads_book_id"].tolist()))

        print(
            f"Pass 1 chunk {chunk_number}: scanned={stats['rows_scanned']:,}, "
            f"likes={stats['liked_rows']:,}, books={len(like_counts):,}",
            flush=True,
        )

    return like_counts, book_to_goodreads, stats


def flush_user_pairs(
    liked_books: list[int],
    pair_counts: Counter[tuple[int, int]],
    max_likes_per_user: int,
) -> None:
    unique_books = sorted(set(liked_books))
    if len(unique_books) > max_likes_per_user:
        unique_books = unique_books[:max_likes_per_user]
    for left, right in combinations(unique_books, 2):
        pair_counts[(left, right)] += 1


def count_co_likes(
    interactions_path: Path,
    candidate_books: set[int],
    min_like_rating: int,
    chunk_size: int,
    max_rows: int | None,
    max_likes_per_user: int,
) -> tuple[Counter, dict]:
    pair_counts: Counter[tuple[int, int]] = Counter()
    stats = {"rows_scanned": 0, "liked_candidate_rows": 0, "users_processed": 0}
    current_user_id: int | None = None
    current_likes: list[int] = []

    for chunk_number, chunk in enumerate(read_chunks(interactions_path, chunk_size), start=1):
        if max_rows is not None:
            remaining = max_rows - stats["rows_scanned"]
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        stats["rows_scanned"] += len(chunk)
        liked = chunk[
            (chunk["rating"] >= min_like_rating)
            & (chunk["book_id"].isin(candidate_books))
        ][["user_id", "book_id"]]
        stats["liked_candidate_rows"] += len(liked)

        for row in liked.itertuples(index=False):
            user_id = int(row.user_id)
            book_id = int(row.book_id)
            if current_user_id is None:
                current_user_id = user_id
            elif user_id != current_user_id:
                flush_user_pairs(current_likes, pair_counts, max_likes_per_user)
                stats["users_processed"] += 1
                current_user_id = user_id
                current_likes = []
            current_likes.append(book_id)

        print(
            f"Pass 2 chunk {chunk_number}: scanned={stats['rows_scanned']:,}, "
            f"liked candidates={stats['liked_candidate_rows']:,}, pairs={len(pair_counts):,}",
            flush=True,
        )

    if current_user_id is not None:
        flush_user_pairs(current_likes, pair_counts, max_likes_per_user)
        stats["users_processed"] += 1

    return pair_counts, stats


def write_neighbors(
    output_path: Path,
    pair_counts: Counter[tuple[int, int]],
    like_counts: Counter[int],
    book_to_goodreads: dict[int, int],
    neighbors_per_book: int,
    min_co_likes: int,
) -> int:
    neighbors: dict[int, list[dict]] = defaultdict(list)

    for (left, right), co_likes in pair_counts.items():
        if co_likes < min_co_likes:
            continue
        left_likes = like_counts[left]
        right_likes = like_counts[right]
        cosine = co_likes / math.sqrt(left_likes * right_likes)
        score = cosine * min(co_likes / 50, 1.0)
        neighbors[left].append(
            {
                "book_id": left,
                "goodreads_book_id": book_to_goodreads.get(left),
                "similar_book_id": right,
                "similar_goodreads_book_id": book_to_goodreads.get(right),
                "score": score,
                "co_likes": co_likes,
                "book_likes": left_likes,
                "similar_likes": right_likes,
            }
        )
        neighbors[right].append(
            {
                "book_id": right,
                "goodreads_book_id": book_to_goodreads.get(right),
                "similar_book_id": left,
                "similar_goodreads_book_id": book_to_goodreads.get(left),
                "score": score,
                "co_likes": co_likes,
                "book_likes": right_likes,
                "similar_likes": left_likes,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "book_id",
        "goodreads_book_id",
        "similar_book_id",
        "similar_goodreads_book_id",
        "score",
        "co_likes",
        "book_likes",
        "similar_likes",
    ]
    rows_written = 0
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for book_id in sorted(neighbors):
            ranked = sorted(
                neighbors[book_id],
                key=lambda row: (row["score"], row["co_likes"]),
                reverse=True,
            )[:neighbors_per_book]
            for row in ranked:
                row = row.copy()
                row["score"] = f"{row['score']:.8f}"
                writer.writerow(row)
                rows_written += 1
    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interactions",
        type=Path,
        default=Path("data/recommendations/interactions_filtered.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/recommendations/item_cf_similar.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/recommendations/item_cf_metadata.json"),
    )
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--min-like-rating", type=int, default=4)
    parser.add_argument("--top-books", type=int, default=5_000)
    parser.add_argument("--neighbors-per-book", type=int, default=25)
    parser.add_argument("--min-co-likes", type=int, default=5)
    parser.add_argument("--max-likes-per-user", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    like_counts, book_to_goodreads, pass1_stats = count_likes(
        args.interactions,
        args.min_like_rating,
        args.chunk_size,
        args.max_rows,
    )
    candidate_books = {
        book_id
        for book_id, _ in like_counts.most_common(args.top_books)
    }
    print(f"Candidate books kept for pair counting: {len(candidate_books):,}", flush=True)

    pair_counts, pass2_stats = count_co_likes(
        args.interactions,
        candidate_books,
        args.min_like_rating,
        args.chunk_size,
        args.max_rows,
        args.max_likes_per_user,
    )
    rows_written = write_neighbors(
        args.output,
        pair_counts,
        like_counts,
        book_to_goodreads,
        args.neighbors_per_book,
        args.min_co_likes,
    )

    metadata = {
        "method": "item_item_collaborative_filtering",
        "similarity": "co_likes / sqrt(book_likes * similar_likes), with significance weighting",
        "interactions": str(args.interactions),
        "output": str(args.output),
        "rows_written": rows_written,
        "top_books": args.top_books,
        "neighbors_per_book": args.neighbors_per_book,
        "min_like_rating": args.min_like_rating,
        "min_co_likes": args.min_co_likes,
        "max_likes_per_user": args.max_likes_per_user,
        "pass1": pass1_stats,
        "pass2": pass2_stats,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote neighbors: {args.output} ({rows_written:,} rows)", flush=True)
    print(f"Wrote metadata: {args.metadata}", flush=True)


if __name__ == "__main__":
    main()
