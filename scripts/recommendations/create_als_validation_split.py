#!/usr/bin/env python3
"""Create a real train/validation split for ALS recommendation evaluation.

The split is leave-one-liked-book-out: pick users with enough explicit likes,
hold out one liked book per user, and write a training interactions file with
those user/book pairs removed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


INTERACTION_COLUMNS = ["user_id", "book_id", "goodreads_book_id", "is_read", "rating", "is_reviewed"]
DTYPES = {
    "user_id": "int64",
    "book_id": "int64",
    "goodreads_book_id": "int64",
    "is_read": "int8",
    "rating": "int8",
    "is_reviewed": "int8",
}


def read_chunks(path: Path, chunk_size: int):
    return pd.read_csv(path, usecols=INTERACTION_COLUMNS, dtype=DTYPES, chunksize=chunk_size)


def load_candidate_books(model_dir: Path | None) -> set[int] | None:
    if model_dir is None:
        return None
    with (model_dir / "mappings.pkl").open("rb") as file:
        mappings = pickle.load(file)
    return {int(book_id) for book_id in mappings["book_to_idx"]}


def liked_mask(chunk: pd.DataFrame, min_like_rating: int, candidate_books: set[int] | None) -> pd.Series:
    mask = chunk["rating"] >= min_like_rating
    if candidate_books is not None:
        mask &= chunk["goodreads_book_id"].isin(candidate_books)
    return mask


def count_user_likes(
    interactions: Path,
    chunk_size: int,
    min_like_rating: int,
    candidate_books: set[int] | None,
) -> tuple[Counter, dict]:
    counts: Counter = Counter()
    stats = {
        "rows_scanned": 0,
        "candidate_likes": 0,
        "candidate_books": len(candidate_books) if candidate_books is not None else None,
    }
    for chunk_number, chunk in enumerate(read_chunks(interactions, chunk_size), start=1):
        stats["rows_scanned"] += len(chunk)
        liked = chunk[liked_mask(chunk, min_like_rating, candidate_books)]
        stats["candidate_likes"] += len(liked)
        counts.update(liked["user_id"].astype(int).tolist())
        print(
            f"Pass 1 chunk {chunk_number}: scanned={stats['rows_scanned']:,}, "
            f"candidate likes={stats['candidate_likes']:,}, users with likes={len(counts):,}",
            flush=True,
        )
    return counts, stats


def choose_validation_users(
    counts: Counter,
    min_likes: int,
    max_users: int,
    seed: int,
) -> list[int]:
    eligible = [int(user_id) for user_id, count in counts.items() if count >= min_likes]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:max_users]


def select_holdouts(
    interactions: Path,
    validation_users: set[int],
    chunk_size: int,
    min_like_rating: int,
    candidate_books: set[int] | None,
    seed: int,
) -> tuple[dict[int, dict], dict]:
    rng = random.Random(seed)
    seen_by_user: defaultdict[int, int] = defaultdict(int)
    holdouts: dict[int, dict] = {}
    stats = {"rows_scanned": 0, "candidate_likes_for_validation_users": 0}

    for chunk_number, chunk in enumerate(read_chunks(interactions, chunk_size), start=1):
        stats["rows_scanned"] += len(chunk)
        liked = chunk[liked_mask(chunk, min_like_rating, candidate_books)]
        liked = liked[liked["user_id"].isin(validation_users)]
        stats["candidate_likes_for_validation_users"] += len(liked)

        for row in liked.itertuples(index=False):
            user_id = int(row.user_id)
            seen_by_user[user_id] += 1
            if rng.randrange(seen_by_user[user_id]) == 0:
                holdouts[user_id] = {
                    "user_id": user_id,
                    "book_id": int(row.book_id),
                    "goodreads_book_id": int(row.goodreads_book_id),
                    "is_read": int(row.is_read),
                    "rating": int(row.rating),
                    "is_reviewed": int(row.is_reviewed),
                }

        print(
            f"Pass 2 chunk {chunk_number}: scanned={stats['rows_scanned']:,}, "
            f"validation likes seen={stats['candidate_likes_for_validation_users']:,}, "
            f"holdouts={len(holdouts):,}",
            flush=True,
        )

    stats["users_missing_holdout"] = len(validation_users) - len(holdouts)
    return holdouts, stats


def write_split_files(
    interactions: Path,
    train_output: Path,
    validation_output: Path,
    holdouts: dict[int, dict],
    chunk_size: int,
) -> dict:
    train_output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    train_output.unlink(missing_ok=True)
    validation_output.unlink(missing_ok=True)

    holdout_pairs = {(row["user_id"], row["goodreads_book_id"]) for row in holdouts.values()}
    validation_frame = pd.DataFrame(list(holdouts.values()), columns=INTERACTION_COLUMNS)
    validation_frame.to_csv(validation_output, index=False)

    stats = {
        "rows_scanned": 0,
        "train_rows_written": 0,
        "rows_removed_for_validation": 0,
        "validation_rows_written": len(validation_frame),
    }

    wrote_header = False
    for chunk_number, chunk in enumerate(read_chunks(interactions, chunk_size), start=1):
        stats["rows_scanned"] += len(chunk)
        pairs = list(zip(chunk["user_id"].astype(int), chunk["goodreads_book_id"].astype(int)))
        remove = pd.Series([pair in holdout_pairs for pair in pairs], index=chunk.index)
        train_chunk = chunk[~remove]
        stats["rows_removed_for_validation"] += int(remove.sum())
        stats["train_rows_written"] += len(train_chunk)
        train_chunk.to_csv(train_output, index=False, mode="a", header=not wrote_header)
        wrote_header = True
        print(
            f"Pass 3 chunk {chunk_number}: scanned={stats['rows_scanned']:,}, "
            f"removed={stats['rows_removed_for_validation']:,}, "
            f"train rows={stats['train_rows_written']:,}",
            flush=True,
        )

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, default=Path("data/recommendations/interactions_with_reads.csv"))
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidate-books-model-dir", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--max-users", type=int, default=10_000)
    parser.add_argument("--min-likes", type=int, default=5)
    parser.add_argument("--min-like-rating", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_books = load_candidate_books(args.candidate_books_model_dir)

    user_like_counts, pass1 = count_user_likes(
        args.interactions,
        args.chunk_size,
        args.min_like_rating,
        candidate_books,
    )
    validation_users = choose_validation_users(user_like_counts, args.min_likes, args.max_users, args.seed)
    print(f"Eligible users: {sum(1 for count in user_like_counts.values() if count >= args.min_likes):,}", flush=True)
    print(f"Validation users selected: {len(validation_users):,}", flush=True)

    holdouts, pass2 = select_holdouts(
        args.interactions,
        set(validation_users),
        args.chunk_size,
        args.min_like_rating,
        candidate_books,
        args.seed,
    )
    pass3 = write_split_files(
        args.interactions,
        args.train_output,
        args.validation_output,
        holdouts,
        args.chunk_size,
    )

    summary = {
        "method": "leave_one_liked_book_out_before_training",
        "interactions": str(args.interactions),
        "train_output": str(args.train_output),
        "validation_output": str(args.validation_output),
        "candidate_books_model_dir": str(args.candidate_books_model_dir) if args.candidate_books_model_dir else None,
        "max_users": args.max_users,
        "min_likes": args.min_likes,
        "min_like_rating": args.min_like_rating,
        "seed": args.seed,
        "pass1": pass1,
        "pass2": pass2,
        "pass3": pass3,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote summary: {args.summary}", flush=True)


if __name__ == "__main__":
    main()
