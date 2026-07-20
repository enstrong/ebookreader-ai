#!/usr/bin/env python3
"""Train mean-centered implicit ALS with NumPy only.

This is a teaching implementation. It does the ALS math directly with
``np.linalg.solve`` instead of delegating the factorization to ``implicit``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


RATING_COLUMNS = ["user_id", "goodreads_book_id", "rating"]


def read_chunks(path: Path, chunk_size: int):
    dtypes = {"user_id": "int64", "goodreads_book_id": "int64", "rating": "int8"}
    return pd.read_csv(path, usecols=RATING_COLUMNS, dtype=dtypes, chunksize=chunk_size)


def update_counter(counter: Counter[int], counts: pd.Series) -> None:
    for key, value in counts.items():
        counter[int(key)] += int(value)


def scan_book_counts_and_user_means(
    interactions_path: Path,
    chunk_size: int,
    max_rows: int | None,
) -> tuple[Counter[int], dict[int, float], dict]:
    book_counts: Counter[int] = Counter()
    user_rating_sums: defaultdict[int, float] = defaultdict(float)
    user_rating_counts: Counter[int] = Counter()
    stats = {"rows_scanned": 0}

    for chunk_number, chunk in enumerate(read_chunks(interactions_path, chunk_size), start=1):
        if max_rows is not None:
            remaining = max_rows - stats["rows_scanned"]
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        stats["rows_scanned"] += len(chunk)
        update_counter(book_counts, chunk["goodreads_book_id"].value_counts())

        user_stats = chunk.groupby("user_id")["rating"].agg(["sum", "count"])
        for user_id, row in user_stats.iterrows():
            user_rating_sums[int(user_id)] += float(row["sum"])
            user_rating_counts[int(user_id)] += int(row["count"])

        print(
            f"Pass 1 chunk {chunk_number}: scanned={stats['rows_scanned']:,}, "
            f"books={len(book_counts):,}, users={len(user_rating_counts):,}",
            flush=True,
        )

    user_means = {
        user_id: user_rating_sums[user_id] / user_rating_counts[user_id]
        for user_id in user_rating_counts
    }
    stats["books_seen"] = len(book_counts)
    stats["users_with_means"] = len(user_means)
    return book_counts, user_means, stats


def collect_mean_centered_signals(
    interactions_path: Path,
    candidate_books: set[int],
    user_means: dict[int, float],
    min_centered_magnitude: float,
    centered_scale: float,
    alpha: float,
    min_user_likes: int,
    chunk_size: int,
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict]:
    chunks: list[pd.DataFrame] = []
    positive_counts: Counter[int] = Counter()
    stats = {"rows_scanned": 0, "candidate_rows": 0}

    for chunk_number, chunk in enumerate(read_chunks(interactions_path, chunk_size), start=1):
        if max_rows is not None:
            remaining = max_rows - stats["rows_scanned"]
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        stats["rows_scanned"] += len(chunk)
        rows = chunk[chunk["goodreads_book_id"].isin(candidate_books)].copy()
        means = rows["user_id"].map(user_means).to_numpy(dtype=np.float32)
        ratings = rows["rating"].to_numpy(dtype=np.float32)
        centered = ratings - means
        keep = np.abs(centered) > min_centered_magnitude

        rows = rows.loc[keep, ["user_id", "goodreads_book_id"]].copy()
        centered = centered[keep]
        confidence = 1.0 + alpha * (np.abs(centered) / centered_scale)
        preference = centered > 0

        rows["confidence"] = confidence.astype(np.float32)
        rows["preference"] = preference.astype(np.uint8)
        stats["candidate_rows"] += len(rows)

        positive_rows = rows[rows["preference"] == 1]
        positive_counts.update(positive_rows["user_id"].tolist())
        chunks.append(rows)

        print(
            f"Pass 2 chunk {chunk_number}: scanned={stats['rows_scanned']:,}, "
            f"signals={stats['candidate_rows']:,}, users with positives={len(positive_counts):,}",
            flush=True,
        )

    if not chunks:
        raise SystemExit("No mean-centered signals matched the filters.")

    signals = pd.concat(chunks, ignore_index=True)
    active_users = {
        user_id
        for user_id, count in positive_counts.items()
        if count >= min_user_likes
    }
    signals = signals[signals["user_id"].isin(active_users)].copy()
    stats["active_users"] = len(active_users)
    stats["training_rows"] = len(signals)
    stats["positive_training_rows"] = int((signals["preference"] == 1).sum())
    stats["negative_training_rows"] = int((signals["preference"] == 0).sum())
    return signals, stats


def build_indexed_arrays(signals: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    user_ids = np.array(sorted(signals["user_id"].unique()), dtype=np.int64)
    book_ids = np.array(sorted(signals["goodreads_book_id"].unique()), dtype=np.int64)

    user_to_idx = {int(user_id): index for index, user_id in enumerate(user_ids)}
    book_to_idx = {int(book_id): index for index, book_id in enumerate(book_ids)}
    idx_to_book = {index: int(book_id) for book_id, index in book_to_idx.items()}

    user_indices = signals["user_id"].map(user_to_idx).to_numpy(dtype=np.int32)
    item_indices = signals["goodreads_book_id"].map(book_to_idx).to_numpy(dtype=np.int32)
    confidences = signals["confidence"].to_numpy(dtype=np.float32)
    preferences = signals["preference"].to_numpy(dtype=np.float32)

    mappings = {
        "user_to_idx": user_to_idx,
        "book_to_idx": book_to_idx,
        "idx_to_book": idx_to_book,
        "user_ids": user_ids.tolist(),
        "book_ids": book_ids.tolist(),
    }
    return user_indices, item_indices, confidences, preferences, mappings


def build_compressed_rows(
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    confidences: np.ndarray,
    preferences: np.ndarray,
    row_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((col_indices, row_indices))
    sorted_rows = row_indices[order]
    sorted_cols = col_indices[order]
    sorted_confidences = confidences[order]
    sorted_preferences = preferences[order]

    counts = np.bincount(sorted_rows, minlength=row_count)
    indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return indptr, sorted_cols, sorted_confidences, sorted_preferences


def solve_factor_rows(
    fixed_factors: np.ndarray,
    row_indptr: np.ndarray,
    row_indices: np.ndarray,
    row_confidences: np.ndarray,
    row_preferences: np.ndarray,
    regularization: float,
) -> np.ndarray:
    row_count = len(row_indptr) - 1
    factors = fixed_factors.shape[1]
    gram = fixed_factors.T @ fixed_factors
    identity = np.eye(factors, dtype=np.float32)
    solved = np.zeros((row_count, factors), dtype=np.float32)

    for row_id in range(row_count):
        start = row_indptr[row_id]
        end = row_indptr[row_id + 1]
        if start == end:
            continue

        related_indices = row_indices[start:end]
        related_factors = fixed_factors[related_indices]
        confidence = row_confidences[start:end]
        preference = row_preferences[start:end]

        weighted_factors = related_factors.T * (confidence - 1.0)
        a_matrix = gram + weighted_factors @ related_factors + regularization * identity
        b_vector = related_factors.T @ (confidence * preference)
        solved[row_id] = np.linalg.solve(a_matrix, b_vector).astype(np.float32)

    return solved


def train_numpy_als(
    user_indptr: np.ndarray,
    user_item_indices: np.ndarray,
    user_confidences: np.ndarray,
    user_preferences: np.ndarray,
    item_indptr: np.ndarray,
    item_user_indices: np.ndarray,
    item_confidences: np.ndarray,
    item_preferences: np.ndarray,
    user_count: int,
    item_count: int,
    factors: int,
    iterations: int,
    regularization: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    user_factors = np.zeros((user_count, factors), dtype=np.float32)
    item_factors = rng.normal(0.0, 0.01, size=(item_count, factors)).astype(np.float32)

    for iteration in range(1, iterations + 1):
        started_at = time.time()
        user_factors = solve_factor_rows(
            item_factors,
            user_indptr,
            user_item_indices,
            user_confidences,
            user_preferences,
            regularization,
        )
        item_factors = solve_factor_rows(
            user_factors,
            item_indptr,
            item_user_indices,
            item_confidences,
            item_preferences,
            regularization,
        )
        elapsed = time.time() - started_at
        print(f"Iteration {iteration}/{iterations}: {elapsed:.2f}s", flush=True)

    return user_factors, item_factors


def save_row_artifacts(
    output_dir: Path,
    prefix: str,
    indptr: np.ndarray,
    indices: np.ndarray,
    confidences: np.ndarray,
    preferences: np.ndarray,
) -> str:
    path = output_dir / f"{prefix}_rows.npz"
    np.savez_compressed(
        path,
        indptr=indptr,
        indices=indices,
        confidences=confidences,
        preferences=preferences,
    )
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, default=Path("data/recommendations/interactions_filtered_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/recommendations/als_numpy_sample"))
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--top-books", type=int, default=1_000)
    parser.add_argument("--min-user-likes", type=int, default=3)
    parser.add_argument("--min-centered-magnitude", type=float, default=0.0)
    parser.add_argument("--centered-scale", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    book_counts, user_means, pass1 = scan_book_counts_and_user_means(
        args.interactions,
        args.chunk_size,
        args.max_rows,
    )
    if args.top_books > 0:
        candidate_books = {book_id for book_id, _ in book_counts.most_common(args.top_books)}
    else:
        candidate_books = set(book_counts)
    print(f"Candidate books selected: {len(candidate_books):,}", flush=True)

    signals, pass2 = collect_mean_centered_signals(
        args.interactions,
        candidate_books,
        user_means,
        args.min_centered_magnitude,
        args.centered_scale,
        args.alpha,
        args.min_user_likes,
        args.chunk_size,
        args.max_rows,
    )

    user_indices, item_indices, confidences, preferences, mappings = build_indexed_arrays(signals)
    user_count = len(mappings["user_ids"])
    item_count = len(mappings["book_ids"])

    user_rows = build_compressed_rows(
        user_indices,
        item_indices,
        confidences,
        preferences,
        user_count,
    )
    item_rows = build_compressed_rows(
        item_indices,
        user_indices,
        confidences,
        preferences,
        item_count,
    )

    print(
        f"Compressed matrix: users={user_count:,}, items={item_count:,}, "
        f"signals={len(confidences):,}",
        flush=True,
    )

    user_factors, item_factors = train_numpy_als(
        *user_rows,
        *item_rows,
        user_count,
        item_count,
        args.factors,
        args.iterations,
        args.regularization,
        args.random_state,
    )

    user_factors_path = args.output_dir / "user_factors.npy"
    item_factors_path = args.output_dir / "item_factors.npy"
    mappings_path = args.output_dir / "mappings.pkl"
    metadata_path = args.output_dir / "metadata.json"

    np.save(user_factors_path, user_factors)
    np.save(item_factors_path, item_factors)
    with mappings_path.open("wb") as file:
        pickle.dump(mappings, file)

    user_rows_path = save_row_artifacts(args.output_dir, "user", *user_rows)
    item_rows_path = save_row_artifacts(args.output_dir, "item", *item_rows)

    metadata = {
        "method": "numpy_mean_centered_implicit_als",
        "interactions": str(args.interactions),
        "top_books": args.top_books,
        "min_user_likes": args.min_user_likes,
        "min_centered_magnitude": args.min_centered_magnitude,
        "centered_scale": args.centered_scale,
        "alpha": args.alpha,
        "factors": args.factors,
        "iterations": args.iterations,
        "regularization": args.regularization,
        "random_state": args.random_state,
        "user_factors_shape": list(user_factors.shape),
        "item_factors_shape": list(item_factors.shape),
        "signals": int(len(confidences)),
        "positive_signals": int(preferences.sum()),
        "negative_signals": int(len(preferences) - preferences.sum()),
        "pass1": pass1,
        "pass2": pass2,
        "artifacts": {
            "user_factors": str(user_factors_path),
            "item_factors": str(item_factors_path),
            "mappings": str(mappings_path),
            "user_rows": user_rows_path,
            "item_rows": item_rows_path,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Saved user factors: {user_factors_path}", flush=True)
    print(f"Saved item factors: {item_factors_path}", flush=True)
    print(f"Saved mappings: {mappings_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
