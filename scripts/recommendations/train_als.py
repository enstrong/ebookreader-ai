#!/usr/bin/env python3
"""Train an ALS matrix factorization recommender with the implicit library."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares


RATING_COLUMNS = ["user_id", "goodreads_book_id", "is_read", "rating"]
SIGNAL_MODES = ["positive", "mean-centered"]


def read_chunks(path: Path, chunk_size: int):
    dtypes = {"user_id": "int64", "goodreads_book_id": "int64", "is_read": "int8", "rating": "int8"}
    return pd.read_csv(path, usecols=RATING_COLUMNS, dtype=dtypes, chunksize=chunk_size)


def update_counter(counter: Counter[int], counts: pd.Series) -> None:
    for key, value in counts.items():
        counter[int(key)] += int(value)


def scan_training_population(
    interactions_path: Path,
    signal_mode: str,
    min_like_rating: int,
    mean_shrinkage: float,
    chunk_size: int,
    max_rows: int | None,
) -> tuple[Counter[int], dict[int, float], dict]:
    book_counts: Counter[int] = Counter()
    user_rating_sums: defaultdict[int, float] = defaultdict(float)
    user_rating_counts: Counter[int] = Counter()
    observed_users: set[int] = set()
    global_rating_sum = 0.0
    global_rating_count = 0
    stats = {"rows_scanned": 0, "signal_rows": 0}

    for chunk_number, chunk in enumerate(read_chunks(interactions_path, chunk_size), start=1):
        if max_rows is not None:
            remaining = max_rows - stats["rows_scanned"]
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        stats["rows_scanned"] += len(chunk)
        if signal_mode == "positive":
            signal_rows = chunk[chunk["rating"] >= min_like_rating]
        else:
            explicit_rows = chunk[chunk["rating"] > 0]
            read_unrated_rows = chunk[(chunk["rating"] == 0) & (chunk["is_read"] == 1)]
            signal_rows = pd.concat([explicit_rows, read_unrated_rows], ignore_index=True)
            observed_users.update(signal_rows["user_id"].astype(int).tolist())
            user_stats = explicit_rows.groupby("user_id")["rating"].agg(["sum", "count"])
            for user_id, row in user_stats.iterrows():
                user_rating_sums[int(user_id)] += float(row["sum"])
                user_rating_counts[int(user_id)] += int(row["count"])
            global_rating_sum += float(explicit_rows["rating"].sum())
            global_rating_count += int(len(explicit_rows))

        stats["signal_rows"] += len(signal_rows)
        update_counter(book_counts, signal_rows["goodreads_book_id"].value_counts())
        print(
            f"Pass 1 chunk {chunk_number}: scanned={stats['rows_scanned']:,}, "
            f"signals={stats['signal_rows']:,}, books={len(book_counts):,}",
            flush=True,
        )

    if signal_mode == "mean-centered" and global_rating_count == 0:
        raise SystemExit("Mean-centered ALS needs at least one explicit rating.")

    global_mean = global_rating_sum / global_rating_count if global_rating_count else 0.0
    user_means = {
        user_id: (user_rating_sums[user_id] + mean_shrinkage * global_mean)
        / (user_rating_counts[user_id] + mean_shrinkage)
        for user_id in user_rating_counts
    }
    stats["users_with_means"] = len(user_means)
    stats["users_with_no_explicit_ratings"] = max(len(observed_users) - len(user_means), 0)
    stats["global_explicit_rating_mean"] = global_mean
    stats["global_explicit_rating_count"] = global_rating_count
    stats["mean_shrinkage"] = mean_shrinkage
    return book_counts, user_means, stats


def collect_interactions(
    interactions_path: Path,
    candidate_books: set[int],
    signal_mode: str,
    user_means: dict[int, float],
    min_like_rating: int,
    min_centered_magnitude: float,
    centered_scale: float,
    alpha: float,
    global_mean: float,
    min_user_likes: int,
    chunk_size: int,
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict]:
    chunks: list[pd.DataFrame] = []
    stats = {
        "rows_scanned": 0,
        "candidate_signals": 0,
        "candidate_explicit_positive_signals": 0,
        "candidate_explicit_negative_signals": 0,
        "candidate_weak_read_unrated_signals": 0,
    }

    for chunk_number, chunk in enumerate(read_chunks(interactions_path, chunk_size), start=1):
        if max_rows is not None:
            remaining = max_rows - stats["rows_scanned"]
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        stats["rows_scanned"] += len(chunk)
        candidate_rows = chunk[chunk["goodreads_book_id"].isin(candidate_books)].copy()

        if signal_mode == "positive":
            candidate_rows = candidate_rows[candidate_rows["rating"] >= min_like_rating].copy()
            ratings = candidate_rows["rating"].to_numpy(dtype=np.float32)
            signals = 1.0 + alpha * (ratings / 5.0)
            candidate_rows = candidate_rows[["user_id", "goodreads_book_id"]].copy()
            candidate_rows["signal"] = signals.astype(np.float32)
            candidate_rows["signal_kind"] = "explicit_positive"
        else:
            explicit_rows = candidate_rows[candidate_rows["rating"] > 0].copy()
            weak_read_rows = candidate_rows[
                (candidate_rows["rating"] == 0) & (candidate_rows["is_read"] == 1)
            ].copy()

            explicit_signals = pd.DataFrame(columns=["user_id", "goodreads_book_id", "signal", "signal_kind"])
            if not explicit_rows.empty:
                means = explicit_rows["user_id"].map(user_means).fillna(global_mean).to_numpy(dtype=np.float32)
                ratings = explicit_rows["rating"].to_numpy(dtype=np.float32)
                centered = ratings - means
                keep = np.abs(centered) > min_centered_magnitude
                explicit_rows = explicit_rows.loc[keep, ["user_id", "goodreads_book_id"]].copy()
                centered = centered[keep]
                signals = np.sign(centered) * (1.0 + alpha * (np.abs(centered) / centered_scale))
                explicit_rows["signal"] = signals.astype(np.float32)
                explicit_rows["signal_kind"] = np.where(signals > 0, "explicit_positive", "explicit_negative")
                explicit_signals = explicit_rows

            weak_signals = pd.DataFrame(columns=["user_id", "goodreads_book_id", "signal", "signal_kind"])
            if not weak_read_rows.empty:
                weak_signals = weak_read_rows[["user_id", "goodreads_book_id"]].copy()
                weak_signals["signal"] = np.float32(1.0)
                weak_signals["signal_kind"] = "weak_read_unrated"

            candidate_rows = pd.concat([explicit_signals, weak_signals], ignore_index=True)
            if not candidate_rows.empty:
                candidate_rows["priority"] = np.where(candidate_rows["signal_kind"] == "weak_read_unrated", 0, 1)
                candidate_rows = (
                    candidate_rows.sort_values("priority", ascending=False)
                    .drop_duplicates(["user_id", "goodreads_book_id"], keep="first")
                    .drop(columns=["priority"])
                )

        stats["candidate_signals"] += len(candidate_rows)
        stats["candidate_explicit_positive_signals"] += int((candidate_rows["signal_kind"] == "explicit_positive").sum())
        stats["candidate_explicit_negative_signals"] += int((candidate_rows["signal_kind"] == "explicit_negative").sum())
        stats["candidate_weak_read_unrated_signals"] += int((candidate_rows["signal_kind"] == "weak_read_unrated").sum())
        chunks.append(candidate_rows)
        print(
            f"Pass 2 chunk {chunk_number}: scanned={stats['rows_scanned']:,}, "
            f"candidate signals={stats['candidate_signals']:,}, "
            f"weak reads={stats['candidate_weak_read_unrated_signals']:,}",
            flush=True,
        )

    if not chunks:
        raise SystemExit("No interactions matched the ALS filters.")

    interactions = pd.concat(chunks, ignore_index=True)
    if not interactions.empty:
        interactions["priority"] = np.where(interactions["signal_kind"] == "weak_read_unrated", 0, 1)
        interactions = (
            interactions.sort_values("priority", ascending=False)
            .drop_duplicates(["user_id", "goodreads_book_id"], keep="first")
            .drop(columns=["priority"])
        )
    positive_counts = interactions[interactions["signal"] > 0]["user_id"].value_counts()
    active_users = set(positive_counts[positive_counts >= min_user_likes].index.astype(int))
    interactions = interactions[interactions["user_id"].isin(active_users)].copy()
    stats["active_users"] = len(active_users)
    stats["training_rows"] = len(interactions)
    stats["positive_training_rows"] = int((interactions["signal"] > 0).sum())
    stats["negative_training_rows"] = int((interactions["signal"] < 0).sum())
    stats["explicit_positive_signals"] = int((interactions["signal_kind"] == "explicit_positive").sum())
    stats["explicit_negative_signals"] = int((interactions["signal_kind"] == "explicit_negative").sum())
    stats["weak_read_unrated_signals"] = int((interactions["signal_kind"] == "weak_read_unrated").sum())
    return interactions, stats


def build_user_items(interactions: pd.DataFrame) -> tuple[sparse.csr_matrix, dict, dict, dict]:
    user_ids = np.array(sorted(interactions["user_id"].unique()), dtype=np.int64)
    book_ids = np.array(sorted(interactions["goodreads_book_id"].unique()), dtype=np.int64)

    user_to_idx = {int(user_id): index for index, user_id in enumerate(user_ids)}
    book_to_idx = {int(book_id): index for index, book_id in enumerate(book_ids)}
    idx_to_book = {index: int(book_id) for book_id, index in book_to_idx.items()}

    rows = interactions["user_id"].map(user_to_idx).to_numpy(dtype=np.int32)
    cols = interactions["goodreads_book_id"].map(book_to_idx).to_numpy(dtype=np.int32)
    signals = interactions["signal"].to_numpy(dtype=np.float32)

    user_items = sparse.csr_matrix(
        (signals, (rows, cols)),
        shape=(len(user_ids), len(book_ids)),
        dtype=np.float32,
    )
    user_items.sum_duplicates()
    mappings = {
        "user_to_idx": user_to_idx,
        "book_to_idx": book_to_idx,
        "idx_to_book": idx_to_book,
        "user_ids": user_ids.tolist(),
        "book_ids": book_ids.tolist(),
    }
    return user_items, mappings, user_to_idx, book_to_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, default=Path("data/recommendations/interactions_filtered.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/recommendations/als"))
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--signal-mode", choices=SIGNAL_MODES, default="positive")
    parser.add_argument("--min-like-rating", type=int, default=5)
    parser.add_argument("--top-books", type=int, default=5_000)
    parser.add_argument("--min-user-likes", type=int, default=3)
    parser.add_argument("--min-centered-magnitude", type=float, default=0.0)
    parser.add_argument("--centered-scale", type=float, default=4.0)
    parser.add_argument("--mean-shrinkage", type=float, default=5.0)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    book_counts, user_means, pass1 = scan_training_population(
        args.interactions,
        args.signal_mode,
        args.min_like_rating,
        args.mean_shrinkage,
        args.chunk_size,
        args.max_rows,
    )
    if args.top_books > 0:
        candidate_books = {book_id for book_id, _ in book_counts.most_common(args.top_books)}
    else:
        candidate_books = set(book_counts)
    print(f"Candidate books selected for ALS: {len(candidate_books):,}", flush=True)

    interactions, pass2 = collect_interactions(
        args.interactions,
        candidate_books,
        args.signal_mode,
        user_means,
        args.min_like_rating,
        args.min_centered_magnitude,
        args.centered_scale,
        args.alpha,
        pass1["global_explicit_rating_mean"],
        args.min_user_likes,
        args.chunk_size,
        args.max_rows,
    )
    print(f"Training rows after user filter: {len(interactions):,}", flush=True)

    user_items, mappings, _, _ = build_user_items(interactions)
    print(
        f"User-item matrix: shape={user_items.shape}, nonzeros={user_items.nnz:,}, "
        f"density={user_items.nnz / (user_items.shape[0] * user_items.shape[1]):.8f}",
        flush=True,
    )

    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        random_state=args.random_state,
    )
    model.fit(user_items, show_progress=True)

    model_path = args.output_dir / "als_model.pkl"
    mappings_path = args.output_dir / "mappings.pkl"
    matrix_path = args.output_dir / "user_items.npz"
    metadata_path = args.output_dir / "metadata.json"

    with model_path.open("wb") as file:
        pickle.dump(model, file)
    with mappings_path.open("wb") as file:
        pickle.dump(mappings, file)
    sparse.save_npz(matrix_path, user_items)

    metadata = {
        "method": "implicit_als_matrix_factorization",
        "interactions": str(args.interactions),
        "signal_mode": args.signal_mode,
        "min_like_rating": args.min_like_rating,
        "top_books": args.top_books,
        "min_user_likes": args.min_user_likes,
        "min_centered_magnitude": args.min_centered_magnitude,
        "centered_scale": args.centered_scale,
        "mean_shrinkage": args.mean_shrinkage,
        "factors": args.factors,
        "iterations": args.iterations,
        "regularization": args.regularization,
        "alpha": args.alpha,
        "random_state": args.random_state,
        "user_factors_shape": list(model.user_factors.shape),
        "item_factors_shape": list(model.item_factors.shape),
        "user_items_shape": list(user_items.shape),
        "user_items_nonzeros": int(user_items.nnz),
        "user_items_positive_nonzeros": int((user_items.data > 0).sum()),
        "user_items_negative_nonzeros": int((user_items.data < 0).sum()),
        "user_items_density": user_items.nnz / (user_items.shape[0] * user_items.shape[1]),
        "explicit_positive_signals": pass2.get("explicit_positive_signals", 0),
        "explicit_negative_signals": pass2.get("explicit_negative_signals", 0),
        "weak_read_unrated_signals": pass2.get("weak_read_unrated_signals", 0),
        "users_with_no_explicit_ratings": pass1.get("users_with_no_explicit_ratings", 0),
        "global_explicit_rating_mean": pass1.get("global_explicit_rating_mean", 0.0),
        "pass1": pass1,
        "pass2": pass2,
        "artifacts": {
            "model": str(model_path),
            "mappings": str(mappings_path),
            "user_items": str(matrix_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved ALS model: {model_path}", flush=True)
    print(f"Saved mappings: {mappings_path}", flush=True)
    print(f"Saved user-item matrix: {matrix_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)
    print(f"Learned user matrix shape: {model.user_factors.shape}", flush=True)
    print(f"Learned item matrix shape: {model.item_factors.shape}", flush=True)


if __name__ == "__main__":
    main()
