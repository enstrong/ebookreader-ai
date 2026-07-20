#!/usr/bin/env python3
"""Collapse Goodreads edition IDs to one canonical work-level book ID.

The app now hides duplicate editions, but the UCSD interactions still contain
separate Goodreads book IDs for different editions of the same work. This
script gives the recommender the same cleaner view: every book in the same
Goodreads work is mapped to the most-observed edition in our interaction file.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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


def read_interactions(path: Path, chunk_size: int):
    return pd.read_csv(path, usecols=INTERACTION_COLUMNS, dtype=DTYPES, chunksize=chunk_size)


def load_work_map(path: Path) -> dict[int, str]:
    book_to_work: dict[int, str] = {}
    for chunk in pd.read_csv(path, usecols=["goodreads_book_id", "work_id"], chunksize=500_000):
        for book_id, work_id in zip(chunk["goodreads_book_id"], chunk["work_id"]):
            if pd.isna(book_id) or pd.isna(work_id):
                continue
            work_id_text = str(work_id).strip()
            if not work_id_text:
                continue
            book_to_work[int(book_id)] = work_id_text
    return book_to_work


def count_books(path: Path, chunk_size: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    rows = 0
    for chunk_number, chunk in enumerate(read_interactions(path, chunk_size), start=1):
        rows += len(chunk)
        counts.update(chunk["goodreads_book_id"].astype(int).tolist())
        print(
            f"Count pass chunk {chunk_number}: rows={rows:,}, unique_books={len(counts):,}",
            flush=True,
        )
    return counts


def build_canonical_map(book_counts: Counter[int], book_to_work: dict[int, str]) -> dict[int, int]:
    best_by_work: dict[str, tuple[int, int]] = {}
    for book_id, count in book_counts.items():
        work_key = book_to_work.get(book_id, f"book:{book_id}")
        current = best_by_work.get(work_key)
        if current is None or count > current[1] or (count == current[1] and book_id < current[0]):
            best_by_work[work_key] = (book_id, count)

    return {
        book_id: best_by_work[book_to_work.get(book_id, f"book:{book_id}")][0]
        for book_id in book_counts
    }


def write_canonical_map(
    output: Path,
    canonical_by_book: dict[int, int],
    book_counts: Counter[int],
    book_to_work: dict[int, str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "goodreads_book_id",
                "canonical_goodreads_book_id",
                "work_id",
                "observed_interactions",
            ],
        )
        writer.writeheader()
        for book_id in sorted(canonical_by_book):
            writer.writerow(
                {
                    "goodreads_book_id": book_id,
                    "canonical_goodreads_book_id": canonical_by_book[book_id],
                    "work_id": book_to_work.get(book_id, ""),
                    "observed_interactions": book_counts[book_id],
                }
            )


def canonicalize_frame(frame: pd.DataFrame, canonical_by_book: dict[int, int]) -> pd.DataFrame:
    result = frame.copy()
    mapped = result["goodreads_book_id"].map(canonical_by_book)
    result["goodreads_book_id"] = mapped.fillna(result["goodreads_book_id"]).astype("int64")
    return result


def write_validation(
    validation_input: Path,
    validation_output: Path,
    canonical_by_book: dict[int, int],
) -> tuple[set[tuple[int, int]], dict]:
    validation = pd.read_csv(validation_input, usecols=INTERACTION_COLUMNS, dtype=DTYPES)
    original_unique_books = int(validation["goodreads_book_id"].nunique())
    validation = canonicalize_frame(validation, canonical_by_book)
    validation["priority"] = (validation["rating"] > 0).astype("int8")
    validation = (
        validation.sort_values("priority", ascending=False)
        .drop_duplicates(["user_id", "goodreads_book_id"], keep="first")
        .drop(columns=["priority"])
    )
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(validation_output, index=False)
    holdout_pairs = set(zip(validation["user_id"].astype(int), validation["goodreads_book_id"].astype(int)))
    return holdout_pairs, {
        "input_rows": int(len(pd.read_csv(validation_input, usecols=["user_id"]))),
        "output_rows": int(len(validation)),
        "original_unique_books": original_unique_books,
        "canonical_unique_books": int(validation["goodreads_book_id"].nunique()),
    }


def write_train(
    train_input: Path,
    train_output: Path,
    canonical_by_book: dict[int, int],
    holdout_pairs: set[tuple[int, int]],
    chunk_size: int,
) -> dict:
    train_output.parent.mkdir(parents=True, exist_ok=True)
    train_output.unlink(missing_ok=True)
    stats = {
        "rows_scanned": 0,
        "rows_written": 0,
        "rows_removed_for_validation": 0,
        "changed_book_id_rows": 0,
    }
    wrote_header = False

    for chunk_number, chunk in enumerate(read_interactions(train_input, chunk_size), start=1):
        stats["rows_scanned"] += len(chunk)
        original_ids = chunk["goodreads_book_id"].copy()
        transformed = canonicalize_frame(chunk, canonical_by_book)
        stats["changed_book_id_rows"] += int((original_ids != transformed["goodreads_book_id"]).sum())

        pairs = list(zip(transformed["user_id"].astype(int), transformed["goodreads_book_id"].astype(int)))
        remove = pd.Series([pair in holdout_pairs for pair in pairs], index=transformed.index)
        train_chunk = transformed[~remove]
        stats["rows_removed_for_validation"] += int(remove.sum())
        stats["rows_written"] += len(train_chunk)
        train_chunk.to_csv(train_output, index=False, mode="a", header=not wrote_header)
        wrote_header = True
        print(
            f"Train pass chunk {chunk_number}: scanned={stats['rows_scanned']:,}, "
            f"removed={stats['rows_removed_for_validation']:,}, written={stats['rows_written']:,}",
            flush=True,
        )

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-interactions", type=Path, default=Path("data/recommendations/interactions_with_reads.csv"))
    parser.add_argument(
        "--train-input",
        type=Path,
        default=Path("data/recommendations/interactions_with_reads_train_validation_10k.csv"),
    )
    parser.add_argument(
        "--validation-input",
        type=Path,
        default=Path("data/recommendations/interactions_with_reads_validation_10k.csv"),
    )
    parser.add_argument("--work-map", type=Path, default=Path("data/recommendations/hybrid/goodreads_work_map.csv"))
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/recommendations/interactions_with_reads_work_canonical_train_validation_10k.csv"),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("data/recommendations/interactions_with_reads_work_canonical_validation_10k.csv"),
    )
    parser.add_argument(
        "--canonical-map-output",
        type=Path,
        default=Path("data/recommendations/work_canonical_book_map.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/recommendations/interactions_with_reads_work_canonical.summary.json"),
    )
    parser.add_argument("--chunk-size", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading Goodreads work map: {args.work_map}", flush=True)
    book_to_work = load_work_map(args.work_map)
    print(f"Loaded work IDs for {len(book_to_work):,} Goodreads books", flush=True)

    print(f"Counting observed books in {args.source_interactions}", flush=True)
    book_counts = count_books(args.source_interactions, args.chunk_size)
    canonical_by_book = build_canonical_map(book_counts, book_to_work)
    write_canonical_map(args.canonical_map_output, canonical_by_book, book_counts, book_to_work)

    changed_books = sum(1 for book_id, canonical_id in canonical_by_book.items() if book_id != canonical_id)
    canonical_books = set(canonical_by_book.values())
    print(
        f"Canonical map: {len(canonical_by_book):,} source books -> "
        f"{len(canonical_books):,} canonical books; changed={changed_books:,}",
        flush=True,
    )

    holdout_pairs, validation_stats = write_validation(
        args.validation_input,
        args.validation_output,
        canonical_by_book,
    )
    train_stats = write_train(
        args.train_input,
        args.train_output,
        canonical_by_book,
        holdout_pairs,
        args.chunk_size,
    )

    summary = {
        "method": "goodreads_work_canonicalized_split",
        "source_interactions": str(args.source_interactions),
        "train_input": str(args.train_input),
        "validation_input": str(args.validation_input),
        "work_map": str(args.work_map),
        "train_output": str(args.train_output),
        "validation_output": str(args.validation_output),
        "canonical_map_output": str(args.canonical_map_output),
        "source_unique_books": len(canonical_by_book),
        "canonical_unique_books": len(canonical_books),
        "books_mapped_to_different_id": changed_books,
        "validation": validation_stats,
        "train": train_stats,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote summary: {args.summary}", flush=True)


if __name__ == "__main__":
    main()
