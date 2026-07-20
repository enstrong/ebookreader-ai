#!/usr/bin/env python3
"""Download and filter UCSD Goodreads interactions for recommender experiments.

This script is deliberately terminal-first. It can download the official UCSD
files, then process the large interaction CSV in chunks so the whole 4.1 GB file
never has to sit in memory at once.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DATASET_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"
DEFAULT_DATA_DIR = Path("data/goodreads")

FILES = {
    "book_map": "book_id_map.csv",
    "user_map": "user_id_map.csv",
    "interactions": "goodreads_interactions.csv",
}


def dataset_url(file_name: str) -> str:
    return f"{DATASET_BASE_URL}/{file_name}"


def make_session(max_retries: int) -> requests.Session:
    retry = Retry(
        total=max_retries,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_file(file_name: str, output_path: Path, max_retries: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Already exists, skipping download: {output_path}", flush=True)
        return

    url = dataset_url(file_name)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    session = make_session(max_retries)

    print(f"Downloading {url}", flush=True)
    print(f"Saving to {output_path}", flush=True)
    started = time.time()
    last_progress_at = 0

    for attempt in range(1, max_retries + 1):
        resume_from = temp_path.stat().st_size if temp_path.exists() else 0
        headers = {"User-Agent": "ebookreader-recs/1.0"}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
            print(f"Attempt {attempt}/{max_retries}: resuming at {resume_from / (1024 * 1024):,.1f} MB", flush=True)
        else:
            print(f"Attempt {attempt}/{max_retries}: starting download", flush=True)

        try:
            with session.get(url, stream=True, timeout=(10, 120), headers=headers) as response:
                if resume_from and response.status_code == 200:
                    print("Server ignored resume request; restarting this file", flush=True)
                    temp_path.unlink(missing_ok=True)
                    resume_from = 0
                response.raise_for_status()

                mode = "ab" if resume_from and response.status_code == 206 else "wb"
                with temp_path.open(mode) as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        file.write(chunk)
                        bytes_written = temp_path.stat().st_size
                        if bytes_written - last_progress_at >= 250 * 1024 * 1024:
                            last_progress_at = bytes_written
                            mb = bytes_written / (1024 * 1024)
                            elapsed = max(time.time() - started, 1)
                            print(f"  {mb:,.0f} MB downloaded ({mb / elapsed:,.1f} MB/s)", flush=True)
            break
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise
            print(f"Download interrupted: {exc}. Retrying in 5 seconds...", flush=True)
            time.sleep(5)

    temp_path.replace(output_path)
    bytes_written = output_path.stat().st_size
    print(f"Finished {output_path} ({bytes_written / (1024 * 1024):,.1f} MB)", flush=True)


def download_required_files(data_dir: Path, max_retries: int) -> None:
    for file_name in FILES.values():
        download_file(file_name, data_dir / file_name, max_retries)


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")


def read_interaction_chunks(path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    columns = ["user_id", "book_id", "is_read", "rating", "is_reviewed"]
    dtypes = {
        "user_id": "int64",
        "book_id": "int64",
        "is_read": "int8",
        "rating": "int8",
        "is_reviewed": "int8",
    }
    return pd.read_csv(path, usecols=columns, dtype=dtypes, chunksize=chunk_size)


def observed_interaction_mask(chunk: pd.DataFrame, include_read_unrated: bool) -> pd.Series:
    explicit_rating = chunk["rating"] > 0
    if not include_read_unrated:
        return explicit_rating
    read_unrated = (chunk["rating"] == 0) & (chunk["is_read"] == 1)
    return explicit_rating | read_unrated


def count_observed_interactions(
    path: Path,
    chunk_size: int,
    max_rows: int | None,
    include_read_unrated: bool,
) -> tuple[Counter, Counter, dict]:
    user_counts: Counter = Counter()
    book_counts: Counter = Counter()
    stats = {
        "rows_scanned": 0,
        "explicit_rating_rows": 0,
        "read_unrated_rows": 0,
        "observed_rows": 0,
        "max_rows_used": max_rows,
        "include_read_unrated": include_read_unrated,
    }

    for chunk_number, chunk in enumerate(read_interaction_chunks(path, chunk_size), start=1):
        if max_rows is not None:
            remaining = max_rows - stats["rows_scanned"]
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        stats["rows_scanned"] += len(chunk)
        explicit_rating = chunk["rating"] > 0
        read_unrated = (chunk["rating"] == 0) & (chunk["is_read"] == 1)
        observed = chunk[observed_interaction_mask(chunk, include_read_unrated)]

        stats["explicit_rating_rows"] += int(explicit_rating.sum())
        if include_read_unrated:
            stats["read_unrated_rows"] += int(read_unrated.sum())
        stats["observed_rows"] += len(observed)
        stats["rated_rows"] = stats["explicit_rating_rows"]
        user_counts.update(observed["user_id"].tolist())
        book_counts.update(observed["book_id"].tolist())

        print(
            f"Pass 1 chunk {chunk_number}: scanned={stats['rows_scanned']:,}, "
            f"observed={stats['observed_rows']:,}, explicit={stats['explicit_rating_rows']:,}, "
            f"read-unrated={stats['read_unrated_rows']:,}, "
            f"users={len(user_counts):,}, books={len(book_counts):,}"
        )

    return user_counts, book_counts, stats


def load_book_map(path: Path) -> pd.DataFrame:
    book_map = pd.read_csv(path)
    require_columns(book_map, {"book_id_csv", "book_id"}, path)
    return book_map.rename(columns={"book_id_csv": "book_id", "book_id": "goodreads_book_id"})


def filter_interactions(
    interactions_path: Path,
    book_map_path: Path,
    output_path: Path,
    summary_path: Path,
    min_user_ratings: int,
    min_book_ratings: int,
    chunk_size: int,
    max_rows: int | None,
    include_read_unrated: bool,
) -> None:
    print("Pass 1: counting observed interactions per user and per book")
    user_counts, book_counts, stats = count_observed_interactions(
        interactions_path,
        chunk_size,
        max_rows,
        include_read_unrated,
    )

    active_users = {user_id for user_id, count in user_counts.items() if count >= min_user_ratings}
    popular_books = {book_id for book_id, count in book_counts.items() if count >= min_book_ratings}
    print(f"Users kept: {len(active_users):,} with >= {min_user_ratings} observed interactions")
    print(f"Books kept: {len(popular_books):,} with >= {min_book_ratings} observed interactions")

    book_map = load_book_map(book_map_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    written = 0
    stats["rows_written"] = 0
    stats["users_after_filter"] = len(active_users)
    stats["books_after_filter"] = len(popular_books)
    stats["min_user_ratings"] = min_user_ratings
    stats["min_book_ratings"] = min_book_ratings

    print("Pass 2: writing filtered interactions")
    pass2_rows_scanned = 0
    for chunk_number, chunk in enumerate(read_interaction_chunks(interactions_path, chunk_size), start=1):
        if max_rows is not None:
            remaining = max_rows - pass2_rows_scanned
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        pass2_rows_scanned += len(chunk)

        filtered = chunk[
            observed_interaction_mask(chunk, include_read_unrated)
            & (chunk["user_id"].isin(active_users))
            & (chunk["book_id"].isin(popular_books))
        ].copy()

        if filtered.empty:
            continue

        filtered = filtered.merge(book_map, on="book_id", how="inner")
        filtered = filtered[
            ["user_id", "book_id", "goodreads_book_id", "is_read", "rating", "is_reviewed"]
        ]
        filtered.to_csv(output_path, mode="a", header=written == 0, index=False)
        written += len(filtered)
        stats["rows_written"] = written
        print(f"Pass 2 chunk {chunk_number}: wrote total={written:,}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote filtered interactions: {output_path}")
    print(f"Wrote summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true", help="Download interactions and ID maps first")
    parser.add_argument("--filter", action="store_true", help="Filter the local interaction CSV")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--max-rows", type=int, default=None, help="Read only the first N rows for a quick test")
    parser.add_argument("--min-user-ratings", type=int, default=5)
    parser.add_argument("--min-book-ratings", type=int, default=10)
    parser.add_argument(
        "--include-read-unrated",
        action="store_true",
        help="Keep rows where rating is 0 but is_read is 1 as weak implicit feedback.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/recommendations/interactions_filtered.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/recommendations/interactions_filtered.summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interactions_path = args.data_dir / FILES["interactions"]
    book_map_path = args.data_dir / FILES["book_map"]

    if args.download:
        download_required_files(args.data_dir, args.max_retries)

    if args.filter:
        if not interactions_path.exists():
            raise FileNotFoundError(f"Missing {interactions_path}. Run with --download first.")
        if not book_map_path.exists():
            raise FileNotFoundError(f"Missing {book_map_path}. Run with --download first.")
        filter_interactions(
            interactions_path=interactions_path,
            book_map_path=book_map_path,
            output_path=args.output,
            summary_path=args.summary,
            min_user_ratings=args.min_user_ratings,
            min_book_ratings=args.min_book_ratings,
            chunk_size=args.chunk_size,
            max_rows=args.max_rows,
            include_read_unrated=args.include_read_unrated,
        )

    if not args.download and not args.filter:
        raise SystemExit("Nothing to do. Use --download, --filter, or both.")


if __name__ == "__main__":
    main()
