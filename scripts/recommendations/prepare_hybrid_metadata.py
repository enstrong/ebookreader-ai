#!/usr/bin/env python3
"""Prepare Goodreads metadata for the ALS candidate universe used by hybrid reranking."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goodreads_import import download_dataset_file  # noqa: E402


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def iter_json_records(path: Path):
    with open_text(path) as file:
        first = file.readline()
        if not first:
            return
        stripped = first.lstrip()
        if stripped.startswith("["):
            for record in json.loads(first + file.read()):
                yield record
        else:
            if stripped:
                yield json.loads(stripped)
            for line in file:
                if line.strip():
                    yield json.loads(line)


def parse_int(value: Any) -> int:
    try:
        if value in (None, "", "\\N"):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def parse_float(value: Any) -> float:
    try:
        if value in (None, "", "\\N"):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def extract_authors(raw: dict[str, Any]) -> tuple[str, str]:
    authors = raw.get("authors")
    values: list[str] = []
    if isinstance(authors, list):
        for author in authors:
            if isinstance(author, dict):
                value = author.get("name") or author.get("author_id") or author.get("id")
            else:
                value = author
            value = clean_text(value)
            if value:
                values.append(value)
    elif isinstance(authors, dict):
        value = clean_text(authors.get("name") or authors.get("author_id") or authors.get("id"))
        if value:
            values.append(value)
    else:
        value = clean_text(authors)
        if value:
            values.append(value)

    author = ", ".join(values)
    author_key = "|".join(sorted(normalize_key(value) for value in values if normalize_key(value)))
    return author, author_key


def normalize_genres(raw_genres: Any) -> str:
    if not raw_genres:
        return ""
    if isinstance(raw_genres, dict):
        return ";".join(sorted(str(key).strip().lower() for key, value in raw_genres.items() if value))
    if isinstance(raw_genres, list):
        return ";".join(sorted(str(value).strip().lower() for value in raw_genres if value))
    return clean_text(raw_genres).lower()


def load_genres(path: Path, target_ids: set[int]) -> dict[int, str]:
    genres_by_book: dict[int, str] = {}
    if not path.exists():
        return genres_by_book

    for raw in iter_json_records(path):
        book_id = parse_int(raw.get("book_id") or raw.get("bookId") or raw.get("id"))
        if book_id not in target_ids:
            continue
        genres = normalize_genres(raw.get("genres") or raw.get("genre"))
        if genres:
            genres_by_book[book_id] = genres
    return genres_by_book


def load_target_ids(model_dir: Path) -> set[int]:
    with (model_dir / "mappings.pkl").open("rb") as file:
        mappings = pickle.load(file)
    return {int(book_id) for book_id in mappings["book_to_idx"].keys()}


def prepare_metadata(
    model_dir: Path,
    books_json: Path,
    genres_json: Path,
    output: Path,
    summary: Path,
    min_coverage: float,
    download_missing: bool,
) -> None:
    if download_missing and not books_json.exists():
        download_dataset_file("goodreads_books.json.gz", books_json)
    if download_missing and not genres_json.exists():
        download_dataset_file("goodreads_book_genres_initial.json.gz", genres_json)

    if not books_json.exists():
        raise SystemExit(f"Missing Goodreads books metadata: {books_json}")

    target_ids = load_target_ids(model_dir)
    genres_by_book = load_genres(genres_json, target_ids)

    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    records_scanned = 0
    for raw in iter_json_records(books_json):
        records_scanned += 1
        book_id = parse_int(raw.get("book_id") or raw.get("bookId") or raw.get("id"))
        if book_id not in target_ids or book_id in seen:
            continue
        seen.add(book_id)
        author, author_key = extract_authors(raw)
        genres = genres_by_book.get(book_id) or normalize_genres(raw.get("genres") or raw.get("genre"))
        rows.append(
            {
                "goodreads_book_id": book_id,
                "title": clean_text(raw.get("title") or raw.get("title_without_series")),
                "author": author,
                "author_key": author_key,
                "genres": genres,
                "average_rating": parse_float(raw.get("average_rating") or raw.get("averageRating")),
                "ratings_count": parse_int(raw.get("ratings_count") or raw.get("ratingsCount")),
                "page_count": parse_int(raw.get("num_pages") or raw.get("numPages") or raw.get("pages")),
                "language": clean_text(raw.get("language_code") or raw.get("languageCode") or raw.get("language")).lower(),
            }
        )
        if len(rows) == len(target_ids):
            break

    coverage = len(rows) / len(target_ids) if target_ids else 0.0
    genre_coverage = sum(1 for row in rows if row["genres"]) / len(rows) if rows else 0.0
    author_coverage = sum(1 for row in rows if row["author_key"]) / len(rows) if rows else 0.0

    if coverage < min_coverage:
        raise SystemExit(
            f"Metadata coverage {coverage:.2%} is below required {min_coverage:.2%}; "
            f"matched {len(rows):,}/{len(target_ids):,} ALS books."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "goodreads_book_id",
                "title",
                "author",
                "author_key",
                "genres",
                "average_rating",
                "ratings_count",
                "page_count",
                "language",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: int(row["goodreads_book_id"])))

    payload = {
        "model_dir": str(model_dir),
        "books_json": str(books_json),
        "genres_json": str(genres_json),
        "output": str(output),
        "target_books": len(target_ids),
        "metadata_rows": len(rows),
        "records_scanned": records_scanned,
        "coverage": coverage,
        "genre_coverage": genre_coverage,
        "author_coverage": author_coverage,
        "min_coverage": min_coverage,
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/recommendations/experiments/als_reads_20k_f256_i10_lam1p0_validation_split"),
    )
    parser.add_argument("--books-json", type=Path, default=Path("data/goodreads/goodreads_books.json.gz"))
    parser.add_argument(
        "--genres-json",
        type=Path,
        default=Path("data/goodreads/goodreads_book_genres_initial.json.gz"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/recommendations/hybrid/book_metadata_20k.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/recommendations/hybrid/book_metadata_20k.summary.json"))
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_metadata(
        args.model_dir,
        args.books_json,
        args.genres_json,
        args.output,
        args.summary,
        args.min_coverage,
        not args.no_download,
    )


if __name__ == "__main__":
    main()
