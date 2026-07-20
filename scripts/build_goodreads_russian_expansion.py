#!/usr/bin/env python3
"""Build a Russian Goodreads catalog expansion for multilingual recommendations.

The current ALS model is mostly English, but Goodreads `work_id` lets us connect
Russian editions to model-known works. This script exports Russian editions that
are useful for display and content fallback:

1. Russian editions whose work exists in the ALS model.
2. Popular Russian editions outside the ALS work universe.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
from pathlib import Path


NO_PHOTO_MARKERS = (
    "s.gr-assets.com/assets/nophoto",
    "nophoto/book",
)

RUSSIAN_LANGUAGE_ALIASES = {"rus", "ru", "russian", "русский"}


def clean_text(value, limit: int | None = None) -> str:
    text = str(value or "").replace("\x00", "").strip()
    return text[:limit] if limit is not None else text


def parse_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def parse_float(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def has_real_cover(url: str) -> bool:
    lowered = clean_text(url).lower()
    return bool(lowered) and not any(marker in lowered for marker in NO_PHOTO_MARKERS)


def is_russian_language(value: str) -> bool:
    return clean_text(value).lower() in RUSSIAN_LANGUAGE_ALIASES


def normalize_genres(raw) -> str:
    if isinstance(raw, dict):
        ranked = sorted(raw.items(), key=lambda item: parse_int(item[1]), reverse=True)
        values = [clean_text(name) for name, count in ranked if parse_int(count) > 0]
    elif isinstance(raw, list):
        values = [clean_text(item) for item in raw if clean_text(item)]
    elif raw:
        values = [part.strip() for part in str(raw).split(";") if part.strip()]
    else:
        values = []
    return ";".join(dict.fromkeys(values))


def load_author_names(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    if not path.exists():
        return names
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            author_id = clean_text(row.get("author_id"))
            name = clean_text(row.get("name"))
            if author_id and name:
                names[author_id] = name
    return names


def normalize_author(raw, author_names: dict[str, str]) -> str:
    names: list[str] = []
    if isinstance(raw, list):
        for author in raw:
            if isinstance(author, dict):
                author_id = clean_text(author.get("author_id"))
                name = clean_text(author.get("name")) or author_names.get(author_id, "")
            else:
                name = clean_text(author)
            if name:
                names.append(name)
    elif isinstance(raw, dict):
        author_id = clean_text(raw.get("author_id"))
        name = clean_text(raw.get("name")) or author_names.get(author_id, "")
        if name:
            names.append(name)
    elif raw:
        names.append(clean_text(raw))
    return ", ".join(dict.fromkeys(names))


def load_genres(path: Path) -> dict[str, str]:
    genres_by_id: dict[str, str] = {}
    if not path.exists():
        return genres_by_id
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            book_id = clean_text(row.get("book_id"))
            if book_id:
                genres_by_id[book_id] = normalize_genres(row.get("genres"))
    return genres_by_id


def load_work_map(path: Path) -> dict[str, str]:
    work_by_book: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            book_id = clean_text(row.get("goodreads_book_id"))
            work_id = clean_text(row.get("work_id"))
            if book_id and work_id:
                work_by_book[book_id] = work_id
    return work_by_book


def load_model_works(mappings_path: Path, work_by_book: dict[str, str]) -> set[str]:
    with mappings_path.open("rb") as file:
        mappings = pickle.load(file)
    model_ids = mappings.get("book_to_idx", {}).keys()
    return {work_by_book.get(str(book_id), "") for book_id in model_ids if work_by_book.get(str(book_id), "")}


def normalize_book(
    raw: dict,
    genres_by_id: dict[str, str],
    author_names: dict[str, str],
    work_by_book: dict[str, str],
    model_works: set[str],
) -> dict | None:
    language = raw.get("language_code") or raw.get("language") or ""
    if not is_russian_language(language):
        return None

    goodreads_id = clean_text(raw.get("book_id") or raw.get("id"))
    title = clean_text(raw.get("title_without_series") or raw.get("title"))
    cover_url = clean_text(raw.get("image_url") or raw.get("large_image_url") or raw.get("small_image_url"))
    ratings_count = parse_int(raw.get("ratings_count"))
    if not goodreads_id or not title or not has_real_cover(cover_url):
        return None

    work_id = work_by_book.get(goodreads_id, "")
    linked_to_model = bool(work_id and work_id in model_works)
    return {
        "title": title,
        "author": normalize_author(raw.get("authors"), author_names),
        "description": clean_text(raw.get("description"), 2000),
        "cover_url": cover_url,
        "goodreads_id": goodreads_id,
        "average_rating": parse_float(raw.get("average_rating")),
        "ratings_count": ratings_count,
        "review_count": parse_int(raw.get("text_reviews_count")),
        "external_url": clean_text(raw.get("url") or raw.get("link")),
        "genres": genres_by_id.get(goodreads_id) or normalize_genres(raw.get("genres")),
        "language": "rus",
        "page_count": parse_int(raw.get("num_pages")),
        "work_id": work_id,
        "linked_to_model": linked_to_model,
    }


def select_books(
    books_json: Path,
    genres_json: Path,
    authors_json: Path,
    work_map: Path,
    mappings_path: Path,
    max_books: int,
    min_ratings: int,
) -> list[dict]:
    genres_by_id = load_genres(genres_json)
    author_names = load_author_names(authors_json)
    work_by_book = load_work_map(work_map)
    model_works = load_model_works(mappings_path, work_by_book)

    selected: list[dict] = []
    with gzip.open(books_json, "rt", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            book = normalize_book(json.loads(line), genres_by_id, author_names, work_by_book, model_works)
            if book is None:
                continue
            if not book["linked_to_model"] and book["ratings_count"] < min_ratings:
                continue
            selected.append(book)

    selected.sort(
        key=lambda book: (
            0 if book["linked_to_model"] else 1,
            -book["ratings_count"],
            -book["average_rating"],
            book["title"],
        )
    )
    return selected[:max_books]


def write_csv(path: Path, books: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "title",
        "author",
        "description",
        "cover_url",
        "goodreads_id",
        "average_rating",
        "ratings_count",
        "review_count",
        "external_url",
        "genres",
        "language",
        "page_count",
        "work_id",
        "linked_to_model",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(books)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--books-json", type=Path, default=Path("data/goodreads/goodreads_books.json.gz"))
    parser.add_argument("--genres-json", type=Path, default=Path("data/goodreads/goodreads_book_genres_initial.json.gz"))
    parser.add_argument("--authors-json", type=Path, default=Path("data/goodreads/goodreads_book_authors.json.gz"))
    parser.add_argument("--work-map", type=Path, default=Path("data/recommendations/hybrid/goodreads_work_map.csv"))
    parser.add_argument(
        "--als-mappings",
        type=Path,
        default=Path("data/recommendations/experiments/als_reads_20k_f256_i10_lam1p0_validation_split/mappings.pkl"),
    )
    parser.add_argument("--max-books", type=int, default=3000)
    parser.add_argument("--min-ratings", type=int, default=10)
    parser.add_argument("--output-csv", type=Path, default=Path("data/recommendations/russian_catalog_expansion.csv"))
    args = parser.parse_args()

    books = select_books(
        args.books_json,
        args.genres_json,
        args.authors_json,
        args.work_map,
        args.als_mappings,
        args.max_books,
        args.min_ratings,
    )
    if not books:
        raise SystemExit("No Russian Goodreads books matched the selected filters.")
    write_csv(args.output_csv, books)
    linked = sum(1 for book in books if book["linked_to_model"])
    print(f"Wrote {len(books):,} Russian books to {args.output_csv}")
    print(f"Linked to current ALS works: {linked:,}")
    print(f"Popular Russian metadata fallback: {len(books) - linked:,}")


if __name__ == "__main__":
    main()
