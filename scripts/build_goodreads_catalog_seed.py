#!/usr/bin/env python3
"""Build a Goodreads catalog seed that matches the app recommendation universe.

The ALS 20k artifact is the preferred catalog spine. If that universe does not
contain enough books in the diploma language set, the script fills the remainder
from Goodreads metadata by rating count so Russian and Kazakh are represented.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
from pathlib import Path


LANGUAGE_GROUPS = {
    "eng": {"eng", "en", "en-us", "en-gb", "en-ca", "english"},
    "spa": {"spa", "es", "es-mx", "spanish", "español"},
    "ara": {"ara", "ar", "arabic"},
    "por": {"por", "pt", "portuguese", "português"},
    "rus": {"rus", "ru", "russian", "русский"},
    "kaz": {"kaz", "kk", "kazakh", "қазақша"},
}

NO_PHOTO_MARKERS = (
    "s.gr-assets.com/assets/nophoto",
    "nophoto/book",
)


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


def clean_text(value, limit: int | None = None) -> str:
    text = str(value or "").replace("\x00", "").strip()
    if limit is not None:
        return text[:limit]
    return text


def has_real_cover(url: str) -> bool:
    if not url:
        return False
    lowered = url.lower()
    return not any(marker in lowered for marker in NO_PHOTO_MARKERS)


def normalize_language(value: str) -> str | None:
    normalized = clean_text(value).lower()
    for canonical, aliases in LANGUAGE_GROUPS.items():
        if normalized in aliases:
            return canonical
    return None


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


def normalize_author(raw) -> str:
    if isinstance(raw, list):
        names = []
        for author in raw:
            if isinstance(author, dict):
                name = author.get("name") or author.get("author_id")
            else:
                name = str(author)
            if name:
                names.append(str(name))
        return ", ".join(names)
    if isinstance(raw, dict):
        return clean_text(raw.get("name") or raw.get("author_id"))
    return clean_text(raw)


def load_als_rank(path: Path) -> dict[str, int]:
    with path.open("rb") as file:
        mappings = pickle.load(file)
    return {str(book_id): rank for rank, book_id in enumerate(mappings["book_ids"])}


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
            genres_by_id[book_id] = normalize_genres(row.get("genres"))
    return genres_by_id


def normalize_book(raw: dict, genres_by_id: dict[str, str], als_rank: dict[str, int]) -> dict | None:
    language = normalize_language(raw.get("language_code") or raw.get("language") or "")
    if language is None:
        return None

    goodreads_id = clean_text(raw.get("book_id") or raw.get("id"))
    title = clean_text(raw.get("title_without_series") or raw.get("title"))
    cover_url = clean_text(raw.get("image_url") or raw.get("large_image_url") or raw.get("small_image_url"))
    if not goodreads_id or not title or not has_real_cover(cover_url):
        return None

    return {
        "title": title,
        "author": normalize_author(raw.get("authors")),
        "description": clean_text(raw.get("description"), 2000),
        "cover_url": cover_url,
        "goodreads_id": goodreads_id,
        "average_rating": parse_float(raw.get("average_rating")),
        "ratings_count": parse_int(raw.get("ratings_count")),
        "review_count": parse_int(raw.get("text_reviews_count")),
        "external_url": clean_text(raw.get("url") or raw.get("link")),
        "genres": genres_by_id.get(goodreads_id) or normalize_genres(raw.get("genres")),
        "language": language,
        "page_count": parse_int(raw.get("num_pages")),
        "als_rank": als_rank.get(goodreads_id, ""),
    }


def select_books(
    books_path: Path,
    genres_path: Path,
    mappings_path: Path,
    max_books: int,
    min_per_language: int,
) -> list[dict]:
    als_rank = load_als_rank(mappings_path)
    genres_by_id = load_genres(genres_path)
    als_books: list[dict] = []
    filler_books: list[dict] = []
    seen: set[str] = set()

    with gzip.open(books_path, "rt", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            book = normalize_book(json.loads(line), genres_by_id, als_rank)
            if book is None or book["goodreads_id"] in seen:
                continue
            seen.add(book["goodreads_id"])
            if book["goodreads_id"] in als_rank:
                als_books.append(book)
            else:
                filler_books.append(book)

    als_books.sort(key=lambda book: int(book["als_rank"]))
    filler_books.sort(key=lambda book: (book["ratings_count"], book["average_rating"]), reverse=True)
    ranked_books = als_books + filler_books
    selected_by_id: dict[str, dict] = {}

    for language in LANGUAGE_GROUPS:
        language_books = [book for book in ranked_books if book["language"] == language]
        language_books.sort(
            key=lambda book: (
                0 if book["goodreads_id"] in als_rank else 1,
                int(book["als_rank"]) if book["goodreads_id"] in als_rank else 10**9,
                -book["ratings_count"],
                -book["average_rating"],
            )
        )
        for book in language_books[:min_per_language]:
            selected_by_id[book["goodreads_id"]] = book

    for book in ranked_books:
        if len(selected_by_id) >= max_books:
            break
        selected_by_id.setdefault(book["goodreads_id"], book)

    selected = sorted(
        selected_by_id.values(),
        key=lambda book: (
            0 if book["goodreads_id"] in als_rank else 1,
            int(book["als_rank"]) if book["goodreads_id"] in als_rank else 10**9,
            -book["ratings_count"],
            -book["average_rating"],
        ),
    )[:max_books]
    for book in selected:
        book.pop("als_rank", None)
    return selected


def write_csv(path: Path, books: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(books[0].keys()))
        writer.writeheader()
        writer.writerows(books)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--books-json", type=Path, default=Path("data/goodreads/goodreads_books.json.gz"))
    parser.add_argument("--genres-json", type=Path, default=Path("data/goodreads/goodreads_book_genres_initial.json.gz"))
    parser.add_argument("--als-mappings", type=Path, default=Path("data/recommendations/als_reads_20k/mappings.pkl"))
    parser.add_argument("--max-books", type=int, default=20000)
    parser.add_argument("--min-per-language", type=int, default=100)
    parser.add_argument("--output-csv", type=Path, default=Path("goodreads_catalog_seed_20k.csv"))
    args = parser.parse_args()

    books = select_books(
        args.books_json,
        args.genres_json,
        args.als_mappings,
        args.max_books,
        args.min_per_language,
    )
    if not books:
        raise SystemExit("No Goodreads books matched the selected catalog filters.")
    write_csv(args.output_csv, books)
    print(f"Wrote {len(books):,} books to {args.output_csv}")


if __name__ == "__main__":
    main()
