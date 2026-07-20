#!/usr/bin/env python3
"""Build a compact multilingual Goodreads poetry seed from local metadata."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
from pathlib import Path


NO_PHOTO_MARKERS = (
    "s.gr-assets.com/assets/nophoto",
    "nophoto/book",
)

LANGUAGE_GROUPS = {
    "eng": {"eng", "en", "en-us", "en-gb", "english"},
    "ara": {"ara", "ar", "arabic"},
    "spa": {"spa", "es", "spanish", "español"},
    "per": {"per", "fa", "fas", "persian"},
    "por": {"por", "pt", "portuguese", "português"},
}

LANGUAGE_LABELS = {
    "eng": "English",
    "ara": "Arabic",
    "spa": "Spanish",
    "per": "Persian",
    "por": "Portuguese",
}


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
    if not raw:
        return "Poetry"
    if isinstance(raw, list):
        values = [clean_text(item) for item in raw if clean_text(item)]
    elif isinstance(raw, dict):
        values = [clean_text(key) for key, value in raw.items() if value]
    else:
        values = [part.strip() for part in str(raw).split(";") if part.strip()]
    if not any(value.lower() == "poetry" for value in values):
        values.insert(0, "Poetry")
    return ";".join(dict.fromkeys(values))


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_number(value) -> str:
    return str(value or 0)


def select_books(input_path: Path, per_language: int) -> list[dict]:
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    seen_goodreads_ids: set[str] = set()
    seen_titles: set[str] = set()

    with gzip.open(input_path, "rt", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            raw = json.loads(line)
            language = normalize_language(
                raw.get("language_code") or raw.get("language") or raw.get("languageCode") or ""
            )
            if language is None:
                continue

            cover_url = clean_text(
                raw.get("image_url")
                or raw.get("large_image_url")
                or raw.get("small_image_url")
                or ""
            )
            if not has_real_cover(cover_url):
                continue

            goodreads_id = clean_text(raw.get("book_id") or raw.get("bookId") or raw.get("id"))
            title = clean_text(raw.get("title"))
            if not title:
                continue

            dedupe_key = goodreads_id or f"{language}:{title.lower()}"
            title_key = title.lower()
            if dedupe_key in seen_goodreads_ids or title_key in seen_titles:
                continue
            seen_goodreads_ids.add(dedupe_key)
            seen_titles.add(title_key)

            buckets[language].append(
                {
                    "title": title,
                    "author": "",
                    "description": clean_text(raw.get("description"), 2000),
                    "cover_url": cover_url,
                    "goodreads_id": goodreads_id,
                    "average_rating": parse_float(raw.get("average_rating") or raw.get("averageRating")),
                    "ratings_count": parse_int(raw.get("ratings_count") or raw.get("ratingsCount")),
                    "review_count": parse_int(raw.get("text_reviews_count") or raw.get("review_count")),
                    "external_url": clean_text(raw.get("url") or raw.get("link") or raw.get("uri")),
                    "genres": normalize_genres(raw.get("genres")),
                    "language": language,
                    "page_count": parse_int(raw.get("num_pages") or raw.get("numPages") or raw.get("pages")),
                }
            )

    selected = []
    for language in LANGUAGE_GROUPS:
        books = sorted(
            buckets[language],
            key=lambda book: (book["ratings_count"], book["average_rating"]),
            reverse=True,
        )
        selected.extend(books[:per_language])

    return sorted(
        selected,
        key=lambda book: (list(LANGUAGE_GROUPS).index(book["language"]), -book["ratings_count"]),
    )


def write_csv(path: Path, books: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(books[0].keys()))
        writer.writeheader()
        writer.writerows(books)


def write_sql(path: Path, books: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "-- Generated by scripts/build_goodreads_poetry_seed.py",
        "-- Multilingual Goodreads poetry seed with real covers and language metadata.",
        "DELETE FROM book_genres;",
        "DELETE FROM user_books;",
        "DELETE FROM books;",
        "",
    ]

    for book in books:
        columns = (
            "title, author, description, cover_url, goodreads_id, average_rating, "
            "ratings_count, review_count, external_url, language, page_count"
        )
        values = ", ".join(
            [
                sql_string(book["title"]),
                sql_string(book["author"]),
                sql_string(book["description"]),
                sql_string(book["cover_url"]),
                sql_string(book["goodreads_id"]),
                sql_number(book["average_rating"]),
                sql_number(book["ratings_count"]),
                sql_number(book["review_count"]),
                sql_string(book["external_url"]),
                sql_string(book["language"]),
                sql_number(book["page_count"]),
            ]
        )
        lines.append(f"INSERT INTO books ({columns}) VALUES ({values});")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/goodreads_books_poetry.json.gz"))
    parser.add_argument("--per-language", type=int, default=120)
    parser.add_argument("--output-csv", type=Path, default=Path("goodreads_subset_for_import.csv"))
    parser.add_argument("--output-sql", type=Path, default=Path("backend/src/main/resources/data.sql"))
    args = parser.parse_args()

    books = select_books(args.input, args.per_language)
    if not books:
        raise SystemExit("No books matched the selected filters.")

    write_csv(args.output_csv, books)
    write_sql(args.output_sql, books)

    counts = collections.Counter(book["language"] for book in books)
    print(f"Wrote {len(books)} books")
    for language, count in counts.items():
        print(f"{LANGUAGE_LABELS[language]} ({language}): {count}")


if __name__ == "__main__":
    main()
