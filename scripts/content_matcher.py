#!/usr/bin/env python3
"""Find reviewable public-domain content candidates for Goodreads catalog rows.

The script deliberately does not import or download book content. It writes a
JSON review queue so the admin can choose which EPUB/TXT/audio assets should be
attached to catalog books.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests


@dataclass
class Candidate:
    source: str
    confidence: str
    title: str
    author: str
    language: str
    formats: list[str]
    url: str
    notes: str


def normalize(value: str) -> str:
    value = (value or "").lower()
    value = re.sub(r"[^a-zа-яё0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def confidence_for(row: dict, title: str, author: str) -> str:
    wanted_title = normalize(row.get("title", ""))
    wanted_author = normalize(row.get("author", ""))
    found_title = normalize(title)
    found_author = normalize(author)
    if wanted_title and wanted_title == found_title and wanted_author and wanted_author in found_author:
        return "high"
    if wanted_title and (wanted_title in found_title or found_title in wanted_title) and wanted_author and wanted_author in found_author:
        return "medium"
    if wanted_title and (wanted_title in found_title or found_title in wanted_title):
        return "low"
    return "reject"


def load_catalog(path: Path, limit: int | None) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return rows[:limit] if limit else rows


def query_gutenberg(row: dict) -> Iterable[Candidate]:
    response = requests.get(
        "https://gutendex.com/books/",
        params={"search": row.get("title", "")},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    for item in data.get("results", [])[:5]:
        authors = ", ".join(author.get("name", "") for author in item.get("authors", []))
        confidence = confidence_for(row, item.get("title", ""), authors)
        if confidence == "reject":
            continue
        formats = item.get("formats", {})
        available_formats = []
        for mime, url in formats.items():
            if "text/plain" in mime:
                available_formats.append("TXT")
            elif "epub" in mime:
                available_formats.append("EPUB")
        if not available_formats:
            continue
        yield Candidate(
            source="Project Gutenberg",
            confidence=confidence,
            title=item.get("title", ""),
            author=authors,
            language=",".join(item.get("languages", [])),
            formats=sorted(set(available_formats)),
            url=f"https://www.gutenberg.org/ebooks/{item.get('id')}",
            notes="Structured text candidate; prefer EPUB when available.",
        )


def query_librivox(row: dict) -> Iterable[Candidate]:
    response = requests.get(
        "https://librivox.org/api/feed/audiobooks",
        params={"title": row.get("title", ""), "format": "json"},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    for item in data.get("books", [])[:5]:
        authors = ", ".join(author.get("last_name", "") for author in item.get("authors", []))
        confidence = confidence_for(row, item.get("title", ""), authors)
        if confidence == "reject":
            continue
        yield Candidate(
            source="LibriVox",
            confidence=confidence,
            title=item.get("title", ""),
            author=authors,
            language=item.get("language", ""),
            formats=["AUDIO"],
            url=item.get("url_librivox", ""),
            notes="Audio candidate; segment alignment still needs admin review.",
        )


def find_candidates(row: dict) -> list[Candidate]:
    candidates: list[Candidate] = []
    for query in (query_gutenberg, query_librivox):
        try:
            candidates.extend(query(row))
        except requests.RequestException as exc:
            candidates.append(
                Candidate(
                    source=query.__name__.replace("query_", ""),
                    confidence="error",
                    title=row.get("title", ""),
                    author=row.get("author", ""),
                    language=row.get("language", ""),
                    formats=[],
                    url="",
                    notes=str(exc),
                )
            )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=Path("goodreads_subset_for_import.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("content_candidates.json"))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    rows = load_catalog(args.input_csv, args.limit)
    output = []
    for row in rows:
        candidates = find_candidates(row)
        output.append(
            {
                "goodreads_id": row.get("goodreads_id"),
                "title": row.get("title"),
                "author": row.get("author"),
                "language": row.get("language"),
                "candidates": [asdict(candidate) for candidate in candidates],
            }
        )

    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(output)} review rows to {args.output_json}")


if __name__ == "__main__":
    main()
