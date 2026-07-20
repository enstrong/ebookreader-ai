#!/usr/bin/env python3
"""Build a compact Goodreads book-to-work map for recommendation serving."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


def build_map(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with gzip.open(source, "rt", encoding="utf-8") as input_file, output.open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["goodreads_book_id", "work_id", "language"])
        writer.writeheader()
        for line in input_file:
            item = json.loads(line)
            book_id = str(item.get("book_id") or "").strip()
            work_id = str(item.get("work_id") or "").strip()
            if not book_id or not work_id:
                continue
            writer.writerow(
                {
                    "goodreads_book_id": book_id,
                    "work_id": work_id,
                    "language": str(item.get("language_code") or "").strip().lower(),
                }
            )
            rows += 1
            if rows % 250_000 == 0:
                print(f"wrote {rows:,} rows", flush=True)
    print(f"wrote {rows:,} rows to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/goodreads/goodreads_books.json.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/recommendations/hybrid/goodreads_work_map.csv"))
    args = parser.parse_args()
    build_map(args.source, args.output)


if __name__ == "__main__":
    main()
