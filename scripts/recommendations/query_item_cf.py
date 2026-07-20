#!/usr/bin/env python3
"""Query Level 2 item-item collaborative filtering artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--similarities", type=Path, default=Path("data/recommendations/item_cf_similar.csv"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--book-id", type=int, help="UCSD compact CSV book ID")
    group.add_argument("--goodreads-book-id", type=int, help="Real Goodreads book ID")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.similarities)
    if args.book_id is not None:
        matches = frame[frame["book_id"] == args.book_id]
        label = f"book_id={args.book_id}"
    else:
        matches = frame[frame["goodreads_book_id"] == args.goodreads_book_id]
        label = f"goodreads_book_id={args.goodreads_book_id}"

    if matches.empty:
        raise SystemExit(f"No Level 2 neighbors found for {label}. It may not be in the candidate set.")

    matches = matches.sort_values(["score", "co_likes"], ascending=False).head(args.limit)
    columns = ["similar_goodreads_book_id", "similar_book_id", "score", "co_likes", "similar_likes"]
    print(matches[columns].to_string(index=False))


if __name__ == "__main__":
    main()
