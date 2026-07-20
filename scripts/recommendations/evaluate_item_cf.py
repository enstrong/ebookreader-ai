#!/usr/bin/env python3
"""Evaluate the Level 2 item-item recommender with a leave-one-out test.

For each eligible user:
1. collect books they liked
2. hide one liked book
3. recommend from the remaining liked books
4. check whether the hidden book appears in the top K results
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_similarity_graph(path: Path) -> dict[int, list[tuple[int, float]]]:
    graph: dict[int, list[tuple[int, float]]] = defaultdict(list)
    frame = pd.read_csv(path)
    for row in frame.itertuples(index=False):
        source = int(row.goodreads_book_id)
        target = int(row.similar_goodreads_book_id)
        score = float(row.score)
        graph[source].append((target, score))
    return graph


def graph_book_ids(graph: dict[int, list[tuple[int, float]]]) -> set[int]:
    ids = set(graph)
    for neighbors in graph.values():
        ids.update(target for target, _ in neighbors)
    return ids


def load_user_likes(
    interactions_path: Path,
    candidate_books: set[int],
    min_like_rating: int,
    chunk_size: int,
    max_users: int,
    min_likes: int,
) -> dict[int, set[int]]:
    user_likes: dict[int, set[int]] = defaultdict(set)
    dtypes = {"user_id": "int64", "goodreads_book_id": "int64", "rating": "int8"}
    columns = ["user_id", "goodreads_book_id", "rating"]

    for chunk_number, chunk in enumerate(pd.read_csv(interactions_path, usecols=columns, dtype=dtypes, chunksize=chunk_size), start=1):
        liked = chunk[
            (chunk["rating"] >= min_like_rating)
            & (chunk["goodreads_book_id"].isin(candidate_books))
        ]
        for row in liked.itertuples(index=False):
            user_likes[int(row.user_id)].add(int(row.goodreads_book_id))

        eligible_count = sum(1 for likes in user_likes.values() if len(likes) >= min_likes)
        print(
            f"Loaded chunk {chunk_number}: users={len(user_likes):,}, eligible={eligible_count:,}",
            flush=True,
        )
        if eligible_count >= max_users:
            break

    eligible = {
        user_id: likes
        for user_id, likes in user_likes.items()
        if len(likes) >= min_likes
    }
    return dict(list(eligible.items())[:max_users])


def recommend_from_likes(
    graph: dict[int, list[tuple[int, float]]],
    source_books: set[int],
    blocked_books: set[int],
    limit: int,
) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    evidence: dict[int, int] = defaultdict(int)

    for source in source_books:
        for target, score in graph.get(source, []):
            if target in blocked_books:
                continue
            scores[target] += score
            evidence[target] += 1

    ranked = sorted(scores, key=lambda book_id: (scores[book_id], evidence[book_id]), reverse=True)
    return ranked[:limit]


def reciprocal_rank(recommendations: list[int], hidden_book: int) -> float:
    for index, book_id in enumerate(recommendations, start=1):
        if book_id == hidden_book:
            return 1.0 / index
    return 0.0


def evaluate(
    graph: dict[int, list[tuple[int, float]]],
    user_likes: dict[int, set[int]],
    cutoffs: list[int],
    rng: random.Random,
) -> dict:
    max_k = max(cutoffs)
    recommendable_books = graph_book_ids(graph)
    hits = {k: 0 for k in cutoffs}
    reciprocal_ranks = []
    evaluated = 0
    skipped = 0
    examples = []

    for user_id, likes in user_likes.items():
        hideable_books = sorted(likes.intersection(recommendable_books))
        if not hideable_books:
            skipped += 1
            continue
        hidden_book = rng.choice(hideable_books)
        source_books = set(likes)
        source_books.remove(hidden_book)
        recommendations = recommend_from_likes(graph, source_books, source_books, max_k)
        evaluated += 1

        for k in cutoffs:
            if hidden_book in recommendations[:k]:
                hits[k] += 1
        rr = reciprocal_rank(recommendations, hidden_book)
        reciprocal_ranks.append(rr)

        if len(examples) < 10:
            examples.append(
                {
                    "user_id": user_id,
                    "hidden_goodreads_book_id": hidden_book,
                    "source_books_count": len(source_books),
                    "hit_rank": int(1 / rr) if rr else None,
                    "top_recommendations": recommendations[:10],
                }
            )

    return {
        "users_evaluated": evaluated,
        "users_skipped": skipped,
        "hit_rate": {f"@{k}": hits[k] / evaluated if evaluated else 0.0 for k in cutoffs},
        "mean_reciprocal_rank": sum(reciprocal_ranks) / evaluated if evaluated else 0.0,
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, default=Path("data/recommendations/interactions_filtered.csv"))
    parser.add_argument("--similarities", type=Path, default=Path("data/recommendations/item_cf_similar.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/recommendations/item_cf_evaluation.json"))
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--max-users", type=int, default=10_000)
    parser.add_argument("--min-likes", type=int, default=5)
    parser.add_argument("--min-like-rating", type=int, default=4)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading similarity graph", flush=True)
    graph = load_similarity_graph(args.similarities)
    candidate_books = set(graph.keys())
    print(f"Similarity graph sources: {len(candidate_books):,}", flush=True)

    print("Loading eligible user likes", flush=True)
    user_likes = load_user_likes(
        args.interactions,
        candidate_books,
        args.min_like_rating,
        args.chunk_size,
        args.max_users,
        args.min_likes,
    )
    print(f"Eligible users selected: {len(user_likes):,}", flush=True)

    results = evaluate(graph, user_likes, sorted(args.cutoffs), random.Random(args.seed))
    results.update(
        {
            "interactions": str(args.interactions),
            "similarities": str(args.similarities),
            "max_users": args.max_users,
            "min_likes": args.min_likes,
            "min_like_rating": args.min_like_rating,
            "cutoffs": sorted(args.cutoffs),
            "seed": args.seed,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"Wrote evaluation: {args.output}", flush=True)


if __name__ == "__main__":
    main()
