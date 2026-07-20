#!/usr/bin/env python3
"""Small local HTTP service for the Level 4 hybrid recommendation model.

Run from the repo root:

    .venv/bin/python scripts/recommendations/hybrid_recommendation_service.py

The service intentionally uses only the Python standard library plus the ML
packages already used by the project, so it can run without installing FastAPI.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import scipy.sparse as sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(
    os.environ.get(
        "RECOMMENDATION_MODEL_DIR",
        REPO_ROOT / "data/recommendations/experiments/als_reads_workcanon_20k_f256_i2_lam1p0_validation_split",
    )
)
METADATA_CSV = Path(
    os.environ.get(
        "RECOMMENDATION_METADATA_CSV",
        REPO_ROOT / "data/recommendations/hybrid/book_metadata_workcanon_20k_i5.csv",
    )
)
WORK_MAP_CSV = Path(
    os.environ.get(
        "RECOMMENDATION_WORK_MAP_CSV",
        REPO_ROOT / "data/recommendations/hybrid/goodreads_work_map.csv",
    )
)
HOST = os.environ.get("RECOMMENDATION_HOST", "127.0.0.1")
PORT = int(os.environ.get("RECOMMENDATION_PORT", "8001"))

sys.path.insert(0, str(REPO_ROOT / "scripts/recommendations"))
from evaluate_hybrid_validation_split import (  # noqa: E402
    EXPERIMENTS,
    content_scores_for_candidates,
    load_metadata,
    minmax,
)

CONFIG = EXPERIMENTS["author_50_content_heavy"]


def load_model_artifacts():
    with (MODEL_DIR / "als_model.pkl").open("rb") as file:
        model = pickle.load(file)
    with (MODEL_DIR / "mappings.pkl").open("rb") as file:
        mappings = pickle.load(file)
    metadata, _metadata_stats = load_metadata(METADATA_CSV)
    model_meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    return model, mappings, metadata, model_meta


def load_work_map(path: Path) -> tuple[dict[int, str], dict[str, list[int]]]:
    if not path.exists():
        return {}, {}

    book_to_work: dict[int, str] = {}
    work_to_books: dict[str, list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in pd.read_csv(file, usecols=["goodreads_book_id", "work_id"], chunksize=250_000):
            for book_id, work_id in zip(row["goodreads_book_id"], row["work_id"]):
                try:
                    book_id_int = int(book_id)
                except (TypeError, ValueError):
                    continue
                work_id_str = "" if pd.isna(work_id) else str(work_id).strip()
                if not work_id_str:
                    continue
                book_to_work[book_id_int] = work_id_str
                work_to_books.setdefault(work_id_str, []).append(book_id_int)
    return book_to_work, work_to_books


MODEL, MAPPINGS, METADATA, MODEL_META = load_model_artifacts()
BOOK_TO_IDX = {int(k): int(v) for k, v in MAPPINGS["book_to_idx"].items()}
IDX_TO_BOOK = {int(k): int(v) for k, v in MAPPINGS["idx_to_book"].items()}
N_ITEMS = len(BOOK_TO_IDX)
META_FRAME = pd.read_csv(METADATA_CSV)
BOOK_TO_WORK, WORK_TO_BOOKS = load_work_map(WORK_MAP_CSV)
POPULAR_BOOK_IDS = META_FRAME.sort_values(
    ["ratings_count", "average_rating"],
    ascending=[False, False],
)["goodreads_book_id"].astype(int).tolist()
MODEL_BOOK_IDS_BY_WORK: dict[str, list[int]] = {}
MODEL_BOOK_RATING_COUNT = {
    int(row.goodreads_book_id): int(row.ratings_count or 0)
    for row in META_FRAME.itertuples(index=False)
}
for model_book_id in BOOK_TO_IDX:
    work_id = BOOK_TO_WORK.get(model_book_id)
    if work_id:
        MODEL_BOOK_IDS_BY_WORK.setdefault(work_id, []).append(model_book_id)
for model_book_ids in MODEL_BOOK_IDS_BY_WORK.values():
    model_book_ids.sort(key=lambda book_id: MODEL_BOOK_RATING_COUNT.get(book_id, 0), reverse=True)


def model_book_for(book_id: int) -> int | None:
    if book_id in BOOK_TO_IDX:
        return book_id
    work_id = BOOK_TO_WORK.get(book_id)
    if not work_id:
        return None
    candidates = MODEL_BOOK_IDS_BY_WORK.get(work_id) or []
    return candidates[0] if candidates else None


def explicit_signal(rating: float, user_mean: float) -> float:
    centered = rating - user_mean
    if centered <= 0:
        return 0.0
    alpha = float(MODEL_META.get("alpha", 40.0))
    centered_scale = float(MODEL_META.get("centered_scale", 4.0))
    return float(1.0 + alpha * abs(centered) / centered_scale)


def one_book_five_star_signal() -> float:
    global_mean = float(MODEL_META.get("global_explicit_rating_mean", 3.94))
    shrinkage = float(MODEL_META.get("mean_shrinkage", 5.0))
    user_mean = (5.0 + shrinkage * global_mean) / (1.0 + shrinkage)
    return explicit_signal(5.0, user_mean)


def normalize_interactions(raw: list[dict]) -> tuple[list[tuple[int, float]], set[int], set[str]]:
    ratings = [
        float(item.get("rating") or 0)
        for item in raw
        if float(item.get("rating") or 0) > 0
    ]
    user_mean = sum(ratings) / len(ratings) if ratings else 0.0
    signal_by_book_id: dict[int, float] = {}
    blocked_book_ids: set[int] = set()
    blocked_work_ids: set[str] = set()

    for item in raw:
        try:
            raw_book_id = int(
                item.get("goodreadsBookId")
                or item.get("goodreads_book_id")
                or item.get("goodreadsId")
            )
        except (TypeError, ValueError):
            continue

        work_id = BOOK_TO_WORK.get(raw_book_id)
        if work_id:
            blocked_work_ids.add(work_id)
        book_id = model_book_for(raw_book_id)
        if book_id is None:
            continue
        blocked_book_ids.add(book_id)

        rating = float(item.get("rating") or 0)
        status = str(item.get("status") or "").upper()
        bookmarked = bool(item.get("bookmarked") or False)

        signal = 0.0
        if rating > 0 and user_mean > 0:
            signal = explicit_signal(rating, user_mean)
            if rating >= 4:
                signal = max(signal, 1.0 + (rating - 3.0))
        elif status == "FINISHED":
            signal = 1.0
        elif status == "READING":
            signal = 0.7
        elif bookmarked:
            signal = 0.4

        if signal > 0:
            signal_by_book_id[book_id] = max(signal_by_book_id.get(book_id, 0.0), signal)

    sources = list(signal_by_book_id.items())
    sources.sort(key=lambda pair: pair[1], reverse=True)
    return sources[:50], blocked_book_ids, blocked_work_ids


def fake_user_row(sources: list[tuple[int, float]]) -> sparse.csr_matrix:
    if not sources:
        return sparse.csr_matrix((1, N_ITEMS), dtype=np.float32)
    indices = np.asarray([BOOK_TO_IDX[book_id] for book_id, _ in sources], dtype=np.int32)
    values = np.asarray([signal for _, signal in sources], dtype=np.float32)
    indptr = np.asarray([0, len(indices)], dtype=np.int32)
    return sparse.csr_matrix((values, indices, indptr), shape=(1, N_ITEMS), dtype=np.float32)


def reason_for(candidate_id: int, sources: list[tuple[int, float]]) -> str:
    candidate = METADATA.get(candidate_id)
    if candidate is None or not sources:
        return "Похоже на ваши оценки"
    for source_id, _ in sources[:10]:
        source = METADATA.get(source_id)
        if source is None:
            continue
        if candidate.author_key and set(candidate.author_key.split("|")) & set(source.author_key.split("|")):
            return "Похожий автор"
    if candidate.genres:
        return "Похожий жанр"
    return "Понравилось похожим читателям"


def recommend_from_sources(
    sources: list[tuple[int, float]],
    limit: int,
    candidate_pool: int = 500,
    blocked_book_ids: set[int] | None = None,
    blocked_work_ids: set[str] | None = None,
) -> dict:
    blocked_book_ids = blocked_book_ids or set()
    blocked_work_ids = blocked_work_ids or set()
    if not sources:
        return {
            "recommendations": popular_fallback(limit),
            "sourceCount": 0,
            "model": "popular_fallback",
        }

    user_row = fake_user_row(sources)
    item_indices, als_scores = MODEL.recommend(
        0,
        user_row,
        N=max(candidate_pool + len(blocked_book_ids), limit),
        filter_already_liked_items=True,
        recalculate_user=True,
    )
    item_indices = np.asarray(item_indices, dtype=np.int32)
    als_scores = np.asarray(als_scores, dtype=np.float32)
    content_scores = content_scores_for_candidates(item_indices, sources, IDX_TO_BOOK, METADATA, CONFIG.weights)
    final_scores = CONFIG.alpha * minmax(als_scores) + (1.0 - CONFIG.alpha) * minmax(content_scores)
    order = np.argsort(-final_scores, kind="stable")
    rows = []
    for offset in order:
        item_idx = int(item_indices[offset])
        book_id = int(IDX_TO_BOOK[item_idx])
        if book_id in blocked_book_ids:
            continue
        work_id = BOOK_TO_WORK.get(book_id)
        if work_id and work_id in blocked_work_ids:
            continue
        rows.append(
            {
                "goodreadsBookId": book_id,
                "score": float(final_scores[offset]),
                "alsScore": float(als_scores[offset]),
                "contentScore": float(content_scores[offset]),
                "reason": reason_for(book_id, sources),
            }
        )
        if len(rows) >= limit:
            break
    return {
        "recommendations": rows,
        "sourceCount": len(sources),
        "model": "hybrid_als_metadata",
    }


def popular_fallback(limit: int) -> list[dict]:
    rows = []
    for book_id in POPULAR_BOOK_IDS[:limit]:
        rows.append(
            {
                "goodreadsBookId": int(book_id),
                "score": 0.0,
                "alsScore": 0.0,
                "contentScore": 0.0,
                "reason": "Популярная книга",
            }
        )
    return rows


def similar_for_book(book_id: int, limit: int) -> dict:
    model_book_id = model_book_for(book_id)
    if model_book_id is None:
        return {"similar": [], "sourceCount": 0, "model": "hybrid_als_metadata"}
    blocked_work_ids = {BOOK_TO_WORK[book_id]} if book_id in BOOK_TO_WORK else set()
    result = recommend_from_sources(
        [(model_book_id, one_book_five_star_signal())],
        limit,
        blocked_book_ids={model_book_id},
        blocked_work_ids=blocked_work_ids,
    )
    return {
        "similar": result["recommendations"],
        "sourceCount": 1,
        "model": result["model"],
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        limit = int(params.get("limit", ["20"])[0])
        if parsed.path.startswith("/similar/"):
            try:
                book_id = int(parsed.path.removeprefix("/similar/"))
            except ValueError:
                self.write_json({"similar": []}, status=400)
                return
            self.write_json(similar_for_book(book_id, min(max(limit, 1), 500)))
            return
        if parsed.path == "/health":
            self.write_json(
                {
                    "status": "ok",
                    "model": "hybrid_als_metadata",
                    "modelDir": str(MODEL_DIR),
                    "metadataCsv": str(METADATA_CSV),
                    "iterations": MODEL_META.get("iterations"),
                    "factors": MODEL_META.get("factors"),
                    "regularization": MODEL_META.get("regularization"),
                }
            )
            return
        self.write_json({"message": "Not found"}, status=404)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = json.loads(self.read_request_body().decode("utf-8") or "{}")
        except (json.JSONDecodeError, ValueError):
            self.write_json({"message": "Invalid JSON"}, status=400)
            return

        if parsed.path in {"/recommend", "/preview"}:
            limit = int(payload.get("limit") or 50)
            interactions = payload.get("interactions") or []
            sources, blocked_book_ids, blocked_work_ids = normalize_interactions(interactions)
            self.write_json(
                recommend_from_sources(
                    sources,
                    min(max(limit, 1), 500),
                    blocked_book_ids=blocked_book_ids,
                    blocked_work_ids=blocked_work_ids,
                )
            )
            return
        self.write_json({"message": "Not found"}, status=404)

    def read_request_body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() != "chunked":
            length = int(self.headers.get("Content-Length", "0"))
            return self.rfile.read(length)

        chunks = []
        while True:
            size_line = self.rfile.readline().strip()
            if not size_line:
                continue
            chunk_size = int(size_line.split(b";", 1)[0], 16)
            if chunk_size == 0:
                while self.rfile.readline().strip():
                    pass
                return b"".join(chunks)
            chunks.append(self.rfile.read(chunk_size))
            self.rfile.read(2)

    def log_message(self, format, *args):  # noqa: A003
        return

    def write_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Hybrid recommendation service listening on http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
