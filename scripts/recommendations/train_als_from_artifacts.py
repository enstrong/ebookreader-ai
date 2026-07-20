#!/usr/bin/env python3
"""Train an implicit ALS model from an existing user-item matrix artifact."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

from implicit.als import AlternatingLeastSquares
from scipy import sparse


def link_or_copy(source: Path, destination: Path) -> str:
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factors", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--regularization", type=float, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_matrix_path = args.source_dir / "user_items.npz"
    source_mappings_path = args.source_dir / "mappings.pkl"
    source_metadata_path = args.source_dir / "metadata.json"
    if not source_matrix_path.exists():
        raise SystemExit(f"Missing source matrix: {source_matrix_path}")
    if not source_mappings_path.exists():
        raise SystemExit(f"Missing source mappings: {source_mappings_path}")
    if not source_metadata_path.exists():
        raise SystemExit(f"Missing source metadata: {source_metadata_path}")

    print(f"Loading matrix: {source_matrix_path}", flush=True)
    user_items = sparse.load_npz(source_matrix_path).tocsr()
    print(
        f"User-item matrix: shape={user_items.shape}, nonzeros={user_items.nnz:,}, "
        f"density={user_items.nnz / (user_items.shape[0] * user_items.shape[1]):.8f}",
        flush=True,
    )

    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        random_state=args.random_state,
    )
    model.fit(user_items, show_progress=True)

    model_path = args.output_dir / "als_model.pkl"
    mappings_path = args.output_dir / "mappings.pkl"
    matrix_path = args.output_dir / "user_items.npz"
    metadata_path = args.output_dir / "metadata.json"

    with model_path.open("wb") as file:
        pickle.dump(model, file)
    mappings_storage = link_or_copy(source_mappings_path, mappings_path)
    matrix_storage = link_or_copy(source_matrix_path, matrix_path)

    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    metadata = {
        **source_metadata,
        "method": "implicit_als_matrix_factorization",
        "source_artifact_dir": str(args.source_dir),
        "factors": args.factors,
        "iterations": args.iterations,
        "regularization": args.regularization,
        "random_state": args.random_state,
        "user_factors_shape": list(model.user_factors.shape),
        "item_factors_shape": list(model.item_factors.shape),
        "user_items_shape": list(user_items.shape),
        "user_items_nonzeros": int(user_items.nnz),
        "user_items_positive_nonzeros": int((user_items.data > 0).sum()),
        "user_items_negative_nonzeros": int((user_items.data < 0).sum()),
        "user_items_density": user_items.nnz / (user_items.shape[0] * user_items.shape[1]),
        "artifacts": {
            "model": str(model_path),
            "mappings": str(mappings_path),
            "user_items": str(matrix_path),
        },
        "artifact_storage": {
            "mappings": mappings_storage,
            "user_items": matrix_storage,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Saved ALS model: {model_path}", flush=True)
    print(f"Saved mappings: {mappings_path} ({mappings_storage})", flush=True)
    print(f"Saved user-item matrix: {matrix_path} ({matrix_storage})", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)
    print(f"Learned user matrix shape: {model.user_factors.shape}", flush=True)
    print(f"Learned item matrix shape: {model.item_factors.shape}", flush=True)


if __name__ == "__main__":
    main()
