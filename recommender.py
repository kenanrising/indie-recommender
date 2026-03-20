"""
recommender.py
────────────────────────────────────────────────────────────────────────────────
Core recommendation logic.

Loads pre-built embeddings from disk once and exposes get_similar_tracks().
Designed to be imported by the FastAPI app (api.py) so embeddings are loaded
once at process start, not on every request.

Similarity algorithm: cosine similarity via normalised dot product.
  - All embedding vectors are L2-normalised at load time.
  - Similarity = dot(query_vec, all_vecs.T), giving values in [-1, 1].
  - This is equivalent to sklearn's cosine_similarity but ~10× faster at scale
    because we avoid per-request normalisation.

Scalability note (for future engineer):
  - Up to ~50 k tracks: this in-memory NumPy approach is perfectly fine.
  - Beyond that: replace the numpy similarity block with a vector database
    (Pinecone, Weaviate, Qdrant, pgvector) using the same 512-dim embeddings.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# ── Module-level state (loaded once on import) ────────────────────────────────
_embeddings: np.ndarray | None = None       # (N, 512) float32, L2-normalised
_track_index: pd.DataFrame | None = None    # maps row index ↔ track metadata
_track_id_to_row: dict[str, int] | None = None  # O(1) lookup: track_id → row


def _load_state(embeddings_dir: str = "embeddings/") -> None:
    """
    Load embeddings.npy and track_index.csv from disk.
    Called automatically on first use of get_similar_tracks().
    """
    global _embeddings, _track_index, _track_id_to_row

    emb_path   = Path(embeddings_dir) / "embeddings.npy"
    index_path = Path(embeddings_dir) / "track_index.csv"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {emb_path}\n"
            "Run  python scripts/build_embeddings.py  first."
        )
    if not index_path.exists():
        raise FileNotFoundError(f"Track index not found: {index_path}")

    raw = np.load(emb_path, allow_pickle=False)           # (N, 512) float32

    # L2-normalise once so cosine similarity reduces to a fast dot product
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)              # guard zero-vectors
    _embeddings = (raw / norms).astype(np.float32)

    _track_index = pd.read_csv(index_path)

    # Build a fast reverse-lookup dictionary
    _track_id_to_row = {
        str(tid): int(idx)
        for idx, tid in enumerate(_track_index["track_id"])
    }

    print(f"[recommender] Loaded {len(_embeddings)} track embeddings from {emb_path}")


def get_similar_tracks(
    track_id: str,
    k: int = 10,
    embeddings_dir: str = "embeddings/",
) -> List[Dict[str, Any]]:
    """
    Return the top-k most similar tracks to `track_id` based on deep audio
    embeddings (cosine similarity).

    Parameters
    ----------
    track_id      : The ID of the seed track (must exist in track_index.csv).
    k             : Number of results to return (excluding the seed track).
    embeddings_dir: Directory containing embeddings.npy and track_index.csv.

    Returns
    -------
    A list of dicts, sorted by similarity descending:
        [
            {"track_id": "TRK042", "title": "...", "artist": "...", "score": 0.94},
            ...
        ]

    Raises
    ------
    ValueError  if track_id is not found in the index.
    RuntimeError if embeddings have not been built yet.
    """
    # Lazy-load embeddings on first call
    if _embeddings is None:
        _load_state(embeddings_dir)

    # ── 1. Look up the query embedding ────────────────────────────────────────
    track_id = str(track_id)
    if track_id not in _track_id_to_row:
        raise ValueError(f"track_id '{track_id}' not found. "
                         "Make sure it exists in the embeddings index.")

    query_row = _track_id_to_row[track_id]
    query_vec = _embeddings[query_row]                    # (512,) already normalised

    # ── 2. Cosine similarity against all tracks (dot product of normed vecs) ──
    # Shape: (N,) — one similarity score per track
    similarities = _embeddings @ query_vec

    # ── 3. Exclude the seed track and sort ────────────────────────────────────
    similarities[query_row] = -2.0                        # sentinel: exclude self

    # Efficiently get top-k indices (unsorted), then sort only those k values
    top_k_indices = np.argpartition(similarities, -k)[-k:]
    top_k_sorted  = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]

    # ── 4. Build the result list ───────────────────────────────────────────────
    results = []
    for idx in top_k_sorted:
        row = _track_index.iloc[int(idx)]
        results.append({
            "track_id": str(row["track_id"]),
            "title":    str(row.get("title",  "")),
            "artist":   str(row.get("artist", "")),
            "score":    float(round(float(similarities[idx]), 6)),
        })

    return results


# ── Convenience: pre-warm the cache (optional, call from API startup) ─────────
def preload(embeddings_dir: str = "embeddings/") -> None:
    """Call this at API startup to load embeddings before the first request."""
    _load_state(embeddings_dir)
