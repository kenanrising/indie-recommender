"""
recommender.py
────────────────────────────────────────────────────────────────────────────────
Core similarity logic. Loaded once at API startup.

Scalability note: works in-memory up to ~50k tracks.
Beyond that, swap the numpy dot product for a vector DB (Qdrant, Pinecone)
using the same 512-dim vectors — function signature stays identical.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

_embeddings:       np.ndarray | None = None
_track_index:      pd.DataFrame | None = None
_track_id_to_row:  dict[str, int] | None = None


def _load_state(embeddings_dir: str = "embeddings/") -> None:
    global _embeddings, _track_index, _track_id_to_row

    emb_path   = Path(embeddings_dir) / "embeddings.npy"
    index_path = Path(embeddings_dir) / "track_index.csv"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {emb_path}. "
            "Run: python scripts/build_embeddings.py --csv data/tracks.csv --out embeddings/"
        )

    raw = np.load(emb_path, allow_pickle=False)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    _embeddings = (raw / norms).astype(np.float32)   # L2-normalised

    _track_index = pd.read_csv(index_path)
    _track_id_to_row = {
        str(tid): int(i)
        for i, tid in enumerate(_track_index["track_id"])
    }
    print(f"[recommender] {len(_embeddings)} embeddings loaded.")


def get_similar_tracks(
    track_id: str,
    k: int = 10,
    embeddings_dir: str = "embeddings/",
) -> List[Dict[str, Any]]:
    """
    Return top-k most similar tracks by cosine similarity of CLAP embeddings.
    Raises ValueError if track_id is unknown.
    """
    if _embeddings is None:
        _load_state(embeddings_dir)

    track_id = str(track_id)
    if track_id not in _track_id_to_row:
        raise ValueError(f"track_id '{track_id}' not found in index.")

    qrow = _track_id_to_row[track_id]
    sims = _embeddings @ _embeddings[qrow]   # cosine sim (vectors are normalised)
    sims[qrow] = -2.0                         # exclude self

    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    results = []
    for idx in top_idx:
        row = _track_index.iloc[int(idx)]
        results.append({
            "track_id": str(row["track_id"]),
            "title":    str(row.get("title", "")),
            "artist":   str(row.get("artist", "")),
            "score":    float(round(float(sims[idx]), 6)),
        })
    return results


def preload(embeddings_dir: str = "embeddings/") -> None:
    """Call at API startup to warm the cache before first request."""
    _load_state(embeddings_dir)
