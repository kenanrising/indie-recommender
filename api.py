"""
api.py
────────────────────────────────────────────────────────────────────────────────
FastAPI application — exposes /recommend_deep for Adalo Custom Actions.

Endpoints
─────────
POST /recommend_deep
    Body : { "track_id": "TRK001", "k": 10 }
    200  : { "results": [ {"track_id": ..., "title": ..., "artist": ..., "score": ...}, ... ] }
    404  : { "error": "track_id 'XYZ' not found" }
    422  : Pydantic validation error (bad input types)

GET  /health
    Simple liveness check — returns {"status": "ok", "tracks_loaded": N}

────────────────────────────────────────────────────────────────────────────────
Adalo Custom Action setup (copy these values into Adalo):
  Endpoint URL : https://<your-render-url>/recommend_deep
  Method       : POST
  Headers      : Content-Type: application/json
  Body (Input) :
    {
      "track_id": "<Magic Text — current song ID>",
      "k": 10
    }
  Response mapping:
    results[].track_id  → bind to "Recommended Track ID"
    results[].title     → bind to "Track Title"
    results[].artist    → bind to "Artist Name"
    results[].score     → bind to "Similarity Score"
────────────────────────────────────────────────────────────────────────────────
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import recommender

# ── Configuration ─────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "embeddings/")

# ── Pydantic models ───────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    track_id: str = Field(..., description="ID of the seed track to find similar songs for")
    k: Optional[int] = Field(10, ge=1, le=100, description="Number of recommendations (1–100)")


class TrackResult(BaseModel):
    track_id: str
    title:    str
    artist:   str
    score:    float   # cosine similarity in [-1, 1], higher = more similar


class RecommendResponse(BaseModel):
    results: List[TrackResult]


class HealthResponse(BaseModel):
    status: str
    tracks_loaded: int


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Try to pre-warm embeddings — if not built yet, start anyway."""
    try:
        print(f"[startup] Loading embeddings from '{EMBEDDINGS_DIR}'...")
        recommender.preload(EMBEDDINGS_DIR)
        print("[startup] Embeddings ready.")
    except FileNotFoundError:
        print("[startup] No embeddings found — API starting without them.")
        print("[startup] /recommend_deep will return 503 until embeddings are built.")
    yield


app = FastAPI(
    title="Indie Music Recommender API",
    description="Deep audio embedding recommendations for independent artists.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, summary="Liveness check")
def health() -> HealthResponse:
    """Returns 200 when the API is up and embeddings are loaded."""
    n = len(recommender._embeddings) if recommender._embeddings is not None else 0
    return HealthResponse(status="ok", tracks_loaded=n)


@app.post(
    "/recommend_deep",
    response_model=RecommendResponse,
    summary="Get similar tracks by deep audio embeddings",
)
def recommend_deep(req: RecommendRequest) -> RecommendResponse:
    """
    Given a track_id, returns the top-k most similar tracks ranked by cosine
    similarity of their CLAP audio embeddings.

    **Sample request:**
    ```json
    {
      "track_id": "TRK001",
      "k": 5
    }
    ```

    **Sample response:**
    ```json
    {
      "results": [
        {"track_id": "TRK004", "title": "Low Signal",      "artist": "Mara Blue",          "score": 0.912},
        {"track_id": "TRK002", "title": "Static Dream",    "artist": "Neon Wolves",         "score": 0.887},
        {"track_id": "TRK003", "title": "Coastal Drive",   "artist": "The Paperbacks",      "score": 0.841},
        {"track_id": "TRK005", "title": "Velvet Underground", "artist": "Cass & The Tide",  "score": 0.798},
        {"track_id": "TRK006", "title": "Glass Hours",     "artist": "Wren Valley",         "score": 0.763}
      ]
    }
    ```
    """
    try:
        raw_results = recommender.get_similar_tracks(
            track_id=req.track_id,
            k=req.k,
            embeddings_dir=EMBEDDINGS_DIR,
        )
    except ValueError as exc:
        # track_id not found in the index
        raise HTTPException(status_code=404, detail=str(exc))
    except (FileNotFoundError, RuntimeError) as exc:
        # Embeddings not built yet, or other startup issue
        raise HTTPException(status_code=503, detail=str(exc))

    results = [TrackResult(**r) for r in raw_results]
    return RecommendResponse(results=results)


# ── Local dev entry-point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
