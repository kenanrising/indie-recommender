"""
Indie Music Recommender API
"""
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import recommender

EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "embeddings/")

app = FastAPI(title="Indie Music Recommender API", version="1.0.0")

# Try loading embeddings at startup — but don't crash if they don't exist yet
try:
    recommender.preload(EMBEDDINGS_DIR)
except FileNotFoundError:
    print("[startup] No embeddings file yet — API running, /recommend_deep returns 503 until embeddings are built.")


class RecommendRequest(BaseModel):
    track_id: str
    k: Optional[int] = Field(10, ge=1, le=100)

class TrackResult(BaseModel):
    track_id: str
    title: str
    artist: str
    score: float

class RecommendResponse(BaseModel):
    results: List[TrackResult]


@app.get("/health")
def health():
    n = len(recommender._embeddings) if recommender._embeddings is not None else 0
    return {"status": "ok", "tracks_loaded": n}


@app.post("/recommend_deep", response_model=RecommendResponse)
def recommend_deep(req: RecommendRequest):
    try:
        raw = recommender.get_similar_tracks(req.track_id, req.k, EMBEDDINGS_DIR)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return RecommendResponse(results=[TrackResult(**r) for r in raw])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
