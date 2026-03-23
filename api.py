"""
Indie Music Recommender API
─────────────────────────────────────────────────────────────────────────────
Endpoints:

  GET  /health
       Returns {"status":"ok","tracks_loaded":N}

  POST /recommend_deep
       Input:  {"track_id": "TRK001", "k": 10}
       Output: {"results": [{"track_id","title","artist","score"}, ...]}
       Use this when the user taps a track already in your catalog.

  POST /recommend_by_song
       Input:  {"song": "Blinding Lights The Weeknd", "k": 10}
       Output: {"results": [...], "matched_title": "...", "matched_artist": "..."}
       Use this when the user types any song name.
       Finds it on Spotify, grabs the 30-sec preview, embeds it with CLAP,
       and returns the most similar indie tracks from your catalog.

─────────────────────────────────────────────────────────────────────────────
Adalo Custom Action for /recommend_by_song:
  Method : POST
  URL    : https://indie-recommender.onrender.com/recommend_by_song
  Body   : {"song": "<Magic Text — Search Input>", "k": 10}
  Map    : results[].track_id / results[].title / results[].artist / results[].score
─────────────────────────────────────────────────────────────────────────────
"""

import io
import os
import tempfile
import urllib.request
from typing import List, Optional

import numpy as np
import torch
import librosa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import ClapModel, ClapProcessor

import recommender
import spotify_search

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "embeddings/")
SAMPLE_RATE    = 48_000
CLIP_DURATION  = 10.0

app = FastAPI(title="Indie Music Recommender API", version="2.0.0")

# ── Load embeddings at startup (non-fatal if not built yet) ───────────────────
try:
    recommender.preload(EMBEDDINGS_DIR)
except FileNotFoundError:
    print("[startup] No embeddings yet — /recommend_deep and /recommend_by_song "
          "will return 503 until embeddings are built.")

# ── Lazy-load CLAP model for on-the-fly embedding of Spotify previews ─────────
_clap_model     = None
_clap_processor = None

def get_clap():
    global _clap_model, _clap_processor
    if _clap_model is None:
        print("[clap] Loading CLAP model for live inference...")
        _clap_model     = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        _clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        _clap_model.eval()
        print("[clap] CLAP model ready.")
    return _clap_model, _clap_processor


# ── Helper: embed a raw audio array on the fly ────────────────────────────────
def embed_audio(audio: np.ndarray) -> np.ndarray:
    model, processor = get_clap()
    inputs = processor(audios=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_audio_features(**inputs)
    vec = emb.squeeze().cpu().numpy().astype(np.float32)
    vec /= np.linalg.norm(vec) or 1.0   # L2 normalise
    return vec


# ── Helper: download & decode a preview URL into a numpy array ────────────────
def download_preview(url: str) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=15) as r:
        audio_bytes = r.read()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True, duration=CLIP_DURATION)
    os.unlink(tmp_path)
    target = int(SAMPLE_RATE * CLIP_DURATION)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    return audio.astype(np.float32)


# ── Pydantic models ───────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    track_id: str
    k: Optional[int] = Field(10, ge=1, le=100)

class SongRequest(BaseModel):
    song: str = Field(..., description="Song name and/or artist, e.g. 'Blinding Lights The Weeknd'")
    k: Optional[int] = Field(10, ge=1, le=100)

class TrackResult(BaseModel):
    track_id: str
    title:    str
    artist:   str
    score:    float

class RecommendResponse(BaseModel):
    results: List[TrackResult]

class SongRecommendResponse(BaseModel):
    matched_title:  str
    matched_artist: str
    results: List[TrackResult]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    n = len(recommender._embeddings) if recommender._embeddings is not None else 0
    return {"status": "ok", "tracks_loaded": n}


@app.post("/recommend_deep", response_model=RecommendResponse)
def recommend_deep(req: RecommendRequest):
    """Recommend by catalog track_id (used internally / for Adalo browse screens)."""
    try:
        raw = recommender.get_similar_tracks(req.track_id, req.k, EMBEDDINGS_DIR)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return RecommendResponse(results=[TrackResult(**r) for r in raw])


@app.post("/recommend_by_song", response_model=SongRecommendResponse)
def recommend_by_song(req: SongRequest):
    """
    User types any song name → we find it on Spotify, analyze the preview,
    and return the most similar indie tracks from your catalog.

    Sample request:
        {"song": "Blinding Lights The Weeknd", "k": 10}

    Sample response:
        {
          "matched_title":  "Blinding Lights",
          "matched_artist": "The Weeknd",
          "results": [
            {"track_id": "TRK003", "title": "Coastal Drive", "artist": "The Paperbacks", "score": 0.87},
            ...
          ]
        }
    """
    # Step 1: check embeddings exist
    if recommender._embeddings is None:
        try:
            recommender.preload(EMBEDDINGS_DIR)
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="No catalog embeddings built yet. Add tracks and run the embedding pipeline first."
            )

    # Step 2: find song on Spotify and get preview URL
    try:
        spotify_result = spotify_search.find_preview(req.song)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Step 3: download the 30-sec preview
    try:
        audio = download_preview(spotify_result["preview_url"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not download preview: {e}")

    # Step 4: embed the preview with CLAP
    try:
        query_vec = embed_audio(audio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # Step 5: cosine similarity against the catalog
    sims = recommender._embeddings @ query_vec
    k    = min(req.k, len(sims))
    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    results = []
    for idx in top_idx:
        row = recommender._track_index.iloc[int(idx)]
        results.append(TrackResult(
            track_id=str(row["track_id"]),
            title=str(row.get("title", "")),
            artist=str(row.get("artist", "")),
            score=float(round(float(sims[idx]), 6)),
        ))

    return SongRecommendResponse(
        matched_title=spotify_result["title"],
        matched_artist=spotify_result["artist"],
        results=results,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
