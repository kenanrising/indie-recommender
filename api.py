"""
Indie Music Recommender API
"""
import io, os, tempfile, urllib.request
from typing import List, Optional

import numpy as np
import torch
import librosa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import ClapModel, ClapProcessor

import recommender
import spotify_search

EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "embeddings/")
SAMPLE_RATE    = 48_000
CLIP_DURATION  = 10.0

app = FastAPI(title="Indie Music Recommender API", version="2.0.0")

# ── Preload everything at startup ─────────────────────────────────────────────
print("[startup] Loading embeddings...")
try:
    recommender.preload(EMBEDDINGS_DIR)
    print(f"[startup] {len(recommender._embeddings)} tracks loaded.")
except FileNotFoundError:
    print("[startup] No embeddings yet.")

print("[startup] Loading CLAP model (this takes ~60s on first boot)...")
_clap_model     = ClapModel.from_pretrained("laion/clap-htsat-unfused")
_clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
_clap_model.eval()
print("[startup] CLAP model ready.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def embed_audio(audio: np.ndarray) -> np.ndarray:
    inputs = _clap_processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        out = _clap_model.audio_model(**inputs)
        emb = _clap_model.audio_projection(out.pooler_output)
    vec = emb.squeeze().cpu().numpy().astype(np.float32)
    vec /= np.linalg.norm(vec) or 1.0
    return vec

def download_preview(url: str) -> np.ndarray:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = r.read()
    suffix = ".m4a" if "m4a" in url else ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data); tmp = f.name
    audio, _ = librosa.load(tmp, sr=SAMPLE_RATE, mono=True, duration=CLIP_DURATION)
    os.unlink(tmp)
    target = int(SAMPLE_RATE * CLIP_DURATION)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    return audio.astype(np.float32)


# ── Pydantic models ───────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    track_id: str
    k: Optional[int] = Field(10, ge=1, le=100)

class SongRequest(BaseModel):
    song: str
    k: Optional[int] = Field(10, ge=1, le=100)

class TrackResult(BaseModel):
    track_id: str
    title: str
    artist: str
    score: float

class RecommendResponse(BaseModel):
    results: List[TrackResult]

class SongRecommendResponse(BaseModel):
    matched_title: str
    matched_artist: str
    results: List[TrackResult]


# ── Routes ────────────────────────────────────────────────────────────────────
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


@app.post("/recommend_by_song", response_model=SongRecommendResponse)
def recommend_by_song(req: SongRequest):
    """
    User types any song → finds it via iTunes → embeds 30s preview with CLAP
    → returns most similar tracks from catalog.

    Sample request:  {"song": "Blinding Lights The Weeknd", "k": 10}
    Sample response: {"matched_title": "Blinding Lights", "matched_artist": "The Weeknd",
                      "results": [{"track_id":..., "title":..., "artist":..., "score":...}]}
    """
    if recommender._embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings not loaded yet.")

    try:
        result = spotify_search.find_preview(req.song)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        audio = download_preview(result["preview_url"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not download preview: {e}")

    query_vec = embed_audio(audio)

    sims    = recommender._embeddings @ query_vec
    k       = min(req.k, len(sims))
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
        matched_title=result["title"],
        matched_artist=result["artist"],
        results=results,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
