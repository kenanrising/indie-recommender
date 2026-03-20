"""
build_embeddings.py
────────────────────────────────────────────────────────────────────────────────
PHASE 1 — Offline embedding pipeline.

Run this script once (or whenever your track catalog changes) to:
  1. Load the LAION-CLAP music checkpoint (downloads automatically on first run).
  2. Iterate every row in your tracks CSV.
  3. Load each audio file, resample to 48 kHz mono, and slice a 10-second window.
  4. Produce a 512-dimensional embedding vector per track.
  5. Save two artefacts to ./embeddings/:
       - embeddings.npy   — float32 NumPy matrix  (N_tracks × 512)
       - track_index.csv  — maps row index → track_id, title, artist

Usage:
    python scripts/build_embeddings.py \
        --csv data/tracks_sample.csv \
        --out embeddings/

Typical runtime: ~0.5–2 sec per track on CPU.  Use a GPU machine for >5 k tracks.
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import warnings
from pathlib import Path

import laion_clap
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48_000          # CLAP was trained at 48 kHz
CLIP_DURATION = 10.0          # Seconds to sample from each track
EMBEDDING_DIM = 512           # CLAP audio embedding dimension

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_and_preprocess(path: str, sr: int = SAMPLE_RATE, duration: float = CLIP_DURATION) -> np.ndarray:
    """
    Load an audio file, resample to `sr`, convert to mono, and take the first
    `duration` seconds.  Returns a 1-D float32 numpy array shaped (sr*duration,).

    The CLAP model internally normalises the waveform, so we just need clean
    float32 PCM at 48 kHz.
    """
    # librosa returns float32 in [-1, 1], resampled to target sr, mixed to mono
    audio, _ = librosa.load(path, sr=sr, mono=True, duration=duration)

    # Pad with silence if the file is shorter than the clip window
    target_len = int(sr * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))

    return audio.astype(np.float32)


def build_embeddings(csv_path: str, out_dir: str) -> None:
    """
    Main pipeline: read CSV → embed each track → persist artefacts.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Load track metadata ────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    required_cols = {"track_id", "title", "artist", "audio_file_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    print(f"[INFO] Found {len(df)} tracks in {csv_path}")

    # ── 2. Load LAION-CLAP model ──────────────────────────────────────────────
    # 'music_audioset_epoch_15_esc_90.14.pt' is the music-optimised checkpoint.
    # It downloads ~600 MB on first run and is cached in ~/.cache/laion_clap/.
    print("[INFO] Loading CLAP model (downloads on first run, ~600 MB)...")
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt(ckpt="music_audioset_epoch_15_esc_90.14.pt")
    model.eval()
    print("[INFO] Model loaded.")

    # ── 3. Compute embeddings ─────────────────────────────────────────────────
    embeddings = []
    valid_rows = []   # Keep only rows we successfully embedded

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding tracks"):
        audio_path = row["audio_file_path"]

        # Skip missing files with a warning rather than crashing
        if not os.path.exists(audio_path):
            warnings.warn(f"[SKIP] File not found: {audio_path}  (track_id={row['track_id']})")
            continue

        try:
            audio = load_and_preprocess(audio_path)

            # CLAP expects shape (1, T) — batch of 1 audio clip
            audio_batch = audio.reshape(1, -1)

            # get_audio_embedding_from_data returns numpy (1, 512)
            emb = model.get_audio_embedding_from_data(x=audio_batch, use_tensor=False)
            embeddings.append(emb[0])        # shape (512,)
            valid_rows.append(row.to_dict())

        except Exception as exc:
            warnings.warn(f"[ERROR] Failed to embed track_id={row['track_id']}: {exc}")

    if not embeddings:
        raise RuntimeError("No embeddings were produced — check your audio file paths.")

    # ── 4. Persist artefacts ──────────────────────────────────────────────────
    emb_matrix = np.array(embeddings, dtype=np.float32)   # (N, 512)
    index_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    # Save the raw matrix as a .npy file — fast to load at API startup
    emb_path = out_path / "embeddings.npy"
    np.save(emb_path, emb_matrix)

    # Save the index so the API can map row numbers back to track metadata
    index_path = out_path / "track_index.csv"
    index_df.to_csv(index_path, index=False)

    print(f"\n[DONE] Saved {len(embeddings)} embeddings → {emb_path}")
    print(f"[DONE] Saved track index          → {index_path}")
    print(f"[DONE] Embedding matrix shape: {emb_matrix.shape}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CLAP audio embeddings for a music catalog.")
    parser.add_argument("--csv", default="data/tracks_sample.csv", help="Path to tracks CSV")
    parser.add_argument("--out", default="embeddings/",            help="Output directory")
    args = parser.parse_args()

    build_embeddings(csv_path=args.csv, out_dir=args.out)
