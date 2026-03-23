"""
build_embeddings.py
────────────────────────────────────────────────────────────────────────────────
Offline pipeline: reads tracks CSV → computes CLAP audio embeddings → saves
embeddings.npy + track_index.csv to the output directory.

Uses HuggingFace Transformers' built-in CLAP implementation (laion/clap-htsat-unfused)
— no numpy pinning conflicts, installs cleanly on Python 3.11.

Run once (or whenever your catalog changes):
    python scripts/build_embeddings.py --csv data/tracks.csv --out embeddings/
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID      = "laion/clap-htsat-unfused"   # HuggingFace CLAP — no numpy conflict
SAMPLE_RATE   = 48_000                        # CLAP expects 48 kHz
CLIP_DURATION = 10.0                          # seconds to sample per track
EMBEDDING_DIM = 512                           # CLAP audio embedding size


def load_audio(path: str) -> np.ndarray:
    """Load audio file, resample to 48 kHz mono, clip/pad to CLIP_DURATION seconds."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=CLIP_DURATION)
    target_len = int(SAMPLE_RATE * CLIP_DURATION)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    return audio.astype(np.float32)


def build_embeddings(csv_path: str, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Load catalog ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    required = {"track_id", "title", "artist", "audio_file_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    print(f"[INFO] {len(df)} tracks found in {csv_path}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading CLAP model ({MODEL_ID}) — downloads ~600 MB on first run...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model     = ClapModel.from_pretrained(MODEL_ID).to(device)
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model.eval()
    print(f"[INFO] Model ready on {device}.")

    # ── Embed each track ──────────────────────────────────────────────────────
    embeddings, valid_rows = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):
        path = row["audio_file_path"]
        if not os.path.exists(path):
            warnings.warn(f"[SKIP] Not found: {path}")
            continue
        try:
            audio = load_audio(path)
            inputs = processor(
                audios=audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                emb = model.get_audio_features(**inputs)   # (1, 512)
            embeddings.append(emb.squeeze().cpu().numpy())
            valid_rows.append(row.to_dict())
        except Exception as e:
            warnings.warn(f"[ERROR] track_id={row['track_id']}: {e}")

    if not embeddings:
        raise RuntimeError("No embeddings produced — check audio file paths.")

    # ── Save artefacts ────────────────────────────────────────────────────────
    emb_matrix = np.array(embeddings, dtype=np.float32)
    np.save(out_path / "embeddings.npy", emb_matrix)
    pd.DataFrame(valid_rows).reset_index(drop=True).to_csv(
        out_path / "track_index.csv", index=False
    )
    print(f"\n[DONE] {len(embeddings)} embeddings saved → {out_path}")
    print(f"[DONE] Matrix shape: {emb_matrix.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/tracks_sample.csv")
    parser.add_argument("--out", default="embeddings/")
    args = parser.parse_args()
    build_embeddings(args.csv, args.out)
