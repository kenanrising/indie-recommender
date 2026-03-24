"""
build_embeddings_url.py
────────────────────────────────────────────────────────────────────────────
Same as build_embeddings.py but works with audio URLs instead of local files.
Downloads each 30-sec preview, embeds it with CLAP, saves embeddings.npy.
────────────────────────────────────────────────────────────────────────────
"""
import os, sys, warnings, tempfile, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

SAMPLE_RATE   = 48_000
CLIP_DURATION = 10.0
MODEL_ID      = "laion/clap-htsat-unfused"

def download_audio(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = r.read()
    suffix = ".m4a" if "m4a" in url else ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        return f.name

def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=CLIP_DURATION)
    target = int(SAMPLE_RATE * CLIP_DURATION)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    return audio.astype(np.float32)

def build(csv_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"[INFO] {len(df)} tracks to embed")

    print(f"[INFO] Loading CLAP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model     = ClapModel.from_pretrained(MODEL_ID).to(device)
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model.eval()
    print(f"[INFO] Model ready on {device}")

    embeddings, valid_rows = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):
        url = row["audio_file_path"]
        tmp = None
        try:
            tmp   = download_audio(url)
            audio = load_audio(tmp)
            inputs = processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
            with torch.no_grad():
                audio_out = model.audio_model(**inputs)
                emb = model.audio_projection(audio_out.pooler_output)  # (1, 512)
            embeddings.append(emb.squeeze().cpu().numpy())
            valid_rows.append(row.to_dict())
        except Exception as e:
            warnings.warn(f"[SKIP] {row['track_id']}: {e}")
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    if not embeddings:
        raise RuntimeError("No embeddings produced.")

    matrix = np.array(embeddings, dtype=np.float32)
    np.save(out / "embeddings.npy", matrix)
    pd.DataFrame(valid_rows).reset_index(drop=True).to_csv(out / "track_index.csv", index=False)
    print(f"\n[DONE] {len(embeddings)} embeddings — shape {matrix.shape}")
    print(f"[DONE] Saved to {out}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/tracks.csv")
    p.add_argument("--out", default="embeddings/")
    args = p.parse_args()
    build(args.csv, args.out)
