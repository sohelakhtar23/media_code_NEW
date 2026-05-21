"""
Two targeted feature extractors:
  1. CLIP zero-shot: cosine similarity between frames and memorability prompts
  2. STT semantic: CLIP text embedding of first ~60s of transcript

Usage:
    python extract_targeted.py \
        --csv    devset_videolist_GT.csv \
        --frames frames/ \
        --stt    devset-stt/ \
        --out    features/

Output:
    features/clip_zeroshot.npy   (N, n_prompts*3) — mean/max/std per prompt
    features/stt_semantic.npy    (N, 768)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from tqdm import tqdm

MAX_FRAMES = 60

# Prompts designed to capture memorability-relevant visual properties
PROMPTS = [
    # high memorability
    "a memorable and engaging advertisement",
    "a clear and confident presenter speaking to camera",
    "a visually striking and professional video",
    "a bold brand identity with strong visuals",
    # low memorability
    "a boring and generic corporate video",
    "a low quality or amateur video",
    "a confusing or cluttered video",
    # brand salience
    "a video with a prominent company logo",
    "a branded financial services advertisement",
]

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",    required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--stt",    required=True)
    p.add_argument("--out",    default="features")
    return p.parse_args()


def extract_zeroshot(df, frames_root, model, preprocess, device):
    # pre-encode all prompts once
    tokens    = clip.tokenize(PROMPTS).to(device)
    with torch.no_grad():
        prompt_embs = model.encode_text(tokens).float()  # (n_prompts, 768)
        prompt_embs = prompt_embs / prompt_embs.norm(dim=-1, keepdim=True)

    n_prompts = len(PROMPTS)
    all_feats = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="zero-shot visual"):
        frame_dir  = Path(frames_root) / row["id"]
        frame_files = sorted(frame_dir.iterdir())[:MAX_FRAMES] if frame_dir.exists() else []

        if not frame_files:
            all_feats.append(np.zeros(n_prompts * 3))
            continue

        imgs = []
        for fp in frame_files:
            try:
                imgs.append(preprocess(Image.open(fp).convert("RGB")))
            except Exception:
                pass

        if not imgs:
            all_feats.append(np.zeros(n_prompts * 3))
            continue

        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            frame_embs = model.encode_image(batch).float()           # (T, 768)
            frame_embs = frame_embs / frame_embs.norm(dim=-1, keepdim=True)

        # cosine similarity: (T, n_prompts)
        sims = (frame_embs @ prompt_embs.T).cpu().numpy()

        # aggregate per prompt: mean, max, std → (n_prompts*3,)
        feat = np.concatenate([
            sims.mean(axis=0),
            sims.max(axis=0),
            sims.std(axis=0),
        ])
        all_feats.append(feat)

    return np.array(all_feats)


def extract_stt_semantic(df, stt_dir, model, device, max_words=200):
    """Encode the first max_words of each transcript with CLIP text encoder."""
    stt_path  = Path(stt_dir)
    all_feats = []

    texts = []
    for _, row in df.iterrows():
        fpath = stt_path / f"{row['id']}.txt"
        if fpath.exists():
            words = fpath.read_text(encoding="utf-8", errors="ignore").split()
            txt   = " ".join(words[:max_words])
        else:
            txt = ""
        texts.append(txt if txt.strip() else "no speech")

    # CLIP tokenize truncates at 77 tokens automatically
    batch_size = 64
    all_embs   = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(device)
        with torch.no_grad():
            embs = model.encode_text(tokens).float().cpu().numpy()
        all_embs.append(embs)

    return np.concatenate(all_embs, axis=0)


def main():
    args = get_args()
    out  = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df   = pd.read_csv(args.csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    # 1. CLIP zero-shot
    zs = extract_zeroshot(df, args.frames, model, preprocess, device)
    print(f"Zero-shot feature shape: {zs.shape}")
    np.save(out / "clip_zeroshot.npy", zs)

    # 2. STT semantic
    stt = extract_stt_semantic(df, args.stt, model, device)
    print(f"STT semantic feature shape: {stt.shape}")
    np.save(out / "stt_semantic.npy", stt)

    print(f"Saved to {out}/")

if __name__ == "__main__":
    main()
