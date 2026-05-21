"""
Extract CLIP visual features from the first 60 frames of each video.
Saves a (N, D) numpy array per video, then aggregates to a single feature matrix.

Usage:
    python extract_visual.py \
        --csv  devset_videolist_GT.csv \
        --frames frames/ \
        --out  features/

Output:
    features/visual_features.npy   shape (339, 768*4) — mean/std/first/last pooling
    features/video_ids.npy         matching video id order
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from tqdm import tqdm

MAX_FRAMES = 60

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",    required=True)
    p.add_argument("--frames", required=True, help="Root dir containing {id}/ subdirs")
    p.add_argument("--out",    default="features")
    return p.parse_args()

def pool_embeddings(embs: np.ndarray) -> np.ndarray:
    """embs: (T, D) → (4*D,) via mean, std, first, last"""
    return np.concatenate([
        embs.mean(axis=0),
        embs.std(axis=0),
        embs[0],
        embs[-1],
    ])

def main():
    args = get_args()
    out  = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df   = pd.read_csv(args.csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    feat_dim  = 768  # ViT-L/14 embedding dim
    all_feats = []
    all_ids   = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        vid_id    = row["id"]
        frame_dir = Path(args.frames) / vid_id

        if not frame_dir.exists():
            print(f"  [WARN] missing frames dir: {vid_id}")
            all_feats.append(np.zeros(feat_dim * 4))
            all_ids.append(vid_id)
            continue

        # sorted frames → take first MAX_FRAMES
        frame_files = sorted(frame_dir.iterdir())[:MAX_FRAMES]
        if len(frame_files) == 0:
            all_feats.append(np.zeros(feat_dim * 4))
            all_ids.append(vid_id)
            continue

        # batch inference
        imgs = []
        for fp in frame_files:
            try:
                imgs.append(preprocess(Image.open(fp).convert("RGB")))
            except Exception:
                pass

        if len(imgs) == 0:
            all_feats.append(np.zeros(feat_dim * 4))
            all_ids.append(vid_id)
            continue

        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            embs = model.encode_image(batch).float().cpu().numpy()  # (T, 768)
            # images from CPU → GPU → GPU compute → CPU → NumPy/Pandas/etc
            # we move back to CPU, since .numpy() only works on CPU tensors

        all_feats.append(pool_embeddings(embs))
        all_ids.append(vid_id)

    X = np.array(all_feats)  # (N, 4*768)
    print(f"Visual feature matrix shape: {X.shape}")

    np.save(out / "visual_features.npy", X)
    np.save(out / "video_ids.npy",       np.array(all_ids))
    print(f"Saved to {out}/")

if __name__ == "__main__":
    main()
