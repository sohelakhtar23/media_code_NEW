"""
Extract text (CLIP) and metadata features.

Usage:
    python extract_text_meta.py \
        --csv  devset_videolist_GT.csv \
        --out  features/

Output:
    features/text_features.npy    CLIP text embeddings (title + description)
    features/meta_features.npy    Numeric metadata features
    features/meta_feature_names.txt
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import clip
from tqdm import tqdm

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="features")
    return p.parse_args()

# ── text helpers ──────────────────────────────────────────────────────────────
def clean_text(s: str, max_words: int = 60) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r'\s+', ' ', s).strip()
    return " ".join(s.split()[:max_words])  # CLIP token limit ~77

def encode_texts(texts, model, device, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(device)
        with torch.no_grad():
            embs = model.encode_text(tokens).float().cpu().numpy()
        all_embs.append(embs)
    return np.concatenate(all_embs, axis=0)

# ── metadata helpers ──────────────────────────────────────────────────────────
def build_meta_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    # target-encode channelName using leave-one-out mean
    ch_mem   = df.groupby("channelName")["memorability_score"].mean()
    ch_brand = df.groupby("channelName")["brand_memorability"].mean()

    feat = pd.DataFrame()
    feat["log_views"]       = np.log1p(df["viewsCount"])
    feat["log_likes"]       = np.log1p(df["likesCount"])
    feat["log_dislikes"]    = np.log1p(df["dislikesCount"])
    feat["log_comments"]    = np.log1p(df["commentsCount"])
    feat["log_engagements"] = np.log1p(df["engagementsCount"])
    feat["engagement_rate"] = df["engagementRate"]
    feat["log_duration"]    = np.log1p(df["durationSeconds"])
    feat["is_long"]         = (df["durationSeconds"] > 60).astype(float)
    feat["nb_annotations"]  = df["nb_annotations"]
    feat["channel_mem_enc"] = df["channelName"].map(ch_mem).fillna(ch_mem.mean())
    feat["channel_brand_enc"] = df["channelName"].map(ch_brand).fillna(ch_brand.mean())

    # category one-hot (only categories appearing ≥ 5 times)
    cat_counts = df["categoryName"].value_counts()
    keep_cats  = cat_counts[cat_counts >= 5].index.tolist()
    for cat in keep_cats:
        feat[f"cat_{cat.replace(' ', '_').replace('&','n')}"] = (df["categoryName"] == cat).astype(float)

    names = feat.columns.tolist()
    return feat.values.astype(np.float32), names

def main():
    args = get_args()
    out  = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df   = pd.read_csv(args.csv)

    # ── CLIP text features ────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, _ = clip.load("ViT-L/14", device=device)
    model.eval()

    # combine title + truncated description
    texts = []
    for _, row in df.iterrows():
        title = clean_text(row.get("title", ""), max_words=20)
        desc  = clean_text(row.get("description", ""), max_words=40)
        texts.append(f"{title} {desc}".strip())

    print("Encoding text with CLIP...")
    text_embs = encode_texts(texts, model, device)  # (N, 768)
    print(f"Text feature matrix shape: {text_embs.shape}")
    np.save(out / "text_features.npy", text_embs)

    # ── Metadata features ─────────────────────────────────────────────────────
    meta, names = build_meta_features(df)
    print(f"Meta feature matrix shape: {meta.shape}")
    print(f"Features: {names}")
    np.save(out / "meta_features.npy", meta)
    Path(out / "meta_feature_names.txt").write_text("\n".join(names))

    print(f"Saved to {out}/")

if __name__ == "__main__":
    main()
