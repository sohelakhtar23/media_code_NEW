"""
Apply rank aggregation to LSA, LLM, face_rate individually and in combination.

Usage:
    python train_rank.py \
        --csv  devset_videolist_GT.csv \
        --stt  devset-stt/ \
        --feat features/ \
        --llm  llm_scalar_cache_v2.json
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

LLM_KEYS = [
    "emotional_valence", "human_presence", "message_simplicity", "novelty_surprise",
    "narrative_arc", "brand_prominence", "repetition_hooks", "direct_memorability",
]

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  required=True)
    p.add_argument("--stt",  required=True)
    p.add_argument("--feat", default="features")
    p.add_argument("--llm",  required=True)
    return p.parse_args()


def spearman(a, b):
    return spearmanr(a, b).statistic


def build_folds(df, n_folds=5):
    ch_counts  = df["channelName"].value_counts()
    fold_map   = {}
    fold_sizes = {i: 0 for i in range(n_folds)}
    for ch, cnt in ch_counts.items():
        t = min(range(n_folds), key=lambda f: fold_sizes[f])
        fold_map[ch] = t
        fold_sizes[t] += cnt
    return df["channelName"].map(fold_map).values


def load_transcripts(df, stt_dir):
    stt_dir = Path(stt_dir)
    return [
        (stt_dir / f"{vid}.txt").read_text(encoding="utf-8").strip()
        if (stt_dir / f"{vid}.txt").exists() else ""
        for vid in df["id"]
    ]


def load_llm(df, llm_path):
    cache = json.loads(Path(llm_path).read_text())
    return np.array([
        [float(cache.get(vid, {}).get(k, 5.0)) for k in LLM_KEYS]
        for vid in df["id"]
    ], dtype=np.float32)


def rank_aggregate(X, y, folds, threshold=0.03):
    """Column-wise rank aggregation with train-fold correlation filtering."""
    preds = np.zeros(len(y))
    # for fold_id in range(5):
    for fold_id in np.unique(folds):
        val_mask   = folds == fold_id
        X_tr, X_val = X[~val_mask], X[val_mask]
        y_tr = y[~val_mask]
        scores = np.zeros(val_mask.sum())
        
        for col in range(X_tr.shape[1]):
            r, _ = spearmanr(X_tr[:, col], y_tr)
            if np.isnan(r) or abs(r) < threshold:
                continue
            scores += np.sign(r) * rankdata(X_val[:, col])
        preds[val_mask] = scores
    return preds


def combine_ranks(*rank_arrays):
    """Normalize each rank array to [0,1] then sum."""
    combined = np.zeros(len(rank_arrays[0]))
    for r in rank_arrays:
        r_norm = (r - r.min()) / (r.max() - r.min() + 1e-9)
        combined += r_norm
    return combined


def main():
    args  = get_args()
    df    = pd.read_csv(args.csv)
    folds = build_folds(df)
    y_mem   = df["memorability_score"].values
    y_brand = df["brand_memorability"].values

    # ── Build features ────────────────────────────────────────────────────────
    corpus    = load_transcripts(df, args.stt)
    llm       = load_llm(df, args.llm)
    face_rate = np.load(Path(args.feat) / "frame_stats.npy")[:, 6:7]

    tfidf = TfidfVectorizer(
        max_features=3000, min_df=3, max_df=0.8,
        ngram_range=(1, 3), stop_words="english", sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(corpus)

    for target, y in [("memorability_score", y_mem), ("brand_memorability", y_brand)]:
        n_comp = 150 if target == "memorability_score" else 100
        svd    = TruncatedSVD(n_components=n_comp, random_state=42)
        X_lsa  = svd.fit_transform(X_tfidf).astype(np.float32)

        # individual rank aggregations
        r_lsa  = rank_aggregate(X_lsa,      y, folds)
        r_llm  = rank_aggregate(llm,        y, folds)
        r_face = rank_aggregate(face_rate,  y, folds)

        # combinations
        combos = {
            "lsa"               : r_lsa,
            "llm"               : r_llm,
            "face"              : r_face,
            "lsa+llm"           : combine_ranks(r_lsa, r_llm),
            "lsa+face"          : combine_ranks(r_lsa, r_face),
            "llm+face"          : combine_ranks(r_llm, r_face),
            "lsa+llm+face"      : combine_ranks(r_lsa, r_llm, r_face),
        }

        print(f"\n{'═'*45}\n  TARGET: {target}\n{'═'*45}")
        for name, preds in combos.items():
            r = spearman(y, preds)
            print(f"  {name:20s}  ρ = {r:.4f}")


if __name__ == "__main__":
    main()
