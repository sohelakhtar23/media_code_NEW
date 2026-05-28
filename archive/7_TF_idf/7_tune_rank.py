"""
Joint tuning of LSA n_components and rank aggregation threshold.

Usage:
    python tune_rank.py \
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


def rank_aggregate(X, y, folds, threshold):
    preds = np.zeros(len(y))
    # for fold_id in range(5):
    for fold_id in np.unique(folds):
        val_mask     = folds == fold_id
        X_tr, X_val  = X[~val_mask], X[val_mask]
        y_tr         = y[~val_mask]
        scores       = np.zeros(val_mask.sum())

        for col in range(X_tr.shape[1]):
            r, _ = spearmanr(X_tr[:, col], y_tr)
            if np.isnan(r) or abs(r) < threshold:
                continue
            scores += np.sign(r) * rankdata(X_val[:, col])
        preds[val_mask] = scores
    return preds


def combine_ranks(*arrays):
    combined = np.zeros(len(arrays[0]))
    for a in arrays:
        rng = a.max() - a.min()
        combined += (a - a.min()) / (rng + 1e-9)
    return combined


def main():
    args  = get_args()
    df    = pd.read_csv(args.csv)
    folds = build_folds(df)
    y_mem   = df["memorability_score"].values
    y_brand = df["brand_memorability"].values

    corpus = load_transcripts(df, args.stt)
    llm    = load_llm(df, args.llm)

    tfidf = TfidfVectorizer(
        max_features=3000, min_df=3, max_df=0.8,
        ngram_range=(1, 3), stop_words="english", sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(corpus)

    n_comp_values  = [30, 50, 80, 100, 120, 150, 200, 250]
    threshold_values = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]

    for target, y, best_combo in [
        ("memorability_score", y_mem,   "lsa+llm"),
        ("brand_memorability",  y_brand, "lsa"),
    ]:
        print(f"\n{'═'*60}")
        print(f"  TARGET: {target}  (combo: {best_combo})")
        print(f"{'═'*60}")
        print(f"  {'n_comp':<8}", end="")
        for t in threshold_values:
            print(f"  thresh={t:.2f}", end="")
        print()

        best_score, best_n, best_t = -1, None, None

        for n_comp in n_comp_values:
            svd   = TruncatedSVD(n_components=n_comp, random_state=42)
            X_lsa = svd.fit_transform(X_tfidf).astype(np.float32)
            print(f"  {n_comp:<8}", end="")

            for thresh in threshold_values:
                r_lsa = rank_aggregate(X_lsa, y, folds, thresh)

                if best_combo == "lsa+llm":
                    r_llm  = rank_aggregate(llm, y, folds, thresh)
                    preds  = combine_ranks(r_lsa, r_llm)
                else:
                    preds  = r_lsa

                r = spearman(y, preds)
                marker = " *" if r > best_score else "  "
                print(f"  {r:.4f}{marker}    ", end="")

                if r > best_score:
                    best_score, best_n, best_t = r, n_comp, thresh

            print()

        print(f"\n  ★ Best: n_comp={best_n}  threshold={best_t}  ρ={best_score:.4f}")


if __name__ == "__main__":
    main()
