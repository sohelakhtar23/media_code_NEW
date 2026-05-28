"""
Combine title + description + transcript into TF-IDF corpus.
All fields available for every video — no availability gaps.

Usage:
    python train_text_combined.py \
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


def load_stt(df, stt_dir):
    stt_dir = Path(stt_dir)
    return [
        (stt_dir / f"{vid}.txt").read_text(encoding="utf-8").strip()
        if (stt_dir / f"{vid}.txt").exists() else ""
        for vid in df["id"]
    ]


def build_corpus(df, stt_texts, mode):
    """Build text corpus based on mode."""
    titles = df["title"].fillna("").tolist()
    descs  = df["description"].fillna("").str[:500].tolist()  # cap description length

    if mode == "stt":
        return stt_texts
    elif mode == "title+desc":
        return [f"{t} {d}".strip() for t, d in zip(titles, descs)]
    elif mode == "title+stt":
        return [f"{t} {s}".strip() for t, s in zip(titles, stt_texts)]
    elif mode == "desc+stt":
        return [f"{d} {s}".strip() for d, s in zip(descs, stt_texts)]
    elif mode == "all":
        return [f"{t} {d} {s}".strip() for t, d, s in zip(titles, descs, stt_texts)]


def load_llm(df, llm_path):
    cache = json.loads(Path(llm_path).read_text())
    return np.array([
        [float(cache.get(vid, {}).get(k, 5.0)) for k in LLM_KEYS]
        for vid in df["id"]
    ], dtype=np.float32)


def rank_aggregate(X, y, folds, threshold):
    preds = np.zeros(len(y))
    for fold_id in np.unique(folds):
        val_mask    = folds == fold_id
        X_tr, X_val = X[~val_mask], X[val_mask]
        y_tr        = y[~val_mask]
        scores      = np.zeros(val_mask.sum())
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
        combined += (a - a.min()) / (a.max() - a.min() + 1e-9)
    return combined


def eval_corpus(corpus, llm, y, folds, n_comp, threshold, use_llm):
    tfidf = TfidfVectorizer(
        max_features=3000, min_df=3, max_df=0.8,
        ngram_range=(1, 3), stop_words="english", sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(corpus)
    svd     = TruncatedSVD(n_components=n_comp, random_state=42)
    X_lsa   = svd.fit_transform(X_tfidf).astype(np.float32)

    r_lsa = rank_aggregate(X_lsa, y, folds, threshold)
    if use_llm:
        r_llm = rank_aggregate(llm, y, folds, threshold)
        preds = combine_ranks(r_lsa, r_llm)
    else:
        preds = r_lsa

    return spearman(y, preds)


def main():
    args  = get_args()
    df    = pd.read_csv(args.csv)
    folds = build_folds(df)
    y_mem   = df["memorability_score"].values
    y_brand = df["brand_memorability"].values

    stt_texts = load_stt(df, args.stt)
    llm       = load_llm(df, args.llm)

    corpus_modes = ["stt", "title+desc", "title+stt", "desc+stt", "all"]

    # best configs from tuning
    configs = {
        "memorability_score": {"n_comp": 50,  "threshold": 0.07, "use_llm": True},
        "brand_memorability": {"n_comp": 100, "threshold": 0.01, "use_llm": False},
    }

    for target, y in [("memorability_score", y_mem), ("brand_memorability", y_brand)]:
        cfg = configs[target]
        print(f"\n{'═'*50}\n  TARGET: {target}\n{'═'*50}")
        print(f"  (n_comp={cfg['n_comp']}, threshold={cfg['threshold']}, llm={cfg['use_llm']})\n")

        best_score, best_mode = -1, None
        for mode in corpus_modes:
            corpus = build_corpus(df, stt_texts, mode)
            r = eval_corpus(corpus, llm, y, folds,
                            cfg["n_comp"], cfg["threshold"], cfg["use_llm"])
            marker = " ★" if r > best_score else ""
            print(f"  {mode:15s}  ρ = {r:.4f}{marker}")
            if r > best_score:
                best_score, best_mode = r, mode

        print(f"\n  Best corpus: {best_mode}  ρ = {best_score:.4f}")


if __name__ == "__main__":
    main()
