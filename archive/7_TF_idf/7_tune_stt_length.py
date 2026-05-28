"""
Test different STT truncation lengths.
Annotators watched at most 60 seconds, so only early transcript is relevant.

Usage:
    python tune_stt_length.py \
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


def load_stt(df, stt_dir, max_words=None):
    stt_dir = Path(stt_dir)
    texts   = []
    for vid in df["id"]:
        fp = stt_dir / f"{vid}.txt"
        if fp.exists():
            words = fp.read_text(encoding="utf-8", errors="replace").strip().split()
            txt   = " ".join(words[:max_words]) if max_words else " ".join(words)
        else:
            txt = ""
        texts.append(txt)
    return texts


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


def eval_stt_length(corpus, llm, y, folds, n_comp, threshold, use_llm):
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
    llm     = load_llm(df, args.llm)

    # None = full transcript
    word_limits = [None, 1000, 600, 400, 200, 130, 75]

    configs = {
        "memorability_score": {"n_comp": 50,  "threshold": 0.07, "use_llm": True},
        "brand_memorability": {"n_comp": 100, "threshold": 0.01, "use_llm": False},
    }

    # print word count stats for context
    all_lengths = []
    for vid in df["id"]:
        fp = Path(args.stt) / f"{vid}.txt"
        if fp.exists():
            text = fp.read_text(encoding="utf-8", errors="replace")
            all_lengths.append(len(text.split()))
    all_lengths = np.array(all_lengths)
    print(f"STT word counts — min={all_lengths.min()}  median={int(np.median(all_lengths))}"
          f"  mean={int(all_lengths.mean())}  max={all_lengths.max()}")
    print(f"Videos with <130 words: {(all_lengths < 130).sum()}/{len(all_lengths)}\n")

    print(f"{'max_words':<12}  ρ_video   ρ_brand")
    print("-" * 35)

    best_video, best_brand = -1, -1
    best_video_lim, best_brand_lim = None, None

    for lim in word_limits:
        corpus  = load_stt(df, args.stt, max_words=lim)
        label   = str(lim) if lim else "full"

        r_video = eval_stt_length(corpus, llm, y_mem,   folds,
                                  **configs["memorability_score"])
        r_brand = eval_stt_length(corpus, llm, y_brand, folds,
                                  **configs["brand_memorability"])

        v_mark = " ★" if r_video > best_video else ""
        b_mark = " ★" if r_brand > best_brand else ""
        print(f"{label:<12}  {r_video:.4f}{v_mark:<3}  {r_brand:.4f}{b_mark}")

        if r_video > best_video: best_video, best_video_lim = r_video, lim
        if r_brand > best_brand: best_brand, best_brand_lim = r_brand, lim

    print(f"\n  ★ Best video : max_words={best_video_lim}  ρ={best_video:.4f}")
    print(f"  ★ Best brand : max_words={best_brand_lim}  ρ={best_brand:.4f}")


if __name__ == "__main__":
    main()
