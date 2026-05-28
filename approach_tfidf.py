"""
TF-IDF + LSA + Rank Aggregation approach.
Best configs:
  memorability_score : LSA(n=50) + LLM rank agg, threshold=0.07
  brand_memorability : LSA(n=100) rank agg only,  threshold=0.01

Train CV:
    python approach_tfidf.py --mode train \
        --train-csv devset_videolist_GT.csv \
        --stt       devset-stt/ \
        --llm       llm_scalar_cache_v2.json

Test prediction:
    python approach_tfidf.py --mode test \
        --train-csv devset_videolist_GT.csv \
        --test-csv  predict/testset_videolist_.csv \
        --stt       devset-stt/ \
        --stt-test  predict/testset-stt/ \
        --llm       llm_scalar_cache_v2.json \
        --llm-test  predict/llm_scalar_cache_test.json \
        --out       predict/
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

# Best configs found from tuning
CONFIGS = {
    "memorability_score": {"n_comp": 50,  "threshold": 0.07, "use_llm": True},
    "brand_memorability": {"n_comp": 100, "threshold": 0.01, "use_llm": False},
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       required=True, choices=["train", "test"])
    p.add_argument("--train-csv",  required=True)
    p.add_argument("--test-csv",   default=None)
    p.add_argument("--stt",        required=True, help="Train STT directory")
    p.add_argument("--stt-test",   default=None,  help="Test STT directory")
    p.add_argument("--llm",        required=True, help="Train LLM cache")
    p.add_argument("--llm-test",   default=None,  help="Test LLM cache")
    p.add_argument("--out",        default="predict")
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


def load_llm(df, llm_path):
    cache = json.loads(Path(llm_path).read_text())
    return np.array([
        [float(cache.get(vid, {}).get(k, 5.0)) for k in LLM_KEYS]
        for vid in df["id"]
    ], dtype=np.float32)


def fit_tfidf_svd(corpus, n_comp):
    """Fit TF-IDF + SVD on corpus. Returns transformed features + fitted objects."""
    tfidf = TfidfVectorizer(
        max_features=3000, min_df=3, max_df=0.8,
        ngram_range=(1, 3), stop_words="english", sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(corpus)
    svd     = TruncatedSVD(n_components=n_comp, random_state=42)
    X_lsa   = svd.fit_transform(X_tfidf).astype(np.float32)
    return X_lsa, tfidf, svd


def transform_tfidf_svd(corpus, tfidf, svd):
    """Transform new corpus using fitted TF-IDF + SVD."""
    X_tfidf = tfidf.transform(corpus)
    return svd.transform(X_tfidf).astype(np.float32)


def _rank_agg_oof(X, y, folds, threshold):
    """Compute OOF rank aggregation predictions for a feature matrix."""
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


def rank_aggregate_cv(X_lsa, llm, y, folds, threshold, use_llm):
    """CV rank aggregation — normalize across full OOF array before combining."""
    def norm(a): return (a - a.min()) / (a.max() - a.min() + 1e-9)

    oof_lsa = _rank_agg_oof(X_lsa, y, folds, threshold)
    if not use_llm:
        return oof_lsa

    oof_llm = _rank_agg_oof(llm, y, folds, threshold)
    return norm(oof_lsa) + norm(oof_llm)


def rank_aggregate_test(X_lsa_train, llm_train, y_train,
                         X_lsa_test,  llm_test,
                         threshold, use_llm):
    """
    Test prediction via rank aggregation.
    Strategy (a): rank test videos among themselves using column directions
    learned from full training set.
    """
    n_test = X_lsa_test.shape[0]

    # learn column directions from full training set
    lsa_scores = np.zeros(n_test)
    for col in range(X_lsa_train.shape[1]):
        r, _ = spearmanr(X_lsa_train[:, col], y_train)
        if np.isnan(r) or abs(r) < threshold:
            continue
        lsa_scores += np.sign(r) * rankdata(X_lsa_test[:, col])

    if not use_llm:
        return lsa_scores

    llm_scores = np.zeros(n_test)
    for col in range(llm_train.shape[1]):
        r, _ = spearmanr(llm_train[:, col], y_train)
        if np.isnan(r) or abs(r) < threshold:
            continue
        llm_scores += np.sign(r) * rankdata(llm_test[:, col])

    def norm(a): return (a - a.min()) / (a.max() - a.min() + 1e-9)
    return norm(lsa_scores) + norm(llm_scores)


def run_train(args):
    df    = pd.read_csv(args.train_csv)
    folds = build_folds(df)
    corpus = load_stt(df, args.stt)
    llm    = load_llm(df, args.llm)

    print(f"Training set: {len(df)} videos")

    for target in ["memorability_score", "brand_memorability"]:
        y   = df[target].values
        cfg = CONFIGS[target]

        X_lsa, _, _ = fit_tfidf_svd(corpus, cfg["n_comp"])
        preds = rank_aggregate_cv(X_lsa, llm, y, folds,
                                  cfg["threshold"], cfg["use_llm"])
        r = spearman(y, preds)
        print(f"\n  {target}")
        print(f"    CV ρ = {r:.4f}")
        print(f"    config: n_comp={cfg['n_comp']}  threshold={cfg['threshold']}  use_llm={cfg['use_llm']}")


def run_test(args):
    if not args.test_csv or not args.stt_test or not args.llm_test:
        raise ValueError("--test-csv, --stt-test, --llm-test all required for test mode")

    df_train = pd.read_csv(args.train_csv)
    df_test  = pd.read_csv(args.test_csv)

    corpus_train = load_stt(df_train, args.stt)
    corpus_test  = load_stt(df_test,  args.stt_test)
    llm_train    = load_llm(df_train, args.llm)
    llm_test     = load_llm(df_test,  args.llm_test)

    print(f"Train: {len(df_train)} videos  |  Test: {len(df_test)} videos")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({"id": df_test["id"].values})

    for target in ["memorability_score", "brand_memorability"]:
        y_train = df_train[target].values
        cfg     = CONFIGS[target]

        # fit on full training corpus, transform test
        X_lsa_train, tfidf, svd = fit_tfidf_svd(corpus_train, cfg["n_comp"])
        X_lsa_test               = transform_tfidf_svd(corpus_test, tfidf, svd)

        preds = rank_aggregate_test(
            X_lsa_train, llm_train, y_train,
            X_lsa_test,  llm_test,
            cfg["threshold"], cfg["use_llm"],
        )
        # rescale raw rank sums to training score range
        preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-9)
        preds = preds * (y_train.max() - y_train.min()) + y_train.min()
        pred_df[target] = preds

        print(f"  {target}: min={preds.min():.4f}  max={preds.max():.4f}")

    out_path = out_dir / "predictions_tfidf.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


def main():
    args = get_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_test(args)


if __name__ == "__main__":
    main()
