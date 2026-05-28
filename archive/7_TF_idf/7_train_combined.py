"""
Combine TF-IDF+LSA (transcript), LLM scalars, and face_rate features.
Uses channel-stratified CV, no channel encoding.

Usage:
    python train_combined.py 
        --csv   devset_videolist_GT.csv --stt   devset-stt/ 
        --feat  features/ --llm   llm_scalar_cache_v2.json
"""

import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

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


# ── CV fold builder ───────────────────────────────────────────────────────────
def build_folds(df, n_folds=5):
    ch_counts  = df["channelName"].value_counts()
    fold_map   = {}
    fold_sizes = {i: 0 for i in range(n_folds)}

    for ch, cnt in ch_counts.items():
        target = min(range(n_folds), key=lambda f: fold_sizes[f])
        fold_map[ch]         = target
        fold_sizes[target]  += cnt

    return df["channelName"].map(fold_map).values


# ── Feature loaders ───────────────────────────────────────────────────────────
def load_transcripts(df, stt_dir):
    stt_dir = Path(stt_dir)
    texts   = []
    for vid in df["id"]:
        fp = stt_dir / f"{vid}.txt"
        texts.append(fp.read_text(encoding="utf-8").strip() if fp.exists() else "")
    return texts


def load_llm(df, llm_path):
    cache = json.loads(Path(llm_path).read_text())
    rows  = []
    for vid in df["id"]:
        entry = cache.get(vid, {})
        rows.append([float(entry.get(k, 5.0)) for k in LLM_KEYS])
    return np.array(rows, dtype=np.float32)


def build_engagement(df):
    feat = pd.DataFrame()
    feat["log_views"]       = np.log1p(df["viewsCount"])
    feat["log_likes"]       = np.log1p(df["likesCount"])
    feat["engagement_rate"] = df["engagementRate"]
    feat["log_duration"]    = np.log1p(df["durationSeconds"])
    feat["is_long"]         = (df["durationSeconds"] > 60).astype(float)
    feat["nb_annotations"]  = df["nb_annotations"]
    return feat.values.astype(np.float32)


# ── Rank aggregation predictor (from Feb_Tfidf.py) ───────────────────────────
def rank_aggregate_cv(X_dense, y, folds, threshold=0.03):
    predictions = np.zeros(len(y))
    for fold_id in np.unique(folds):
        val_mask   = folds == fold_id
        train_mask = ~val_mask
        X_tr, X_val = X_dense[train_mask], X_dense[val_mask]
        y_tr = y[train_mask]
        scores = np.zeros(val_mask.sum())
        for col in range(X_tr.shape[1]):
            r, _ = spearmanr(X_tr[:, col], y_tr)
            if np.isnan(r) or abs(r) < threshold:
                continue
            scores += np.sign(r) * rankdata(X_val[:, col])
        predictions[val_mask] = scores
    return spearman(y, predictions)


# ── Standard ML CV ────────────────────────────────────────────────────────────
def tune_ml(X, y, folds):
    grids = [
        (Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
         {"m__alpha": [0.1, 1, 10, 50, 100, 500]}),
        (Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf"))]),
         {"m__C": [0.1, 0.5, 1.0, 5.0], "m__epsilon": [0.01, 0.05, 0.1]}),
        (XGBRegressor(random_state=42, verbosity=0),
         {"n_estimators": [100, 200], "max_depth": [2, 3],
          "learning_rate": [0.03, 0.05, 0.1], "reg_lambda": [1, 5, 10]}),
    ]
    best_score, best_model = -999, None
    for proto, grid in grids:
        for params in ParameterGrid(grid):
            m = copy.deepcopy(proto)
            m.set_params(**params)
            scores = []
            for fold_id in np.unique(folds):
                val_mask = folds == fold_id
                m2 = copy.deepcopy(m)
                m2.fit(X[~val_mask], y[~val_mask])
                scores.append(spearman(y[val_mask], m2.predict(X[val_mask])))
            r = np.mean(scores)
            if r > best_score:
                best_score, best_model = r, copy.deepcopy(m)
    return best_score, best_model


def main():
    args  = get_args()
    df    = pd.read_csv(args.csv)
    feat  = Path(args.feat)
    folds = build_folds(df)

    y_mem   = df["memorability_score"].values
    y_brand = df["brand_memorability"].values

    # ── Build features ────────────────────────────────────────────────────────
    corpus      = load_transcripts(df, args.stt)
    llm         = load_llm(df, args.llm)
    engagement  = build_engagement(df)
    face_rate   = np.load(feat / "frame_stats.npy")[:, 6:7]

    # TF-IDF + LSA — fit on full corpus (mild leak, same as baseline)
    tfidf = TfidfVectorizer(
        max_features=3000, min_df=3, max_df=0.8,
        ngram_range=(1, 3), stop_words="english", sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(corpus)

    # find best LSA dim for each target using rank aggregation (fast scan)
    print("Scanning LSA dimensions...")
    print(f"{'n_comp':<10} {'Mem ρ':>8} {'Brand ρ':>8}")
    best_lsa = {"memorability_score": (50, -1), "brand_memorability": (50, -1)}

    for n_comp in [50, 80, 100, 120, 150, 200]:
        svd   = TruncatedSVD(n_components=n_comp, random_state=42)
        X_lsa = svd.fit_transform(X_tfidf)
        r_mem   = rank_aggregate_cv(X_lsa, y_mem,   folds)
        r_brand = rank_aggregate_cv(X_lsa, y_brand, folds)
        print(f"{n_comp:<10} {r_mem:>8.4f} {r_brand:>8.4f}")
        if r_mem   > best_lsa["memorability_score"][1]: best_lsa["memorability_score"] = (n_comp, r_mem)
        if r_brand > best_lsa["brand_memorability"][1]: best_lsa["brand_memorability"] = (n_comp, r_brand)

    print(f"\nBest LSA dims — video: {best_lsa['memorability_score'][0]}, brand: {best_lsa['brand_memorability'][0]}")

    # ── Evaluate all combos per target ────────────────────────────────────────
    for target, y in [("memorability_score", y_mem), ("brand_memorability", y_brand)]:
        n_comp = best_lsa[target][0]
        svd    = TruncatedSVD(n_components=n_comp, random_state=42)
        X_lsa  = svd.fit_transform(X_tfidf).astype(np.float32)

        combos = {
            # "lsa_only"              : X_lsa,
            # "llm_only"              : llm,
            "engagement_only"       : engagement,
            # "lsa+llm"               : np.concatenate([X_lsa, llm],                        axis=1),
            # "lsa+face"              : np.concatenate([X_lsa, face_rate],                   axis=1),
            # "llm+face"              : np.concatenate([llm,   face_rate],                   axis=1),
            # "lsa+llm+face"          : np.concatenate([X_lsa, llm, face_rate],              axis=1),
            "lsa+llm+engagement"    : np.concatenate([X_lsa, llm, engagement],             axis=1),
            "lsa+llm+face+engagement": np.concatenate([X_lsa, llm, face_rate, engagement], axis=1),
        }

        print(f"\n{'═'*55}\n  TARGET: {target}\n{'═'*55}")

        # rank aggregation baseline for LSA
        r_ra = rank_aggregate_cv(X_lsa, y, folds)
        print(f"  {'lsa_rank_aggregate':30s}  ρ = {r_ra:.4f}  (rank agg)")

        for name, X in combos.items():
            score, model = tune_ml(X, y, folds)
            print(f"  {name:30s}  ρ = {score:.4f}  ({type(model[-1]).__name__ if hasattr(model, '__len__') else type(model).__name__})")


if __name__ == "__main__":
    main()
