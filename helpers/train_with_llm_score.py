"""
Integrate LLM semantic scalars into training.

Usage:
    python train_with_llm_score2.py 
    --csv  devset_videolist_GT.csv --feat features/ 
    --llm  llm_scalar_cache_v2.json
"""

import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
    p.add_argument("--feat", default="features")
    p.add_argument("--llm",  required=True)
    p.add_argument("--out",  default="predictions")
    return p.parse_args()


def load_llm_features(df, llm_path):
    cache = json.loads(Path(llm_path).read_text())
    rows  = []
    missing = 0
    for vid_id in df["id"]:
        if vid_id in cache:
            rows.append([cache[vid_id].get(k, 5.0) for k in LLM_KEYS])
        else:
            rows.append([5.0] * len(LLM_KEYS))  # neutral fallback
            missing += 1
    if missing:
        print(f"  [WARN] {missing} videos missing from LLM cache — using neutral 5.0")
    return np.array(rows, dtype=np.float32)


def build_meta(df):
    ch_mem   = df.groupby("channelName")["memorability_score"].mean()
    ch_brand = df.groupby("channelName")["brand_memorability"].mean()
    feat = pd.DataFrame()
    feat["log_views"]         = np.log1p(df["viewsCount"])
    feat["log_likes"]         = np.log1p(df["likesCount"])
    feat["log_dislikes"]      = np.log1p(df["dislikesCount"])
    feat["log_comments"]      = np.log1p(df["commentsCount"])
    feat["log_engagements"]   = np.log1p(df["engagementsCount"])
    feat["engagement_rate"]   = df["engagementRate"]
    feat["log_duration"]      = np.log1p(df["durationSeconds"])
    feat["is_long"]           = (df["durationSeconds"] > 60).astype(float)
    feat["nb_annotations"]    = df["nb_annotations"]
    feat["channel_mem_enc"]   = df["channelName"].map(ch_mem).fillna(ch_mem.mean())
    feat["channel_brand_enc"] = df["channelName"].map(ch_brand).fillna(ch_brand.mean())
    return feat.values.astype(np.float32)


# def channel_kfold(df, n_splits=5, seed=42):
#     channels  = df["channelName"].values
#     unique_ch = np.unique(channels)
#     rng       = np.random.default_rng(seed)
#     rng.shuffle(unique_ch)
#     ch_folds  = np.array_split(unique_ch, n_splits)
#     idx       = np.arange(len(df))
#     return [(idx[~np.isin(channels, cf)], idx[np.isin(channels, cf)]) for cf in ch_folds]

def channel_stratified_kfold(df, n_splits=5):
    """
    Sort channels by size descending. Assign channels to folds greedily
    to keep fold sizes balanced (like a bin-packing approximation).
    Goldman Sachs (largest) always goes into its own fold first.
    """
    ch_sizes = df["channelName"].value_counts()  # sorted descending
    channels = ch_sizes.index.tolist()
    sizes    = ch_sizes.values.tolist()

    # greedy bin-packing: assign each channel to the smallest fold so far
    fold_channels = [[] for _ in range(n_splits)]
    fold_sizes    = [0] * n_splits

    for ch, sz in zip(channels, sizes):
        smallest = int(np.argmin(fold_sizes))
        fold_channels[smallest].append(ch)
        fold_sizes[smallest] += sz

    print("  Fold composition:")
    for i, (chs, sz) in enumerate(zip(fold_channels, fold_sizes)):
        print(f"    fold {i+1}: {sz:3d} videos — {chs}")

    idx = np.arange(len(df))
    ch_arr = df["channelName"].values
    folds = []
    for val_channels in fold_channels:
        val_mask = np.isin(ch_arr, val_channels)
        folds.append((idx[~val_mask], idx[val_mask]))
    return folds


def cv_score(model, X, y, folds):
    scores = []
    for train_idx, val_idx in folds:
        m = copy.deepcopy(model)
        m.fit(X[train_idx], y[train_idx])
        scores.append(spearman(y[val_idx], m.predict(X[val_idx])))
    return np.mean(scores)

def spearman(a, b):
    return spearmanr(a, b).statistic


def tune(X, y, folds):
    grids = [
        (Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
         {"m__alpha": [0.1, 1, 10, 50, 100, 500]}),
        (Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf"))]),
         {"m__C": [0.1, 0.5, 1.0, 5.0, 10.0], "m__epsilon": [0.01, 0.05, 0.1]}),
        (XGBRegressor(random_state=42, verbosity=0),
         {"n_estimators": [100, 200, 300], "max_depth": [2, 3, 4],
          "learning_rate": [0.03, 0.05, 0.1], "reg_lambda": [1, 5, 10]}),
    ]
    best_score, best_model = -999, None
    for proto, grid in grids:
        for params in ParameterGrid(grid):
            m = copy.deepcopy(proto)
            m.set_params(**params)
            r = cv_score(m, X, y, folds)
            if r > best_score:
                best_score, best_model = r, copy.deepcopy(m)
    return best_model, best_score


def main():
    args = get_args()
    df   = pd.read_csv(args.csv)
    feat = Path(args.feat)

    frame_stats = np.load(feat / "frame_stats.npy")
    face_rate   = frame_stats[:, 6:7]

    meta = build_meta(df)
    llm  = load_llm_features(df, args.llm)

    # ── correlation report ────────────────────────────────────────────────────
    print("\nSpearman ρ — LLM scalars vs targets:")
    print(f"  {'dimension':25s}  ρ_video   ρ_brand")
    for i, k in enumerate(LLM_KEYS):
        r1 = spearman(llm[:, i], df["memorability_score"].values)
        r2 = spearman(llm[:, i], df["brand_memorability"].values)
        print(f"  {k:25s}  {r1:+.3f}     {r2:+.3f}")

    # folds = channel_kfold(df)
    folds = channel_stratified_kfold(df)

    combos = {
        "meta_only"      : meta,
        "llm_only"       : llm,
        "meta+face"      : np.concatenate([meta, face_rate], axis=1),
        "meta+llm"       : np.concatenate([meta, llm], axis=1),
        "meta+face+llm"  : np.concatenate([meta, face_rate, llm], axis=1),
    }

    for target in ["memorability_score", "brand_memorability"]:
        y = df[target].values
        print(f"\n{'═'*55}\n  TARGET: {target}\n{'═'*55}")
        for name, X in combos.items():
            _, score = tune(X, y, folds)
            print(f"  {name:20s}  best CV ρ = {score:.4f}")


if __name__ == "__main__":
    main()
