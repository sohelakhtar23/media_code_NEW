"""
Compare channel-aware vs channel-agnostic models.
Channel-agnostic may generalize better to unseen test channels.

Usage:
    python channel_ablation.py \
        --csv  devset_videolist_GT.csv \
        --feat features/ \
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings("ignore")

LLM_KEYS_V2 = [
    "emotional_valence", "human_presence", "message_simplicity", "novelty_surprise",
    "narrative_arc", "brand_prominence", "repetition_hooks", "direct_memorability",
]

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  required=True)
    p.add_argument("--feat", default="features")
    p.add_argument("--llm",  required=True)
    return p.parse_args()


def spearman(a, b):
    return spearmanr(a, b).statistic


def channel_stratified_kfold(df, n_splits=5):
    ch_sizes = df["channelName"].value_counts()
    fold_channels = [[] for _ in range(n_splits)]
    fold_sizes    = [0] * n_splits
    for ch, sz in zip(ch_sizes.index, ch_sizes.values):
        i = int(np.argmin(fold_sizes))
        fold_channels[i].append(ch)
        fold_sizes[i] += sz
    idx    = np.arange(len(df))
    ch_arr = df["channelName"].values
    return [(idx[~np.isin(ch_arr, cf)], idx[np.isin(ch_arr, cf)]) for cf in fold_channels]


def build_features(df, include_channel=True):
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

    if include_channel:
        ch_mem   = df.groupby("channelName")["memorability_score"].mean()
        ch_brand = df.groupby("channelName")["brand_memorability"].mean()
        feat["channel_mem_enc"]   = df["channelName"].map(ch_mem).fillna(ch_mem.mean())
        feat["channel_brand_enc"] = df["channelName"].map(ch_brand).fillna(ch_brand.mean())

    return feat.values.astype(np.float32)


def load_llm(df, llm_path):
    cache = json.loads(Path(llm_path).read_text())
    rows  = []
    for vid_id in df["id"]:
        entry = cache.get(vid_id, {})
        rows.append([float(entry.get(k, 5.0)) for k in LLM_KEYS_V2])
    return np.array(rows, dtype=np.float32)


def tune(X, y, folds):
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
            for tr, val in folds:
                m2 = copy.deepcopy(m)
                m2.fit(X[tr], y[tr])
                scores.append(spearman(y[val], m2.predict(X[val])))
            r = np.mean(scores)
            if r > best_score:
                best_score, best_model = r, copy.deepcopy(m)
    return best_model, best_score


def cv_per_fold(X, y, folds, model):
    scores = []
    for tr, val in folds:
        m = copy.deepcopy(model)
        m.fit(X[tr], y[tr])
        scores.append(spearman(y[val], m.predict(X[val])))
    return scores


def main():
    args  = get_args()
    df    = pd.read_csv(args.csv)
    feat  = Path(args.feat)
    llm   = load_llm(df, args.llm)
    folds = channel_stratified_kfold(df)

    meta_with_ch    = build_features(df, include_channel=True)
    meta_without_ch = build_features(df, include_channel=False)

    X_aware    = np.concatenate([meta_with_ch,    llm], axis=1)
    X_agnostic = np.concatenate([meta_without_ch, llm], axis=1)

    print(f"\n  {'combo':35s}  ρ_mean   per-fold")
    for target in ["memorability_score", "brand_memorability"]:
        y = df[target].values
        print(f"\n  TARGET: {target}")

        best_aware,    score_aware    = tune(X_aware,    y, folds)
        best_agnostic, score_agnostic = tune(X_agnostic, y, folds)

        folds_aware    = cv_per_fold(X_aware,    y, folds, best_aware)
        folds_agnostic = cv_per_fold(X_agnostic, y, folds, best_agnostic)

        print(f"  {'channel-aware (meta+llm)':35s}  {score_aware:.4f}   {[f'{s:.3f}' for s in folds_aware]}")
        print(f"  {'channel-agnostic (no ch enc)':35s}  {score_agnostic:.4f}   {[f'{s:.3f}' for s in folds_agnostic]}")
        print(f"  {'best_model_aware':35s}  {best_aware}")
        print(f"  {'best_model_agnostic':35s}  {best_agnostic}")


if __name__ == "__main__":
    main()
