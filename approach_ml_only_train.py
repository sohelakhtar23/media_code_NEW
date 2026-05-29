"""
ML Model approach: Ridge / SVR / XGBoost on LLM scalars + engagement metadata.

Train CV:
    python approach_ml_without_test.py \
        --train-csv devset_videolist_GT.csv \
        --feat      features/ \
        --llm       llm_scalar_cache_v2.json
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
    p.add_argument("--train-csv",  required=True)
    p.add_argument("--feat",       default="features")
    p.add_argument("--llm",        required=True, help="Train LLM cache")
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


def build_engagement(df):
    feat = pd.DataFrame()
    feat["log_views"]       = np.log1p(df["viewsCount"])
    feat["engagement_rate"] = df["engagementRate"]
    feat["log_duration"]    = np.log1p(df["durationSeconds"])
    return feat.values.astype(np.float32)


def load_llm(df, llm_path):
    cache = json.loads(Path(llm_path).read_text())
    return np.array([
        [float(cache.get(vid, {}).get(k, 5.0)) for k in LLM_KEYS]
        for vid in df["id"]
    ], dtype=np.float32)


def load_frame_stats(feat_dir, n_videos=None, filename="frame_stats.npy"):
    return np.load(Path(feat_dir) / filename)[:, 6:7]  # face_rate only


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
            for fold_id in np.unique(folds):
                val_mask = folds == fold_id
                m2 = copy.deepcopy(m)
                m2.fit(X[~val_mask], y[~val_mask])
                scores.append(spearman(y[val_mask], m2.predict(X[val_mask])))
            r = np.mean(scores)
            if r > best_score:
                best_score, best_model = r, copy.deepcopy(m)
    return best_model, best_score


def run_train(args):
    df    = pd.read_csv(args.train_csv)
    folds = build_folds(df)

    engagement = build_engagement(df)
    llm        = load_llm(df, args.llm)
    face_rate  = load_frame_stats(args.feat)
    X          = np.concatenate([engagement, llm, face_rate], axis=1)
    # X          = np.concatenate([llm, face_rate], axis=1)

    print(f"Feature matrix: {X.shape}  "
          f"({engagement.shape[1]}d engagement + {llm.shape[1]}d LLM + 1d face_rate)")

    results = {}
    for target in ["memorability_score", "brand_memorability"]:
        y = df[target].values
        best_model, best_score = tune(X, y, folds)
        results[target] = (best_model, best_score)
        print(f"\n  {target}")
        print(f"    best CV ρ = {best_score:.4f}")
        print(f"    model     = {best_model}")

    return results, X, df


def main():
    args = get_args()
    run_train(args)


if __name__ == "__main__":
    main()
