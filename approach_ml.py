"""
ML Model approach: Ridge / SVR / XGBoost on LLM scalars + engagement metadata.

Train CV:
    python approach_ml.py --mode train \
        --train-csv devset_videolist_GT.csv \
        --feat      features/ \
        --llm       llm_scalar_cache_v2.json

Test prediction:
    python approach_ml.py --mode test \
        --train-csv devset_videolist_GT.csv \
        --test-csv  predict/testset_videolist_.csv \
        --feat      features/ \
        --llm       llm_scalar_cache_v2.json \
        --llm-test  predict/llm_scalar_cache_test.json \
        --out       predict/
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
    p.add_argument("--mode",       required=True, choices=["train", "test"])
    p.add_argument("--train-csv",  required=True)
    p.add_argument("--test-csv",   default=None)
    p.add_argument("--feat",       default="features")
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


def build_engagement(df):
    feat = pd.DataFrame()
    feat["log_views"]       = np.log1p(df["viewsCount"])
    # feat["log_likes"]       = np.log1p(df["likesCount"])
    feat["engagement_rate"] = df["engagementRate"]
    feat["log_duration"]    = np.log1p(df["durationSeconds"])
    # feat["is_long"]         = (df["durationSeconds"] > 60).astype(float)
    # feat["nb_annotations"]  = df["nb_annotations"]
    return feat.values.astype(np.float32)


def load_llm(df, llm_path):
    cache = json.loads(Path(llm_path).read_text())
    return np.array([
        [float(cache.get(vid, {}).get(k, 5.0)) for k in LLM_KEYS]
        for vid in df["id"]
    ], dtype=np.float32)


def load_frame_stats(feat_dir, n_videos=None, filename="frame_stats.npy"):
    return np.load(Path(feat_dir) / filename)[:, 6:7]  # face_rate only


def get_models():
    return {
        "ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
        "svr":   Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf"))]),
        "xgb":   XGBRegressor(random_state=42, verbosity=0),
    }


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


def run_test(args):
    # ── train ────────────────────────────────────────────────────────────────
    df_train = pd.read_csv(args.train_csv)
    folds    = build_folds(df_train)

    eng_train  = build_engagement(df_train)
    llm_train  = load_llm(df_train, args.llm)
    face_train = load_frame_stats(args.feat)
    X_train    = np.concatenate([eng_train, llm_train, face_train], axis=1)

    # ── test ─────────────────────────────────────────────────────────────────
    df_test    = pd.read_csv(args.test_csv)
    eng_test   = build_engagement(df_test)
    llm_test   = load_llm(df_test, args.llm_test)
    face_test  = load_frame_stats(args.feat, filename="test_frame_stats.npy")
    X_test     = np.concatenate([eng_test, llm_test, face_test], axis=1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({"id": df_test["id"].values})

    for target in ["memorability_score", "brand_memorability"]:
        y_train = df_train[target].values
        best_model, best_score = tune(X_train, y_train, folds)
        print(f"\n  {target}  CV ρ={best_score:.4f}")

        # retrain on full training set
        best_model.fit(X_train, y_train)
        pred_df[target] = best_model.predict(X_test)

        # rescale to training score range
        preds = pred_df[target].values
        preds = np.clip(preds, y_train.min(), y_train.max())
        pred_df[target] = preds

    out_path = out_dir / "predictions_ml.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


def main():
    args = get_args()
    if args.mode == "train":
        run_train(args)
    else:
        if not args.test_csv or not args.llm_test:
            raise ValueError("--test-csv and --llm-test required for test mode")
        run_test(args)


if __name__ == "__main__":
    main()
