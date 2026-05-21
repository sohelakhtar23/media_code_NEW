"""
Tune hyperparameters on meta+face_rate, then produce test predictions.

Usage:
    python tune_and_predict.py \
        --train-csv  devset_videolist_GT.csv \
        --test-csv   testset_videolist.csv \
        --feat       features/ \
        --out        predictions/

If no --test-csv is provided, skips test prediction.
"""

import argparse
import copy
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


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv",  default=None)
    p.add_argument("--feat",      default="features")
    p.add_argument("--out",       default="predictions")
    return p.parse_args()


def spearman(a, b):
    return spearmanr(a, b).statistic


def channel_kfold(df, n_splits=5, seed=42):
    channels  = df["channelName"].values
    unique_ch = np.unique(channels)
    rng       = np.random.default_rng(seed)
    rng.shuffle(unique_ch)
    ch_folds  = np.array_split(unique_ch, n_splits)
    idx       = np.arange(len(df))
    return [(idx[~np.isin(channels, cf)], idx[np.isin(channels, cf)]) for cf in ch_folds]


def build_meta(df_train, df_test=None):
    """Build meta+face_rate features. Target-encode channel from train only."""
    ch_mem   = df_train.groupby("channelName")["memorability_score"].mean()
    ch_brand = df_train.groupby("channelName")["brand_memorability"].mean()
    global_mem   = df_train["memorability_score"].mean()
    global_brand = df_train["brand_memorability"].mean()

    def featurize(df, is_train=True):
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
        feat["channel_mem_enc"]   = df["channelName"].map(ch_mem).fillna(global_mem)
        feat["channel_brand_enc"] = df["channelName"].map(ch_brand).fillna(global_brand)
        return feat.values.astype(np.float32)

    train_feat = featurize(df_train)
    test_feat  = featurize(df_test) if df_test is not None else None
    return train_feat, test_feat


def cv_score(model, X, y, folds):
    scores = []
    for train_idx, val_idx in folds:
        m = copy.deepcopy(model)
        m.fit(X[train_idx], y[train_idx])
        scores.append(spearman(y[val_idx], m.predict(X[val_idx])))
    return np.mean(scores)


def tune(X, y, folds):
    """Grid search over Ridge, SVR, XGB. Returns best (model, score)."""
    grids = [
        {
            "model": [Pipeline([("sc", StandardScaler()), ("m", Ridge())])],
            "model__m__alpha": [0.1, 1, 10, 50, 100, 500],
        },
        {
            "model": [Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf"))])],
            "model__m__C":       [0.1, 0.5, 1.0, 5.0, 10.0],
            "model__m__epsilon": [0.01, 0.05, 0.1],
        },
        {
            "model": [XGBRegressor(random_state=42, verbosity=0)],
            "model__n_estimators":  [100, 200, 300],
            "model__max_depth":     [2, 3, 4],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__reg_lambda":    [1, 5, 10],
        },
    ]

    best_score, best_model = -999, None
    for grid_spec in grids:
        model_proto = grid_spec.pop("model")[0]
        for params in ParameterGrid(grid_spec):
            m = copy.deepcopy(model_proto)
            for k, v in params.items():
                # set params like model__m__alpha on Pipeline, or model__n_estimators on XGB
                key = k.replace("model__", "")
                if isinstance(m, Pipeline):
                    m.set_params(**{key: v})
                else:
                    m.set_params(**{key.split("__")[-1]: v})
            r = cv_score(m, X, y, folds)
            if r > best_score:
                best_score, best_model = r, copy.deepcopy(m)
        grid_spec["model"] = [model_proto]  # restore

    return best_model, best_score


def main():
    args    = get_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    feat    = Path(args.feat)

    df_train = pd.read_csv(args.train_csv)
    df_test  = pd.read_csv(args.test_csv) if args.test_csv else None

    # load face_rate and append to meta
    frame_stats = np.load(feat / "frame_stats.npy")          # (N, 10)
    face_rate   = frame_stats[:, 6:7]                        # face_rate column only

    X_train_meta, X_test_meta = build_meta(df_train, df_test)
    X_train = np.concatenate([X_train_meta, face_rate], axis=1)

    folds = channel_kfold(df_train)

    results = {}
    for target in ["memorability_score", "brand_memorability"]:
        y = df_train[target].values
        print(f"\nTuning for {target} ...")
        best_model, best_score = tune(X_train, y, folds)
        print(f"  Best CV ρ = {best_score:.4f}  model = {best_model}")
        results[target] = (best_model, best_score)

    # ── test predictions ──────────────────────────────────────────────────────
    if df_test is not None:
        # test face_rate
        test_frame_stats = np.load(feat / "test_frame_stats.npy")
        test_face_rate   = test_frame_stats[:, 6:7]
        X_test = np.concatenate([X_test_meta, test_face_rate], axis=1)

        pred_df = pd.DataFrame({"id": df_test["id"].values})
        for target, (model, score) in results.items():
            y      = df_train[target].values
            m      = copy.deepcopy(model)
            m.fit(X_train, y)
            pred_df[target] = m.predict(X_test)
            print(f"\n{target}: trained on full set, predicting {len(df_test)} test videos")

        pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
        print(f"\nSaved → {out_dir}/test_predictions.csv")
    else:
        print("\nNo test CSV provided — skipping test predictions.")
        print("Run with --test-csv testset_videolist.csv when ready.")

        # save best CV scores summary
        summary = {t: s for t, (_, s) in results.items()}
        print(f"\nFinal CV summary: {summary}")


if __name__ == "__main__":
    main()
