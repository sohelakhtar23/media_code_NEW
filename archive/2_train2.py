"""
Train and evaluate with all feature sets including zero-shot and STT semantic.

Usage:
    python train2.py \
        --csv   devset_videolist_GT.csv \
        --feat  features/ \
        --out   predictions/
"""

import argparse
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  required=True)
    p.add_argument("--feat", default="features")
    p.add_argument("--out",  default="predictions")
    return p.parse_args()


def spearman(y_true, y_pred):
    return spearmanr(y_true, y_pred).statistic


def get_models():
    return {
        "ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]),
        "svr":   Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf", C=1.0, epsilon=0.05))]),
        "xgb":   XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.5,
                               reg_lambda=5.0, random_state=42, verbosity=0),
    }


def channel_kfold(df, n_splits=5):
    channels  = df["channelName"].values
    unique_ch = np.unique(channels)
    rng       = np.random.default_rng(42)
    rng.shuffle(unique_ch)
    ch_folds  = np.array_split(unique_ch, n_splits)
    idx       = np.arange(len(df))
    return [(idx[~np.isin(channels, cf)], idx[np.isin(channels, cf)]) for cf in ch_folds]


def cv_eval(X, y, folds, models):
    results = {n: [] for n in models}
    for train_idx, val_idx in folds:
        for name, model in models.items():
            m = copy.deepcopy(model)
            m.fit(X[train_idx], y[train_idx])
            r = spearman(y[val_idx], m.predict(X[val_idx]))
            results[name].append(r)
    return {n: np.mean(v) for n, v in results.items()}


def main():
    args    = get_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    feat    = Path(args.feat)
    df      = pd.read_csv(args.csv)

    # ── load features ─────────────────────────────────────────────────────────
    visual   = np.load(feat / "visual_features.npy")       # (N, 3072)
    text     = np.load(feat / "text_features.npy")         # (N, 768)  title+desc
    meta     = np.load(feat / "meta_features.npy")         # (N, 18)
    zeroshot = np.load(feat / "clip_zeroshot.npy")         # (N, 27)
    stt      = np.load(feat / "stt_semantic.npy")          # (N, 768)

    # PCA on raw visual
    pca    = PCA(n_components=128, random_state=42)
    visual = pca.fit_transform(visual)

    folds  = channel_kfold(df)
    models = get_models()

    combos = {
        "meta_only"        : meta,
        "zeroshot_only"    : zeroshot,
        "stt_only"         : stt,
        "meta+zeroshot"    : np.concatenate([meta, zeroshot], axis=1),
        "meta+stt"         : np.concatenate([meta, stt], axis=1),
        "zeroshot+stt"     : np.concatenate([zeroshot, stt], axis=1),
        "meta+zeroshot+stt": np.concatenate([meta, zeroshot, stt], axis=1),
        "all"              : np.concatenate([visual, text, meta, zeroshot, stt], axis=1),
    }

    best_per_target = {}

    for target_name, y in [
        ("memorability_score", df["memorability_score"].values),
        ("brand_memorability",  df["brand_memorability"].values),
    ]:
        print(f"\n{'═'*60}\n  TARGET: {target_name}\n{'═'*60}")
        best_score, best_combo, best_model = -1, None, None

        for combo_name, X in combos.items():
            scores = cv_eval(X, y, folds, models)
            for name, r in scores.items():
                print(f"  {combo_name:25s}  {name:6s}  ρ={r:.4f}")
                if r > best_score:
                    best_score, best_combo, best_model = r, combo_name, name

        print(f"\n  ★ Best: combo={best_combo}  model={best_model}  ρ={best_score:.4f}")
        best_per_target[target_name] = (best_combo, best_model, best_score)

    # ── Final ensemble predictions ────────────────────────────────────────────
    print(f"\n{'═'*60}\n  FINAL PREDICTIONS\n{'═'*60}")
    pred_df = pd.DataFrame({"id": df["id"].values})

    for target_name, y in [
        ("memorability_score", df["memorability_score"].values),
        ("brand_memorability",  df["brand_memorability"].values),
    ]:
        best_combo, best_model_name, _ = best_per_target[target_name]
        X = combos[best_combo]
        preds = []
        for name, model in models.items():
            m = copy.deepcopy(model)
            m.fit(X, y)
            preds.append(m.predict(X))
        pred_df[target_name] = np.mean(preds, axis=0)

    pred_df.to_csv(out_dir / "train_predictions.csv", index=False)
    print(f"  Saved to {out_dir}/train_predictions.csv")


if __name__ == "__main__":
    main()
