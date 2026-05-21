"""
Train and evaluate models for memorability prediction.
Uses 5-fold CV stratified by channel.

Usage:
    python train.py \
        --csv   devset_videolist_GT.csv \
        --feat  features/ \
        --out   predictions/
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
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
    p.add_argument("--pca-components", type=int, default=128)
    return p.parse_args()


def load_features(feat_dir: Path) -> np.ndarray:
    visual = np.load(feat_dir / "visual_features.npy")   # (N, 3072)
    text   = np.load(feat_dir / "text_features.npy")     # (N, 768)
    meta   = np.load(feat_dir / "meta_features.npy")     # (N, 18)
    return visual, text, meta


def spearman(y_true, y_pred):
    return spearmanr(y_true, y_pred).statistic


def get_models():
    return {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=10.0)),
        ]),
        "svr": Pipeline([
            ("scaler", StandardScaler()),
            ("svr",    SVR(kernel="rbf", C=1.0, epsilon=0.05)),
        ]),
        "xgb": XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5,
            reg_lambda=5.0, random_state=42, verbosity=0,
        ),
    }


def channel_kfold(df: pd.DataFrame, n_splits=5):
    """K-fold that keeps videos from the same channel together where possible."""
    channels = df["channelName"].values
    unique_ch = np.unique(channels)
    np.random.seed(42)
    np.random.shuffle(unique_ch)
    ch_folds = np.array_split(unique_ch, n_splits)

    folds = []
    idx   = np.arange(len(df))
    for val_channels in ch_folds:
        val_mask  = np.isin(channels, val_channels)
        folds.append((idx[~val_mask], idx[val_mask]))
    return folds


def cross_validate(X, y, folds, models: dict):
    """Returns dict of model → list of fold Spearman scores."""
    results = {name: [] for name in models}

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        for name, model in models.items():
            import copy
            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr)
            pred = m.predict(X_val)
            r    = spearman(y_val, pred)
            results[name].append(r)
            print(f"  fold {fold_i+1}  {name:8s}  ρ={r:.4f}")

    return results


def train_and_predict(X, y, model):
    """Train on full data and return fitted model."""
    import copy
    m = copy.deepcopy(model)
    m.fit(X, y)
    return m


def main():
    args    = get_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    feat    = Path(args.feat)

    df      = pd.read_csv(args.csv)
    y_mem   = df["memorability_score"].values
    y_brand = df["brand_memorability"].values

    visual, text, meta = load_features(feat)

    # ── PCA on visual features ────────────────────────────────────────────────
    print(f"\nApplying PCA: {visual.shape[1]} → {args.pca_components}")
    pca = PCA(n_components=args.pca_components, random_state=42)
    visual_pca = pca.fit_transform(visual)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  Variance explained: {var_explained:.3f}")

    # ── Feature combinations to try ───────────────────────────────────────────
    combos = {
        "visual_only"  : visual_pca,
        "text_only"    : text,
        "meta_only"    : meta,
        "visual+meta"  : np.concatenate([visual_pca, meta], axis=1),
        "visual+text"  : np.concatenate([visual_pca, text], axis=1),
        "all"          : np.concatenate([visual_pca, text, meta], axis=1),
    }

    folds = channel_kfold(df)

    # ── CV for both targets ───────────────────────────────────────────────────
    for target_name, y in [("memorability_score", y_mem), ("brand_memorability", y_brand)]:
        print(f"\n{'═'*60}")
        print(f"  TARGET: {target_name}")
        print(f"{'═'*60}")

        best_score, best_combo, best_model_name = -1, None, None

        for combo_name, X in combos.items():
            print(f"\n  Features: {combo_name}  shape={X.shape}")
            models  = get_models()
            results = cross_validate(X, y, folds, models)

            for name, scores in results.items():
                mean_r = np.mean(scores)
                print(f"    {name:8s}  mean ρ={mean_r:.4f}  folds={[f'{s:.3f}' for s in scores]}")
                if mean_r > best_score:
                    best_score, best_combo, best_model_name = mean_r, combo_name, name

        print(f"\n  ★ Best: combo={best_combo}  model={best_model_name}  ρ={best_score:.4f}")

    # ── Final predictions using best combo (all) with ensemble ───────────────
    print(f"\n{'═'*60}")
    print("  FINAL PREDICTIONS (train on full data)")
    print(f"{'═'*60}")

    X_all = combos["all"]
    models = get_models()

    submission_rows = []
    for target_name, y in [("memorability_score", y_mem), ("brand_memorability", y_brand)]:
        preds = []
        for name, model in models.items():
            m = train_and_predict(X_all, y, model)
            preds.append(m.predict(X_all))  # in-sample just to save; replace with test when available

        # simple average ensemble
        ensemble_pred = np.mean(preds, axis=0)
        submission_rows.append((target_name, ensemble_pred))

    # save predictions (will be overwritten with test set predictions)
    ids = df["id"].values
    pred_df = pd.DataFrame({"id": ids})
    for name, preds in submission_rows:
        pred_df[name] = preds

    pred_df.to_csv(out_dir / "train_predictions.csv", index=False)
    print(f"  Saved predictions to {out_dir}/train_predictions.csv")


if __name__ == "__main__":
    main()

    


# 1. PCA reduces visual from 3072 → 128 dims (prints variance explained — we want ≥ 0.80)
# 2. 6 feature combos × 3 models × 5 folds = 90 CV runs, for both targets
# 3. Prints the winning combo + model per target
# 4. Saves ensemble predictions on train set (placeholder until we have test features)
