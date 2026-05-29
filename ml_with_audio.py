"""
ML Model approach: Ridge / SVR / XGBoost on LLM scalars + engagement + audio features.

Compares four feature combinations to isolate audio contribution:
  A) baseline  — engagement + LLM + face_rate          (original)
  B) +audio    — engagement + LLM + face_rate + audio  (all audio)
  C) +audio_top— engagement + LLM + face_rate + top-K audio features (filtered by ρ)
  D) audio_only— audio features only

Usage:
    python ml_with_audio.py \
        --train-csv devset_videolist_GT.csv \
        --feat      features/ \
        --llm       llm_scalar_cache_v2.json \
        --audio     features/audio_features.json
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

# Audio features to exclude — non-predictive meta fields
AUDIO_EXCLUDE = {"audio_duration_s", "stt_word_count"}

# |ρ| threshold for "top-K" audio feature selection (per target)
AUDIO_RHO_THRESHOLD = 0.08


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", default="devset_videolist_GT.csv")
    p.add_argument("--feat",      default="features")
    p.add_argument("--llm",       default="llm_scalar_cache_v2.json", help="Train LLM cache JSON")
    p.add_argument("--audio",     default="features/audio_features.json",
                   help="Audio features cache JSON from approach_audio.py")
    return p.parse_args()


def spearman(a, b):
    r = spearmanr(a, b).statistic
    return float(r) if not np.isnan(r) else 0.0


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
        [float(cache.get(str(vid), {}).get(k, 5.0)) for k in LLM_KEYS]
        for vid in df["id"]
    ], dtype=np.float32)


def load_frame_stats(feat_dir, filename="frame_stats.npy"):
    return np.load(Path(feat_dir) / filename)[:, 6:7]  # face_rate only


def load_audio(df, audio_path) -> tuple[np.ndarray, list[str]]:
    """
    Load audio features from JSON cache.
    Returns (matrix [n_videos × n_features], feature_names).
    Videos missing from cache get zeros (with a warning).
    """
    if not Path(audio_path).exists():
        print(f"  [WARN] Audio cache not found: {audio_path} — audio features skipped")
        return np.zeros((len(df), 0), dtype=np.float32), []

    cache = json.loads(Path(audio_path).read_text())

    # Determine feature names from first entry
    sample = next(iter(cache.values()))
    feat_names = [k for k in sample.keys() if k not in AUDIO_EXCLUDE and k != "id"]

    missing = 0
    rows = []
    for vid in df["id"].astype(str):
        entry = cache.get(vid)
        if entry is None:
            rows.append([0.0] * len(feat_names))
            missing += 1
        else:
            rows.append([float(entry.get(k, 0.0)) for k in feat_names])

    if missing:
        print(f"  [WARN] {missing}/{len(df)} videos missing from audio cache — filled with zeros")

    return np.array(rows, dtype=np.float32), feat_names


def select_top_audio_features(audio_X: np.ndarray, feat_names: list[str],
                               y: np.ndarray,
                               threshold: float = AUDIO_RHO_THRESHOLD
                               ) -> tuple[np.ndarray, list[str]]:
    """Filter audio features to those with |ρ| > threshold against y."""
    keep_idx = []
    for i in range(audio_X.shape[1]):
        rho = spearman(audio_X[:, i], y)
        if abs(rho) >= threshold:
            keep_idx.append(i)
    if not keep_idx:
        return audio_X, feat_names   # fallback: keep all
    selected_names = [feat_names[i] for i in keep_idx]
    return audio_X[:, keep_idx], selected_names


def tune(X, y, folds, label=""):
    """Grid-search Ridge / SVR / XGB, return best model and its mean CV ρ."""
    grids = [
        (Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
         {"m__alpha": [0.1, 1, 10, 50, 100, 500]}),
        (Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf"))]),
         {"m__C": [0.1, 0.5, 1.0, 5.0], "m__epsilon": [0.01, 0.05, 0.1]}),
        (XGBRegressor(random_state=42, verbosity=0),
         {"n_estimators": [100, 200], "max_depth": [2, 3],
          "learning_rate": [0.03, 0.05, 0.1], "reg_lambda": [1, 5, 10]}),
    ]
    best_score, best_model, best_params = -999, None, ""
    for proto, grid in grids:
        for params in ParameterGrid(grid):
            m = copy.deepcopy(proto)
            m.set_params(**params)
            scores = []
            for fold_id in np.unique(folds):
                val_mask = folds == fold_id
                m2 = copy.deepcopy(m)
                Xtr, Xva = X[~val_mask], X[val_mask]
                Xtr = np.nan_to_num(Xtr)
                Xva = np.nan_to_num(Xva)
                m2.fit(Xtr, y[~val_mask])
                scores.append(spearman(y[val_mask], m2.predict(Xva)))
            r = float(np.mean(scores))
            if r > best_score:
                best_score  = r
                best_model  = copy.deepcopy(m)
                best_params = str(params)
    return best_model, best_score, best_params


def run_combo(label, X, y, folds, n_features_desc):
    """Tune and report one feature combination."""
    model, score, params = tune(X, y, folds, label=label)
    print(f"    {label:<18} | {n_features_desc:<32} | ρ = {score:.4f}  [{params}]")
    return score, model


def run_train(args):
    df    = pd.read_csv(args.train_csv)
    folds = build_folds(df)

    # ── Load feature blocks ──────────────────────────────────────────────────
    engagement = build_engagement(df)                         # 3d
    llm        = load_llm(df, args.llm)                       # 8d
    face_rate  = load_frame_stats(args.feat)                  # 1d
    audio_X, audio_names = load_audio(df, args.audio)         # 77d

    baseline = np.concatenate([engagement, llm, face_rate], axis=1)

    print(f"\nFeature blocks loaded:")
    print(f"  engagement : {engagement.shape[1]}d")
    print(f"  LLM scalars: {llm.shape[1]}d")
    print(f"  face_rate  : {face_rate.shape[1]}d")
    print(f"  audio      : {audio_X.shape[1]}d  ({len(audio_names)} features)")

    # ── Evaluate per target ──────────────────────────────────────────────────
    summary = {}
    for target in ["memorability_score", "brand_memorability"]:
        y = df[target].values
        print(f"\n{'='*72}")
        print(f"  TARGET: {target}")
        print(f"{'='*72}")

        # Top audio features filtered by ρ against this target
        audio_top, top_names = select_top_audio_features(audio_X, audio_names, y)
        print(f"  Audio features above |ρ|≥{AUDIO_RHO_THRESHOLD}: "
              f"{audio_top.shape[1]} / {audio_X.shape[1]}")
        if top_names:
            # Show which features were selected
            corrs = [(n, spearman(audio_X[:, audio_names.index(n)], y))
                     for n in top_names]
            corrs.sort(key=lambda x: abs(x[1]), reverse=True)
            for n, r in corrs[:8]:
                print(f"    {n:<35} ρ={r:+.4f}")
            if len(top_names) > 8:
                print(f"    ... ({len(top_names)-8} more)")

        print(f"\n  Combo results:")
        combos = {
            "A_baseline":   (baseline,
                             f"{baseline.shape[1]}d (eng+llm+face)"),
            "B_+audio_all": (np.concatenate([baseline, audio_X], axis=1),
                             f"{baseline.shape[1]+audio_X.shape[1]}d (+all audio)"),
            "C_+audio_top": (np.concatenate([baseline, audio_top], axis=1),
                             f"{baseline.shape[1]+audio_top.shape[1]}d (+top audio)"),
            "D_audio_only": (audio_X,
                             f"{audio_X.shape[1]}d (audio only)"),
        }

        best_rho, best_label = -999, ""
        target_results = {}
        for label, (X_combo, desc) in combos.items():
            score, model = run_combo(label, X_combo, y, folds, desc)[:2]
            target_results[label] = score
            if score > best_rho:
                best_rho, best_label = score, label

        summary[target] = target_results
        print(f"\n  → Best for {target}: {best_label}  ρ = {best_rho:.4f}")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Target':<28} {'A_baseline':>12} {'B_+audio_all':>14} "
          f"{'C_+audio_top':>14} {'D_audio_only':>14}")
    print(f"  {'-'*28} {'-'*12} {'-'*14} {'-'*14} {'-'*14}")
    for target, scores in summary.items():
        row = f"  {target:<28}"
        for k in ["A_baseline", "B_+audio_all", "C_+audio_top", "D_audio_only"]:
            v = scores.get(k, float("nan"))
            marker = " ◄" if v == max(scores.values()) else ""
            row += f" {v:>13.4f}{marker}"
        print(row)

    print(f"\n  [TF-IDF rank agg baseline]")
    print(f"  memorability_score           ρ = 0.2697")
    print(f"  brand_memorability           ρ = 0.1801")

    return summary


def main():
    args = get_args()
    run_train(args)


if __name__ == "__main__":
    main()
