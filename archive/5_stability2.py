"""
Channel-stratified GroupKFold: largest channels get dedicated folds,
remaining channels are grouped by size to balance fold sizes.

Usage:
    python stability2.py --csv  devset_videolist_GT.csv --feat features/ --llm  llm_scalar_cache_v2.json
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
    p.add_argument("--n-splits", type=int, default=5)
    return p.parse_args()


def spearman(a, b):
    return spearmanr(a, b).statistic


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


def load_llm(df, llm_path):
    cache = json.loads(Path(llm_path).read_text())
    rows  = []
    for vid_id in df["id"]:
        entry = cache.get(vid_id, {})
        rows.append([float(entry.get(k, 5.0)) for k in LLM_KEYS_V2])
    return np.array(rows, dtype=np.float32)


def make_oof_cross(X, y_mem, y_brand, folds, model):
    oof_mem   = np.zeros(len(y_mem))
    oof_brand = np.zeros(len(y_brand))
    for tr, val in folds:
        m = copy.deepcopy(model)
        m.fit(X[tr], y_mem[tr]);   oof_mem[val]   = m.predict(X[val])
        m = copy.deepcopy(model)
        m.fit(X[tr], y_brand[tr]); oof_brand[val] = m.predict(X[val])
    return oof_mem.reshape(-1, 1), oof_brand.reshape(-1, 1)


def cv_score(X, y, folds, model):
    scores = []
    for tr, val in folds:
        m = copy.deepcopy(model)
        m.fit(X[tr], y[tr])
        scores.append(spearman(y[val], m.predict(X[val])))
    return np.mean(scores), scores


def main():
    args = get_args()
    df   = pd.read_csv(args.csv)
    feat = Path(args.feat)

    meta    = build_meta(df)
    llm     = load_llm(df, args.llm)
    y_mem   = df["memorability_score"].values
    y_brand = df["brand_memorability"].values
    X_base  = np.concatenate([meta, llm], axis=1)

    oof_model   = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10))])
    final_model = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=100))])

    print("\nChannel-stratified fold composition:")
    folds = channel_stratified_kfold(df, n_splits=args.n_splits)

    # without cross-target stacking
    mean_mem,   folds_mem   = cv_score(X_base, y_mem,   folds, final_model)
    mean_brand, folds_brand = cv_score(X_base, y_brand, folds, final_model)

    print(f"\n  meta+llm (no stacking):")
    print(f"    video_mem  : mean={mean_mem:.4f}  folds={[f'{s:.3f}' for s in folds_mem]}")
    print(f"    brand_mem  : mean={mean_brand:.4f}  folds={[f'{s:.3f}' for s in folds_brand]}")

    # with cross-target stacking
    oof_mem, oof_brand = make_oof_cross(X_base, y_mem, y_brand, folds, oof_model)
    X_mem   = np.concatenate([X_base, oof_brand], axis=1)
    X_brand = np.concatenate([X_base, oof_mem],   axis=1)

    mean_mem_s,   folds_mem_s   = cv_score(X_mem,   y_mem,   folds, final_model)
    mean_brand_s, folds_brand_s = cv_score(X_brand, y_brand, folds, final_model)

    print(f"\n  meta+llm+cross (with stacking):")
    print(f"    video_mem  : mean={mean_mem_s:.4f}  folds={[f'{s:.3f}' for s in folds_mem_s]}")
    print(f"    brand_mem  : mean={mean_brand_s:.4f}  folds={[f'{s:.3f}' for s in folds_brand_s]}")

    print(f"\n{'═'*50}")
    print(f"  Summary (stratified CV, no seed variance):")
    print(f"  {'combo':30s}  ρ_video   ρ_brand")
    print(f"  {'meta+llm':30s}  {mean_mem:.4f}    {mean_brand:.4f}")
    print(f"  {'meta+llm+cross':30s}  {mean_mem_s:.4f}    {mean_brand_s:.4f}")


if __name__ == "__main__":
    main()
