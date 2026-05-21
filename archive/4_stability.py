"""
Stability check: run CV across multiple random seeds.
Also tries tuned OOF base models for stacking.

Usage:
    python stability.py \
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
    p.add_argument("--seeds", type=int, default=20)
    return p.parse_args()


def spearman(a, b):
    return spearmanr(a, b).statistic


def channel_kfold(df, n_splits=5, seed=42):
    channels  = df["channelName"].values
    unique_ch = np.unique(channels)
    rng       = np.random.default_rng(seed)
    rng.shuffle(unique_ch)
    idx = np.arange(len(df))
    return [(idx[~np.isin(channels, cf)], idx[np.isin(channels, cf)])
            for cf in np.array_split(unique_ch, n_splits)]


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


def make_oof_cross(X, y_mem, y_brand, folds, model_mem, model_brand):
    """Generate OOF cross-target predictions."""
    oof_mem   = np.zeros(len(y_mem))
    oof_brand = np.zeros(len(y_brand))
    for tr, val in folds:
        m = copy.deepcopy(model_mem)
        m.fit(X[tr], y_mem[tr]);   oof_mem[val]   = m.predict(X[val])
        m = copy.deepcopy(model_brand)
        m.fit(X[tr], y_brand[tr]); oof_brand[val] = m.predict(X[val])
    return oof_mem.reshape(-1, 1), oof_brand.reshape(-1, 1)


def eval_with_seed(X_base, y_mem, y_brand, seed,
                   oof_model_mem, oof_model_brand, final_model_mem, final_model_brand):
    folds = channel_kfold_from_seed(X_base, y_mem, seed)

    oof_mem, oof_brand = make_oof_cross(
        X_base, y_mem, y_brand, folds, oof_model_mem, oof_model_brand
    )

    X_mem   = np.concatenate([X_base, oof_brand], axis=1)
    X_brand = np.concatenate([X_base, oof_mem],   axis=1)

    scores_mem, scores_brand = [], []
    for tr, val in folds:
        m = copy.deepcopy(final_model_mem)
        m.fit(X_mem[tr], y_mem[tr])
        scores_mem.append(spearman(y_mem[val], m.predict(X_mem[val])))

        m = copy.deepcopy(final_model_brand)
        m.fit(X_brand[tr], y_brand[tr])
        scores_brand.append(spearman(y_brand[val], m.predict(X_brand[val])))

    return np.mean(scores_mem), np.mean(scores_brand)


def channel_kfold_from_seed(df_or_none, y, seed, n_splits=5, channels=None):
    # reuse channel_kfold — need df reference globally
    return _FOLDS_CACHE[seed]


# We'll store df globally for simplicity
_DF = None
_FOLDS_CACHE = {}


def main():
    global _DF, _FOLDS_CACHE
    args = get_args()
    _DF  = pd.read_csv(args.csv)
    feat = Path(args.feat)

    meta = build_meta(_DF)
    llm  = load_llm(_DF, args.llm)
    y_mem   = _DF["memorability_score"].values
    y_brand = _DF["brand_memorability"].values

    X_base = np.concatenate([meta, llm], axis=1)

    # precompute folds for all seeds
    seeds = list(range(args.seeds))
    for s in seeds:
        _FOLDS_CACHE[s] = channel_kfold(_DF, seed=s)

    # models — using best found so far
    oof_model   = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10))])
    final_model = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=100))])

    results_mem, results_brand = [], []

    print(f"Running CV across {args.seeds} seeds...\n")
    print(f"  {'seed':>4s}   ρ_video   ρ_brand")

    for s in seeds:
        folds = _FOLDS_CACHE[s]

        oof_mem, oof_brand = make_oof_cross(
            X_base, y_mem, y_brand, folds, oof_model, oof_model
        )

        X_mem   = np.concatenate([X_base, oof_brand], axis=1)
        X_brand = np.concatenate([X_base, oof_mem],   axis=1)

        scores_mem, scores_brand = [], []
        for tr, val in folds:
            m = copy.deepcopy(final_model)
            m.fit(X_mem[tr], y_mem[tr])
            scores_mem.append(spearman(y_mem[val], m.predict(X_mem[val])))

            m = copy.deepcopy(final_model)
            m.fit(X_brand[tr], y_brand[tr])
            scores_brand.append(spearman(y_brand[val], m.predict(X_brand[val])))

        r_mem   = np.mean(scores_mem)
        r_brand = np.mean(scores_brand)
        results_mem.append(r_mem)
        results_brand.append(r_brand)
        print(f"  {s:>4d}   {r_mem:.4f}    {r_brand:.4f}")

    print(f"\n{'═'*40}")
    print(f"  video_mem  : mean={np.mean(results_mem):.4f}  std={np.std(results_mem):.4f}"
          f"  min={np.min(results_mem):.4f}  max={np.max(results_mem):.4f}")
    print(f"  brand_mem  : mean={np.mean(results_brand):.4f}  std={np.std(results_brand):.4f}"
          f"  min={np.min(results_brand):.4f}  max={np.max(results_brand):.4f}")


if __name__ == "__main__":
    main()
