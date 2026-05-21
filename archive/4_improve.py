"""
Two improvements:
  1. Refined LLM prompt with memorability-focused dimensions
  2. Cross-target stacking (predicted brand score as feature for video and vice versa)

Step 1 — regenerate LLM scores with new dimensions:
    python improve.py --mode llm \
        --csv devset_videolist_GT.csv \
        --stt devset-stt/ \
        --llm llm_scalar_cache_v2.json \

Step 2 — train with cross-target stacking:
    python improve.py --mode train \
        --csv  devset_videolist_GT.csv \
        --feat features/ \
        --llm  llm_scalar_cache_v2.json
"""

import argparse
import copy
import json
import os
import time
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

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

# ── Refined dimensions focused on memorability ────────────────────────────────
LLM_DIMENSIONS_V2 = {
    "emotional_valence":     "Emotional tone: 0=cold/negative → 10=warm/uplifting/positive",
    "human_presence":        "Are real people (faces, presenters, spokespeople) prominent? 0=none, 10=dominant",
    "message_simplicity":    "How simple and easy to remember is the core message? 0=complex/forgettable, 10=single clear takeaway",
    "novelty_surprise":      "How unexpected or counter-intuitive is the message? 0=predictable, 10=surprising",
    "narrative_arc":         "Clear story structure (problem→solution or journey)? 0=none, 10=strong arc",
    "brand_prominence":      "How centrally/repeatedly is the brand name featured? 0=never, 10=constant",
    "repetition_hooks":      "Repeated phrases, slogans, or memorable hooks? 0=none, 10=strong repetition",
    "direct_memorability":   "Overall: how memorable would this video be to a viewer? 0=very forgettable, 10=highly memorable",
}

LLM_KEYS_V2 = list(LLM_DIMENSIONS_V2.keys())

_SYSTEM = (
    "You are an expert in advertising psychology and memory research. "
    "Given a commercial video's metadata and transcript, score it on specific dimensions. "
    "Return ONLY a valid JSON object — no markdown, no commentary."
)

def build_prompt(row, stt_text, max_words=600):
    stt_clip     = " ".join(stt_text.split()[:max_words])
    schema_lines = "\n".join(f'  "{k}": {v}' for k, v in LLM_DIMENSIONS_V2.items())
    keys_json    = "{" + ", ".join(f'"{k}": ...' for k in LLM_KEYS_V2) + "}"
    return (
        f"Commercial video:\n"
        f"- Brand   : {row.get('channelName', 'Unknown')}\n"
        f"- Title   : {row.get('title', '')}\n"
        f"- Desc    : {str(row.get('description', ''))[:400]}\n"
        f"- Duration: {row.get('durationSeconds', 0):.0f}s\n"
        f"- Transcript (~{max_words} words): {stt_clip}\n\n"
        f"Score on these dimensions (floats 0-10):\n{schema_lines}\n\n"
        f"Return ONLY: {keys_json}"
    )


def run_llm_mode(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    df     = pd.read_csv(args.csv)
    stt_dir = Path(args.stt)

    cache_path = Path(args.llm)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    for _, row in df.iterrows():
        vid_id = row["id"]
        if vid_id in cache:
            continue

        stt_file = stt_dir / f"{vid_id}.txt"
        stt_text = stt_file.read_text(encoding="utf-8", errors="ignore") if stt_file.exists() else ""

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=256,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": build_prompt(row, stt_text)},
                ],
            )
            scores = json.loads(resp.choices[0].message.content)
            cache[vid_id] = {k: float(scores.get(k, 5.0)) for k in LLM_KEYS_V2}
            print(f"  {vid_id}: {cache[vid_id]}")
        except Exception as e:
            print(f"  [ERR] {vid_id}: {e}")
            cache[vid_id] = {k: 5.0 for k in LLM_KEYS_V2}

        cache_path.write_text(json.dumps(cache, indent=2))
        time.sleep(0.1)

    print(f"Done. {len(cache)} videos in cache.")


# ── Training helpers ──────────────────────────────────────────────────────────
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

def load_llm(df, llm_path, keys):
    cache = json.loads(Path(llm_path).read_text())
    rows  = []
    for vid_id in df["id"]:
        entry = cache.get(vid_id, {})
        rows.append([float(entry.get(k, 5.0)) for k in keys])
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


def make_cross_target_features(X, y_mem, y_brand, folds):
    """
    For each fold, train on the other target using other folds,
    then predict val set. Returns OOF predictions for both targets.
    These can be used as features without leaking labels.
    """
    oof_mem   = np.zeros(len(y_mem))
    oof_brand = np.zeros(len(y_brand))
    base_model = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10))])

    for tr, val in folds:
        m = copy.deepcopy(base_model)
        m.fit(X[tr], y_mem[tr]);   oof_mem[val]   = m.predict(X[val])
        m = copy.deepcopy(base_model)
        m.fit(X[tr], y_brand[tr]); oof_brand[val] = m.predict(X[val])

    return oof_mem.reshape(-1, 1), oof_brand.reshape(-1, 1)


def run_train_mode(args):
    df   = pd.read_csv(args.csv)
    feat = Path(args.feat)

    meta        = build_meta(df)
    frame_stats = np.load(feat / "frame_stats.npy")
    face_rate   = frame_stats[:, 6:7]
    y_mem       = df["memorability_score"].values
    y_brand     = df["brand_memorability"].values
    folds       = channel_kfold(df)

    # load whichever LLM cache exists
    llm_path = Path(args.llm)
    if not llm_path.exists():
        print(f"LLM cache not found at {llm_path}, using v1 keys")
        llm_keys = ["brand_prominence","emotional_valence","narrative_arc",
                    "call_to_action","information_density","novelty_surprise",
                    "visual_dynamism","brand_specificity"]
    else:
        # detect version by checking first entry's keys
        cache   = json.loads(llm_path.read_text())
        sample  = next(iter(cache.values()))
        llm_keys = list(sample.keys())
        print(f"Detected LLM keys: {llm_keys}")

    llm = load_llm(df, llm_path, llm_keys)

    # correlation report for new dimensions
    print("\nSpearman ρ — LLM scalars vs targets:")
    print(f"  {'dimension':25s}  ρ_video   ρ_brand")
    for i, k in enumerate(llm_keys):
        r1 = spearman(llm[:, i], y_mem)
        r2 = spearman(llm[:, i], y_brand)
        print(f"  {k:25s}  {r1:+.3f}     {r2:+.3f}")

    X_base = np.concatenate([meta, llm], axis=1)  # best combo so far

    # cross-target OOF features
    oof_mem, oof_brand = make_cross_target_features(X_base, y_mem, y_brand, folds)

    combos = {
        "meta+llm"              : X_base,
        "meta+llm+cross"        : np.concatenate([X_base, oof_mem, oof_brand], axis=1),
        "meta+face+llm"         : np.concatenate([meta, face_rate, llm], axis=1),
        "meta+face+llm+cross"   : np.concatenate([meta, face_rate, llm, oof_mem, oof_brand], axis=1),
    }

    for target, y in [("memorability_score", y_mem), ("brand_memorability", y_brand)]:
        print(f"\n{'═'*50}\n  TARGET: {target}\n{'═'*50}")
        for name, X in combos.items():
            _, score = tune(X, y, folds)
            print(f"  {name:30s}  ρ = {score:.4f}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",    required=True, choices=["llm", "train"])
    p.add_argument("--csv",     required=True)
    p.add_argument("--feat",    default="features")
    p.add_argument("--llm",     required=True)
    p.add_argument("--stt",     default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.mode == "llm":
        run_llm_mode(args)
    else:
        run_train_mode(args)
