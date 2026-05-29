"""
approach_audio.py
-----------------
Extract audio features from the pre-extracted WAV files and evaluate their
predictive signal for memorability_score and brand_memorability.

Feature groups extracted (all from the first 60s audio window):
  1. Energy / loudness   — RMS mean/std/max, dynamic range
  2. Silence             — silence ratio, avg silence run length, # silence segments
  3. Spectral            — MFCC (13 × mean+std), spectral centroid/bandwidth/rolloff mean+std
  4. Rhythm / tempo      — estimated BPM, beat strength, onset rate
  5. Harmony             — chroma mean+std (12 dims), chroma entropy
  6. Speech rate proxy   — words-per-second derived from STT + actual audio duration
  7. ZCR                 — zero-crossing rate mean+std (voice vs noise/music indicator)

CV strategy: same channel-stratified GroupKFold (5-fold) used in other approaches.
Evaluation: Spearman ρ per target, printed per fold and overall.

Usage:
    pip install librosa soundfile scikit-learn scipy pandas numpy
    python approach_audio.py
    python approach_audio.py --audio_dir audio --train_csv devset_videolist_GT.csv
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
AUDIO_DIR   = "audio"
TRAIN_CSV   = "devset_videolist_GT.csv"
TEST_CSV    = "predict/testset_videolist_.csv"
STT_DIR     = "devset-stt"

TARGETS     = ["memorability_score", "brand_memorability"]
N_FOLDS     = 5
N_MFCC      = 13
SILENCE_DB  = -40           # dB threshold for silence detection
HOP_LENGTH  = 512
SR          = 16000         # must match extract_audio.py
# ─────────────────────────────────────────────────────────────────────────────


# ── Feature extraction ────────────────────────────────────────────────────────

def rms_features(y: np.ndarray) -> dict:
    """Energy and loudness features."""
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-9)
    return {
        "rms_mean":       float(rms.mean()),
        "rms_std":        float(rms.std()),
        "rms_max":        float(rms.max()),
        "rms_db_mean":    float(rms_db.mean()),
        "rms_db_std":     float(rms_db.std()),
        "dynamic_range":  float(rms_db.max() - rms_db.min()),
    }


def silence_features(y: np.ndarray, sr: int) -> dict:
    """Silence / pause structure features."""
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-9)
    is_silent = rms_db < SILENCE_DB

    silence_ratio = float(is_silent.mean())

    # Run-length of silence frames
    runs, run_lengths = [], []
    in_run = False
    length = 0
    for s in is_silent:
        if s:
            in_run = True
            length += 1
        else:
            if in_run:
                run_lengths.append(length)
            in_run = False
            length = 0
    if in_run:
        run_lengths.append(length)

    n_silence_segments = len(run_lengths)
    avg_silence_run    = float(np.mean(run_lengths)) if run_lengths else 0.0
    max_silence_run    = float(np.max(run_lengths))  if run_lengths else 0.0

    return {
        "silence_ratio":        silence_ratio,
        "n_silence_segments":   float(n_silence_segments),
        "avg_silence_run":      avg_silence_run,
        "max_silence_run":      max_silence_run,
    }


def spectral_features(y: np.ndarray, sr: int) -> dict:
    """MFCC + spectral shape features."""
    feats = {}

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc{i+1}_std"]  = float(mfcc[i].std())

    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    feats["spec_centroid_mean"] = float(centroid.mean())
    feats["spec_centroid_std"]  = float(centroid.std())

    # Spectral bandwidth
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    feats["spec_bw_mean"] = float(bw.mean())
    feats["spec_bw_std"]  = float(bw.std())

    # Spectral rolloff (energy concentration)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    feats["spec_rolloff_mean"] = float(rolloff.mean())
    feats["spec_rolloff_std"]  = float(rolloff.std())

    # Spectral contrast (music vs speech discriminator)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
    feats["spec_contrast_mean"] = float(contrast.mean())
    feats["spec_contrast_std"]  = float(contrast.std())

    return feats


def rhythm_features(y: np.ndarray, sr: int) -> dict:
    """Tempo and onset features."""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset_env    = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                              hop_length=HOP_LENGTH)
    duration_s   = len(y) / sr
    onset_rate   = len(onset_frames) / max(duration_s, 1)

    return {
        "tempo":         float(tempo),
        "beat_strength": float(onset_env[beats].mean()) if len(beats) > 0 else 0.0,
        "onset_rate":    float(onset_rate),
    }


def harmony_features(y: np.ndarray, sr: int) -> dict:
    """Chroma features — harmonic/tonal content (music presence proxy)."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    feats  = {}
    for i in range(12):
        feats[f"chroma{i}_mean"] = float(chroma[i].mean())
        feats[f"chroma{i}_std"]  = float(chroma[i].std())

    # Chroma entropy: flat = atonal/noise/speech, peaked = music
    chroma_mean = chroma.mean(axis=1) + 1e-9
    chroma_mean /= chroma_mean.sum()
    chroma_entropy = float(-np.sum(chroma_mean * np.log(chroma_mean)))
    feats["chroma_entropy"] = chroma_entropy

    return feats


def zcr_features(y: np.ndarray) -> dict:
    """Zero-crossing rate — voice activity / texture indicator."""
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    return {
        "zcr_mean": float(zcr.mean()),
        "zcr_std":  float(zcr.std()),
    }


def speech_rate_feature(video_id: str, audio_duration_s: float, stt_dir: str) -> dict:
    """Words-per-second from STT transcript over the actual audio duration."""
    stt_path = Path(stt_dir) / f"{video_id}.txt"
    if stt_path.exists():
        text  = stt_path.read_text(encoding="utf-8", errors="ignore").strip()
        words = len(text.split())
    else:
        words = 0
    wps = words / max(audio_duration_s, 1.0)
    return {"words_per_second": float(wps), "stt_word_count": float(words)}


def extract_features_for_video(audio_path: Path, video_id: str,
                                stt_dir: str) -> dict | None:
    """Load WAV and extract all feature groups. Returns None on failure."""
    try:
        y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
        if len(y) == 0:
            return None
        duration_s = len(y) / sr
    except Exception as e:
        print(f"  [WARN] Could not load {audio_path}: {e}")
        return None

    feats = {"id": video_id, "audio_duration_s": duration_s}
    feats.update(rms_features(y))
    feats.update(silence_features(y, sr))
    feats.update(spectral_features(y, sr))
    feats.update(rhythm_features(y, sr))
    feats.update(harmony_features(y, sr))
    feats.update(zcr_features(y))
    feats.update(speech_rate_feature(video_id, duration_s, stt_dir))
    return feats


# ── CV utilities ──────────────────────────────────────────────────────────────

def channel_stratified_kfold(df: pd.DataFrame, n_splits: int = 5):
    """
    Greedy bin-packing GroupKFold by channel size.
    Goldman Sachs (largest channel) gets its own dedicated fold.
    Returns list of (train_idx, val_idx) arrays.
    """
    channel_sizes = df.groupby("channelName").size().sort_values(ascending=False)
    fold_assignment = {}
    fold_sizes = [0] * n_splits

    for ch, size in channel_sizes.items():
        # Assign to least-loaded fold
        target_fold = int(np.argmin(fold_sizes))
        fold_assignment[ch] = target_fold
        fold_sizes[target_fold] += size

    fold_col = df["channelName"].map(fold_assignment).values
    splits = []
    for fold in range(n_splits):
        val_mask   = fold_col == fold
        train_mask = ~val_mask
        splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return splits


def evaluate_features(X: np.ndarray, y: np.ndarray,
                       groups: np.ndarray, target_name: str,
                       model_name: str, model) -> float:
    """Run channel-stratified CV and return mean Spearman ρ."""
    splits = channel_stratified_kfold(
        pd.DataFrame({"channelName": groups}), n_splits=N_FOLDS
    )
    rhos = []
    for fold_i, (tr, va) in enumerate(splits):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        # Replace any NaN/inf
        X_tr_s = np.nan_to_num(X_tr_s)
        X_va_s = np.nan_to_num(X_va_s)

        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_va_s)
        rho, _ = spearmanr(y_va, preds)
        rhos.append(rho if not np.isnan(rho) else 0.0)
        print(f"    Fold {fold_i+1}: ρ = {rho:.4f}  (n_val={len(va)})")

    mean_rho = float(np.mean(rhos))
    print(f"  → {model_name} | {target_name}: mean ρ = {mean_rho:.4f}")
    return mean_rho


def individual_feature_correlations(X: np.ndarray, feature_names: list,
                                     y: np.ndarray, target_name: str,
                                     top_n: int = 15):
    """Print top-N individual feature Spearman correlations with target."""
    corrs = []
    for i, fname in enumerate(feature_names):
        rho, _ = spearmanr(X[:, i], y)
        corrs.append((fname, rho if not np.isnan(rho) else 0.0))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Top-{top_n} individual correlations with {target_name}:")
    for fname, rho in corrs[:top_n]:
        bar = "█" * int(abs(rho) * 30)
        sign = "+" if rho > 0 else "-"
        print(f"    {fname:<35} {sign}{abs(rho):.4f}  {bar}")
    return corrs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audio feature approach for memorability prediction")
    parser.add_argument("--audio_dir",  default=AUDIO_DIR)
    parser.add_argument("--train_csv",  default=TRAIN_CSV)
    parser.add_argument("--stt_dir",    default=STT_DIR)
    parser.add_argument("--cache",      default="features/audio_features.json",
                        help="Cache extracted features to avoid re-computation")
    parser.add_argument("--no_cache",   action="store_true",
                        help="Ignore existing cache and re-extract")
    args = parser.parse_args()

    audio_dir  = Path(args.audio_dir)
    cache_path = Path(args.cache)

    # ── Load train metadata ──────────────────────────────────────────────────
    df = pd.read_csv(args.train_csv)
    print(f"[INFO] Training set: {len(df)} videos")

    # ── Extract or load cached audio features ────────────────────────────────
    if cache_path.exists() and not args.no_cache:
        print(f"[INFO] Loading cached audio features from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        feat_list = list(cached.values())
    else:
        print(f"[INFO] Extracting audio features from {audio_dir} ...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        feat_list = []
        cached    = {}
        missing   = 0

        for _, row in df.iterrows():
            vid_id    = str(row["id"])
            wav_path  = audio_dir / f"{vid_id}.wav"

            if not wav_path.exists():
                missing += 1
                continue

            print(f"  {vid_id} ...", end=" ", flush=True)
            feats = extract_features_for_video(wav_path, vid_id, args.stt_dir)
            if feats is None:
                print("FAILED")
                missing += 1
                continue

            feat_list.append(feats)
            cached[vid_id] = feats
            print(f"OK ({feats['audio_duration_s']:.1f}s)")

        with open(cache_path, "w") as f:
            json.dump(cached, f)
        print(f"\n[INFO] Extracted {len(feat_list)} / {len(df)} videos "
              f"({missing} missing WAV). Cached → {cache_path}")

    if len(feat_list) == 0:
        print("[ERROR] No audio features extracted. Run extract_audio.py first.")
        return

    # ── Build feature matrix ─────────────────────────────────────────────────
    feat_df = pd.DataFrame(feat_list).set_index("id")

    # Align with train labels
    merged = df.set_index("id").join(feat_df, how="inner")
    print(f"[INFO] Videos with audio features + labels: {len(merged)}")

    if len(merged) < 50:
        print("[WARN] Very few matched videos — check that WAV filenames match CSV 'id' column.")

    # Feature columns (everything that isn't metadata/targets)
    meta_cols   = ["video_id", "channelName", "title", "description", "tags",
                   "durationSeconds", "categoryName", "viewsCount", "likesCount",
                   "commentsCount", "engagementRate", "nb_annotations",
                   "memorability_score", "brand_memorability", "url", "split"]
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]

    X      = merged[feature_cols].values.astype(float)
    groups = merged["channelName"].values

    print(f"[INFO] Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

    # ── Evaluate per target ──────────────────────────────────────────────────
    results = {}
    for target in TARGETS:
        if target not in merged.columns:
            print(f"[WARN] Target {target} not found in data — skipping")
            continue

        y = merged[target].values.astype(float)
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")

        # Individual feature correlations
        corrs = individual_feature_correlations(X, feature_cols, y, target)

        # Filter features with |ρ| > threshold for model input
        THRESHOLD = 0.05
        top_feat_idx = [i for i, (_, r) in enumerate(
            sorted(enumerate(corrs), key=lambda x: abs(x[1][1]), reverse=True)
        ) if abs(r) > THRESHOLD]

        if len(top_feat_idx) == 0:
            print(f"  [WARN] No features above threshold {THRESHOLD} — using all")
            top_feat_idx = list(range(len(feature_cols)))

        X_filtered = X[:, [corrs[i][0] if isinstance(corrs[i][0], int)
                           else feature_cols.index(corrs[i][0])
                           for i in range(len(top_feat_idx))]]
        # Simpler: just use all features and let regularisation handle it
        X_use = X

        print(f"\n  [Ridge regression — all audio features]")
        for alpha in [10, 100, 500]:
            print(f"\n  alpha={alpha}:")
            rho = evaluate_features(X_use, y, groups, target,
                                    f"Ridge(α={alpha})", Ridge(alpha=alpha))
            results[f"{target}_ridge_{alpha}"] = rho

        print(f"\n  [SVR — all audio features]")
        for C in [0.1, 0.5, 1.0]:
            print(f"\n  C={C}:")
            rho = evaluate_features(X_use, y, groups, target,
                                    f"SVR(C={C})", SVR(C=C, kernel="rbf"))
            results[f"{target}_svr_{C}"] = rho

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY — Best CV ρ per target")
    print(f"{'='*60}")
    for target in TARGETS:
        best_key = max(
            [k for k in results if k.startswith(target)],
            key=lambda k: results[k],
            default=None
        )
        if best_key:
            model_label = best_key[len(target)+1:]
            print(f"  {target:<28} best={results[best_key]:.4f}  ({model_label})")

    print(f"\n  [Reference] TF-IDF best:")
    print(f"  memorability_score       ρ = 0.2697")
    print(f"  brand_memorability       ρ = 0.1801")

    return results


if __name__ == "__main__":
    main()
