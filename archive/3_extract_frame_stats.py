"""
Extract low-level visual statistics from first 60 frames using OpenCV.
No GPU needed. Fast.

Features per video:
  - brightness mean/std
  - saturation mean/std  
  - colorfulness (Hasler & Susstrunk metric)
  - face detection rate (fraction of frames with ≥1 face)
  - shot cut rate (fraction of consecutive frames with large diff)
  - visual diversity (mean frame-to-frame pixel difference)

Usage:
    python extract_frame_stats.py \
        --csv    devset_videolist_GT.csv \
        --frames frames/ \
        --out    features/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

MAX_FRAMES   = 60
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
CUT_THRESHOLD = 30.0  # mean absolute pixel diff to call a shot cut

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",    required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--out",    default="features")
    return p.parse_args()


def colorfulness(img_bgr):
    """Hasler & Susstrunk (2003) colorfulness metric."""
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    rg  = np.abs(r - g)
    yb  = np.abs(0.5 * (r + g) - b)
    return np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)


def process_video(frame_dir: Path) -> np.ndarray:
    frame_files = sorted(frame_dir.iterdir())[:MAX_FRAMES]
    if not frame_files:
        return np.zeros(10)

    brightness_vals, sat_vals, color_vals = [], [], []
    face_counts = []
    prev_gray   = None
    diffs       = []

    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is None:
            continue

        # resize to fixed size for speed
        img = cv2.resize(img, (160, 90))

        # brightness & saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        brightness_vals.append(hsv[:, :, 2].mean() / 255.0)
        sat_vals.append(hsv[:, :, 1].mean() / 255.0)

        # colorfulness
        color_vals.append(colorfulness(img))

        # face detection (on small gray image)
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
        face_counts.append(1 if len(faces) > 0 else 0)

        # shot cut: mean abs diff to previous frame
        if prev_gray is not None:
            diff = np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)).mean()
            diffs.append(diff)
        prev_gray = gray

    if not brightness_vals:
        return np.zeros(10)

    b  = np.array(brightness_vals)
    s  = np.array(sat_vals)
    c  = np.array(color_vals)
    d  = np.array(diffs) if diffs else np.array([0.0])

    feat = np.array([
        b.mean(),                                   # avg brightness
        b.std(),                                    # brightness variation
        s.mean(),                                   # avg saturation
        s.std(),                                    # saturation variation
        c.mean(),                                   # avg colorfulness
        c.std(),                                    # colorfulness variation
        np.mean(face_counts),                       # face detection rate
        d.mean(),                                   # avg frame-to-frame diff (motion)
        d.std(),                                    # motion variation
        (d > CUT_THRESHOLD).mean(),                 # shot cut rate
    ], dtype=np.float32)

    return feat


FEATURE_NAMES = [
    "brightness_mean", "brightness_std",
    "saturation_mean", "saturation_std",
    "colorfulness_mean", "colorfulness_std",
    "face_rate",
    "motion_mean", "motion_std",
    "shot_cut_rate",
]


def main():
    args = get_args()
    out  = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df   = pd.read_csv(args.csv)

    all_feats = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="frame stats"):
        frame_dir = Path(args.frames) / row["id"]
        if frame_dir.exists():
            feat = process_video(frame_dir)
        else:
            feat = np.zeros(10)
        all_feats.append(feat)

    X = np.array(all_feats)
    print(f"Frame stats shape: {X.shape}")

    # quick correlation report
    from scipy.stats import spearmanr
    print("\nSpearman ρ with targets:")
    print(f"  {'feature':25s}  ρ_video   ρ_brand")
    for i, name in enumerate(FEATURE_NAMES):
        r1 = spearmanr(X[:, i], df["memorability_score"]).statistic
        r2 = spearmanr(X[:, i], df["brand_memorability"]).statistic
        print(f"  {name:25s}  {r1:+.3f}     {r2:+.3f}")

    np.save(out / "frame_stats.npy", X)
    Path(out / "frame_stats_names.txt").write_text("\n".join(FEATURE_NAMES))
    print(f"\nSaved to {out}/frame_stats.npy")


if __name__ == "__main__":
    main()
