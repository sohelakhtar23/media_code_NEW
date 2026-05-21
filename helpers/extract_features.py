import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from tqdm import tqdm
import re

MAX_FRAMES = 60
CSV_PATH = "devset_videolist_GT.csv"
FRAMES_ROOT = "frames/"
OUT_DIR = Path("features/")
# if output dir doesn't exist, create it
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
df   = pd.read_csv(CSV_PATH)

def extract_visual():
    """
    Extract CLIP visual features from the first 60 frames of each video.
    Saves a (N, D) numpy array per video, then aggregates to a single feature matrix.

    Output:
        features/visual_features.npy   shape (339, 768*4) — mean/std/first/last pooling
        features/video_ids.npy         matching video id order
    """
    
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    feat_dim  = 768  # ViT-L/14 embedding dim
    all_feats = []
    all_ids   = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        vid_id    = row["id"]
        frame_dir = Path(FRAMES_ROOT) / vid_id

        if not frame_dir.exists():
            print(f"  [WARN] missing frames dir: {vid_id}")
            all_feats.append(np.zeros(feat_dim * 4))
            all_ids.append(vid_id)
            continue

        # sorted frames → take first MAX_FRAMES
        frame_files = sorted(frame_dir.iterdir())[:MAX_FRAMES]
        if len(frame_files) == 0:
            all_feats.append(np.zeros(feat_dim * 4))
            all_ids.append(vid_id)
            continue

        # batch inference
        imgs = []
        for fp in frame_files:
            try:
                imgs.append(preprocess(Image.open(fp).convert("RGB")))
            except Exception:
                pass

        if len(imgs) == 0:
            all_feats.append(np.zeros(feat_dim * 4))
            all_ids.append(vid_id)
            continue

        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            embs = model.encode_image(batch).float().cpu().numpy()  # (T, 768)
            # images from CPU → GPU → GPU compute → CPU → NumPy/Pandas/etc
            # we move back to CPU, since .numpy() only works on CPU tensors

        all_feats.append(pool_embeddings(embs))
        all_ids.append(vid_id)

    X = np.array(all_feats)  # (N, 4*768)
    print(f"Visual feature matrix shape: {X.shape}")

    np.save(OUT_DIR / "visual_features.npy", X)
    np.save(OUT_DIR / "video_ids.npy",       np.array(all_ids))
    print(f"Saved to {OUT_DIR}/")

def pool_embeddings(embs: np.ndarray) -> np.ndarray:
    """embs: (T, D) → (4*D,) via mean, std, first, last"""
    return np.concatenate([
        embs.mean(axis=0),
        embs.std(axis=0),
        embs[0],
        embs[-1],
    ])


def extract_text_meta():
    """
    Extract CLIP text features from video title + description.
    Saves a (N, D) numpy array, and a list of feature names.

    Output:
        features/text_features.npy    CLIP text embeddings (title + description)
        features/meta_features.npy    Numeric metadata features
        features/meta_feature_names.txt
    """
    model, _ = clip.load("ViT-L/14", device=device)
    model.eval()

    # combine title + truncated description
    texts = []
    for _, row in df.iterrows():
        title = clean_text(row.get("title", ""), max_words=20)
        desc  = clean_text(row.get("description", ""), max_words=40)
        texts.append(f"{title} {desc}".strip())

    print("Encoding text with CLIP...")
    text_embs = encode_texts(texts, model, device)  # (N, 768)
    print(f"Text feature matrix shape: {text_embs.shape}")
    np.save(OUT_DIR / "text_features.npy", text_embs)

    # ── Metadata features ─────────────────────────────────────────────────────
    meta, names = build_meta_features(df)
    print(f"Meta feature matrix shape: {meta.shape}")
    print(f"Features: {names}")
    np.save(OUT_DIR / "meta_features.npy", meta)
    Path(OUT_DIR / "meta_feature_names.txt").write_text("\n".join(names))

    print(f"Saved to {OUT_DIR}/")



# ── text helpers ──────────────────────────────────────────────────────────────
def clean_text(s: str, max_words: int = 60) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r'\s+', ' ', s).strip()
    return " ".join(s.split()[:max_words])  # CLIP token limit ~77

def encode_texts(texts, model, device, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(device)
        with torch.no_grad():
            embs = model.encode_text(tokens).float().cpu().numpy()
        all_embs.append(embs)
    return np.concatenate(all_embs, axis=0)

# ── metadata helpers ──────────────────────────────────────────────────────────
def build_meta_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    # target-encode channelName using leave-one-out mean
    ch_mem   = df.groupby("channelName")["memorability_score"].mean()
    ch_brand = df.groupby("channelName")["brand_memorability"].mean()

    feat = pd.DataFrame()
    feat["log_views"]       = np.log1p(df["viewsCount"])
    feat["log_likes"]       = np.log1p(df["likesCount"])
    feat["log_dislikes"]    = np.log1p(df["dislikesCount"])
    feat["log_comments"]    = np.log1p(df["commentsCount"])
    feat["log_engagements"] = np.log1p(df["engagementsCount"])
    feat["engagement_rate"] = df["engagementRate"]
    feat["log_duration"]    = np.log1p(df["durationSeconds"])
    feat["is_long"]         = (df["durationSeconds"] > 60).astype(float)
    feat["nb_annotations"]  = df["nb_annotations"]
    feat["channel_mem_enc"] = df["channelName"].map(ch_mem).fillna(ch_mem.mean())
    feat["channel_brand_enc"] = df["channelName"].map(ch_brand).fillna(ch_brand.mean())

    # category one-hot (only categories appearing ≥ 5 times)
    cat_counts = df["categoryName"].value_counts()
    keep_cats  = cat_counts[cat_counts >= 5].index.tolist()
    for cat in keep_cats:
        feat[f"cat_{cat.replace(' ', '_').replace('&','n')}"] = (df["categoryName"] == cat).astype(float)

    names = feat.columns.tolist()
    return feat.values.astype(np.float32), names



def extract_frame_stats():
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
    """
    
    FEATURE_NAMES = [
        "brightness_mean", "brightness_std",
        "saturation_mean", "saturation_std",
        "colorfulness_mean", "colorfulness_std",
        "face_rate",
        "motion_mean", "motion_std",
        "shot_cut_rate",
    ]
    all_feats = []
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    CUT_THRESHOLD = 30.0  # mean absolute pixel diff to call a shot cut
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
    for _, row in tqdm(df.iterrows(), total=len(df), desc="frame stats"):
        frame_dir = Path(FRAMES_ROOT) / row["id"]
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

    np.save(OUT_DIR / "frame_stats.npy", X)
    Path(OUT_DIR / "frame_stats_names.txt").write_text("\n".join(FEATURE_NAMES))
    print(f"\nSaved to {OUT_DIR}/frame_stats.npy")



# 1.
# extract_visual() 
# extract_text_meta()

# 3.
extract_frame_stats()