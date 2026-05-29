"""
extract_audio.py
----------------
Extract audio from raw commercial videos.
- If video duration >= 60s → extract first 60 seconds (matching annotation window)
- If video duration  < 60s → extract full audio
Output: WAV mono 16kHz files saved to AUDIO_DIR/{id}.wav

Usage:
    python extract_audio.py
    python extract_audio.py --video_dir /path/to/videos --audio_dir /path/to/audio
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_VIDEO_DIR  = "videos"          # directory containing raw .mp4/.mkv/etc.
DEFAULT_AUDIO_DIR  = "audio"           # output directory for .wav files
TRAIN_CSV          = "devset_videolist_GT.csv"
TEST_CSV           = "predict/testset_videolist_.csv"
ANNOTATION_WINDOW  = 60               # seconds annotators actually watched
SAMPLE_RATE        = 16000            # Hz — good for speech analysis
# ─────────────────────────────────────────────────────────────────────────────


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] ffmpeg not found. Install with: sudo apt install ffmpeg")
        sys.exit(1)


def get_extract_duration(row_duration_seconds: float) -> float:
    """Return how many seconds of audio to extract."""
    return min(float(row_duration_seconds), ANNOTATION_WINDOW)


def extract_audio(video_path: Path, output_path: Path, duration_seconds: float) -> bool:
    """
    Extract audio from video_path using ffmpeg.
    - Mono, 16kHz WAV
    - First `duration_seconds` seconds only
    Returns True on success, False on failure.
    """
    cmd = [
        "ffmpeg",
        "-y",                            # overwrite output if exists
        "-i", str(video_path),
        "-t", str(duration_seconds),     # duration limit
        "-vn",                           # no video
        "-ac", "1",                      # mono
        "-ar", str(SAMPLE_RATE),         # sample rate
        "-acodec", "pcm_s16le",          # uncompressed WAV
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return True


def find_video_file(video_dir: Path, video_id: str) -> Path | None:
    """Look for any video file matching the YouTube ID in video_dir."""
    for ext in [".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv"]:
        candidate = video_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
    # Fallback: glob for partial matches (some tools add suffixes)
    matches = list(video_dir.glob(f"{video_id}*"))
    matches = [m for m in matches if m.suffix.lower()
               in {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv"}]
    return matches[0] if matches else None


def load_all_ids(train_csv: str, test_csv: str) -> pd.DataFrame:
    """Load video IDs and durations from both train and test CSVs."""
    dfs = []
    for path, split in [(train_csv, "train"), (test_csv, "test")]:
        if not os.path.exists(path):
            print(f"[WARN] CSV not found: {path} — skipping")
            continue
        df = pd.read_csv(path)
        df["split"] = split
        dfs.append(df)
    if not dfs:
        print("[ERROR] No CSVs found.")
        sys.exit(1)
    combined = pd.concat(dfs, ignore_index=True)

    # Normalise column names — handle slight variations
    if "id" not in combined.columns:
        # Try common alternatives
        for col in ["video_id", "youtube_id", "ID"]:
            if col in combined.columns:
                combined = combined.rename(columns={col: "id"})
                break

    return combined


def main():
    parser = argparse.ArgumentParser(description="Extract first-60s audio from commercial videos")
    parser.add_argument("--video_dir",  default=DEFAULT_VIDEO_DIR,
                        help="Directory containing raw video files")
    parser.add_argument("--audio_dir",  default=DEFAULT_AUDIO_DIR,
                        help="Output directory for .wav files")
    parser.add_argument("--train_csv",  default=TRAIN_CSV)
    parser.add_argument("--test_csv",   default=TEST_CSV)
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-extract even if .wav already exists")
    args = parser.parse_args()

    check_ffmpeg()

    video_dir = Path(args.video_dir)
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.exists():
        print(f"[ERROR] Video directory not found: {video_dir}")
        sys.exit(1)

    # Load metadata
    df = load_all_ids(args.train_csv, args.test_csv)
    print(f"[INFO] Total videos in metadata: {len(df)} "
          f"({(df['split']=='train').sum()} train, {(df['split']=='test').sum()} test)")

    stats = {"success": 0, "skipped": 0, "missing_video": 0, "failed": 0}

    for _, row in df.iterrows():
        vid_id = str(row["id"])
        duration = float(row.get("durationSeconds", 60))
        extract_dur = get_extract_duration(duration)

        output_wav = audio_dir / f"{vid_id}.wav"

        # Skip if already extracted and not overwriting
        if output_wav.exists() and not args.overwrite:
            stats["skipped"] += 1
            continue

        # Locate video file
        video_file = find_video_file(video_dir, vid_id)
        if video_file is None:
            print(f"[MISS] {vid_id} — video file not found in {video_dir}")
            stats["missing_video"] += 1
            continue

        # Extract
        label = f"{extract_dur:.0f}s" if duration >= ANNOTATION_WINDOW else f"{duration:.0f}s (full)"
        print(f"[EXTRACT] {vid_id} | video={duration:.0f}s → extracting {label} ...", end=" ")

        ok = extract_audio(video_file, output_wav, extract_dur)
        if ok:
            size_kb = output_wav.stat().st_size / 1024
            print(f"OK ({size_kb:.0f} KB)")
            stats["success"] += 1
        else:
            print("FAILED")
            stats["failed"] += 1

    print("\n── Summary ──────────────────────────────")
    print(f"  Extracted     : {stats['success']}")
    print(f"  Skipped       : {stats['skipped']} (already exist)")
    print(f"  Missing video : {stats['missing_video']}")
    print(f"  Failed        : {stats['failed']}")
    print(f"  WAV files in  : {audio_dir.resolve()}")


if __name__ == "__main__":
    main()
