import os
import time
import random
import requests
import pandas as pd
from pathlib import Path

CSV_FILE = "devset_videolist_GT.csv"
THUMBNAIL_DIR = "train"
Path(THUMBNAIL_DIR).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_FILE)

session = requests.Session()
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


available_videos = []
unavailable_videos = []

# -----------------------------
# DOWNLOAD HQ THUMBNAIL ONLY
# -----------------------------
def download_thumbnail(video_id):
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    save_path = os.path.join(
        THUMBNAIL_DIR,
        f"{video_id}.jpg"
    )

    try:
        response = session.get(
            thumbnail_url,
            headers=HEADERS,
            timeout=10
        )

        if response.status_code != 200:
            return False

        # Invalid videos usually return
        # a tiny placeholder image
        if len(response.content) < 2000:
            return False

        with open(save_path, "wb") as f:
            f.write(response.content)

        return True

    except Exception as e:
        print(f"Error downloading {video_id}: {e}")
        return False


# -----------------------------
# MAIN LOOP
# -----------------------------
for idx, row in df.iterrows():
    video_id = str(row["id"]).strip()
    print(f"[{idx+1}/{len(df)}] Processing: {video_id}")
    success = download_thumbnail(video_id)

    if success:
        available_videos.append(video_id)
        print("  -> Downloaded")
    else:
        unavailable_videos.append(video_id)
        print("  -> Unavailable")
        
    # Reduce chances of rate limiting
    time.sleep(random.uniform(0.5, 1.5))


# -----------------------------
# SAVE RESULTS
# -----------------------------
pd.DataFrame({
    "train_available": available_videos
}).to_csv(
    "train_available.csv",
    index=False
)

pd.DataFrame({
    "train_unavailable": unavailable_videos
}).to_csv(
    "train_unavailable.csv",
    index=False
)

# -----------------------------
# SUMMARY
# -----------------------------
print("\n===== SUMMARY =====")
print(f"Available thumbnails: {len(available_videos)}")
print(f"Unavailable thumbnails: {len(unavailable_videos)}")
print(f"Saved in folder: {THUMBNAIL_DIR}")