import os
import cv2
import time
import pandas as pd

# ----------------------------
# Configuration
# ----------------------------
video_folder = "videos"
output_folder = "thumbnails/train_full"
csv_path = "thumbnails/train_unavailable.csv"
target_second = 5
output_size = (480, 360)  # width x height

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

start_time = time.time()

# ----------------------------
# Load video list from CSV
# ----------------------------
video_names = pd.read_csv(csv_path)["train_unavailable"].tolist()
possible_exts = [".mp4", ".mkv", ".webm", ".avi"]

# ----------------------------
# Process each video
# ----------------------------
for video_name in video_names:
    # Find actual video file with extension
    video_path = None
    for ext in possible_exts:
        path = os.path.join(video_folder, video_name + ext)
        if os.path.exists(path):
            video_path = path
            break

    if video_path is None:
        print(f"File not found: {video_name}")
        continue
    # print(f"Processing: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        continue

    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Invalid FPS for: {video_name}")
        cap.release()
        continue

    # Compute frame at target second
    frame_number = int(fps * target_second)

    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()
    if success:
        # Resize frame to 480x360
        frame = cv2.resize(frame, output_size)

        # Save image
        image_name = video_name + ".jpg"
        output_path = os.path.join(output_folder, image_name)

        cv2.imwrite(output_path, frame)
        print(f"Saved: {output_path}")
    else:
        print(f"Could not read frame: {video_name}")

    cap.release()

# ----------------------------
# Done
# ----------------------------
end_time = time.time()
print(f"\nDone! Time taken: {end_time - start_time:.2f} seconds")