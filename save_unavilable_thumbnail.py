import os
import cv2
import time

# Folder containing videos
video_folder = "videos"

# Folder to save frames
output_folder = "saved_frames"

# Which second to capture
target_second = 5

os.makedirs(output_folder, exist_ok=True)

video_extensions = (".mp4", ".avi", ".mov", ".mkv")
start_time = time.time()


video_files = [
    f for f in os.listdir(video_folder)
    if f.lower().endswith(video_extensions)
]

for video_name in video_files:
    video_path = os.path.join(video_folder, video_name)

    cap = cv2.VideoCapture(video_path)

    # Get FPS (frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Frame number at the target second
    frame_number = int(fps * target_second)

    # Jump to that frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    success, frame = cap.read()

    if success:
        image_name = os.path.splitext(video_name)[0] + ".jpg"
        output_path = os.path.join(output_folder, image_name)

        cv2.imwrite(output_path, frame)
        print(f"Saved: {output_path}")
    else:
        print(f"Could not read frame from: {video_name}")

    cap.release()

end_time = time.time()
print(f"Done! Time taken: {end_time - start_time:.2f} seconds")