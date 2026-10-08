from pathlib import Path
import csv
import cv2
import random

ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = ROOT / "dataset" / "videos"
ANN_FILE = ROOT / "annotations" / "ava_train_v2.1.csv"

OUT_IMG = ROOT / "dataset" / "images" / "train"
OUT_LBL = ROOT / "dataset" / "labels" / "train"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

# Behavior labels
# 1=stand, 2=lying, 3=foraging, 4=drinking, 5=rumination
LABEL_MAP = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
}

# Read annotations
annotations = {}

with open(ANN_FILE, "r", newline="") as f:
    reader = csv.reader(f)

    for row in reader:
        if len(row) < 8:
            continue

        try:
            video_id = str(int(row[0]))
            timestamp = int(row[1])

            x1 = float(row[2])
            y1 = float(row[3])
            x2 = float(row[4])
            y2 = float(row[5])

            action = int(row[6])

            if action not in LABEL_MAP:
                continue

            key = (video_id, timestamp)

            annotations.setdefault(key, []).append(
                (LABEL_MAP[action], x1, y1, x2, y2)
            )

        except ValueError:
            continue

print("Annotations loaded:", len(annotations))

# Find videos
videos = list(VIDEO_DIR.glob("*.mp4"))
print("Videos found:", len(videos))

processed = 0
saved = 0

for video in videos:

    video_id = video.stem

    if not video_id.isdigit():
        continue

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        print("Could not open:", video.name)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    timestamps = [
        t for (vid, t) in annotations.keys()
        if vid == video_id
    ]

    for timestamp in timestamps:

        frame_number = int(timestamp * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ok, frame = cap.read()

        if not ok:
            continue

        image_name = f"{video_id}_{timestamp}.jpg"
        label_name = f"{video_id}_{timestamp}.txt"

        image_path = OUT_IMG / image_name
        label_path = OUT_LBL / label_name

        cv2.imwrite(str(image_path), frame)

        with open(label_path, "w") as lf:

            for class_id, x1, y1, x2, y2 in annotations[(video_id, timestamp)]:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

# Keep YOLO values inside valid range
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

# Ignore invalid/empty boxes
                if w <= 0 or h <= 0:
                    continue

                lf.write( 
                f"{class_id} {cx:.6f} {cy:.6f} "
                f"{w:.6f} {h:.6f}\n"
            )

                

        saved += 1

    cap.release()

    processed += 1

    if processed % 25 == 0:
        print(
            f"Processed {processed}/{len(videos)} videos | "
            f"Frames saved: {saved}"
        )

print("\n================================")
print("CBVD PREPARATION COMPLETE")
print("Videos processed:", processed)
print("Frames saved:", saved)
print("================================")