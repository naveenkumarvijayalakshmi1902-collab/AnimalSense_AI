import csv
import cv2
from ultralytics import YOLO

# =========================
# CONFIGURATION
# =========================

MODEL_PATH = r"D:\AnimalSense_AI\runs\detect\runs\comparison\baseline_10ep\weights\best.pt"
VIDEO_PATH = r"D:\AnimalSense_AI\dataset\videos\618.mp4"

OUTPUT_VIDEO = r"D:\AnimalSense_AI\outputs\final_tracked_video.mp4"
OUTPUT_CSV = r"D:\AnimalSense_AI\outputs\final_behavior_history.csv"

CONFIDENCE = 0.25
IMG_SIZE = 320
DEVICE = 0

# =========================
# LOAD MODEL
# =========================

print("Loading YOLOv8 baseline...")
model = YOLO(MODEL_PATH)

# =========================
# OPEN VIDEO
# =========================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height}")
print(f"FPS: {fps}")
print(f"Frames: {total_frames}")
print(f"Duration: {total_frames / fps:.2f} seconds")

# =========================
# VIDEO WRITER
# =========================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

# =========================
# CSV
# =========================

csv_file = open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "timestamp_sec",
    "track_id",
    "behavior",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2"
])

# =========================
# TRACKING
# =========================

frame_number = 0

print()
print("Starting continuous tracking...")
print()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.track(
        frame,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )

    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [-1] * len(xyxy)

        for box, confidence, cls, track_id in zip(
            xyxy,
            confidences,
            classes,
            track_ids
        ):

            x1, y1, x2, y2 = map(int, box)

            behavior = model.names[int(cls)]

            timestamp = frame_number / fps

            # CSV
            csv_writer.writerow([
                frame_number,
                round(timestamp, 3),
                track_id,
                behavior,
                round(float(confidence), 4),
                x1,
                y1,
                x2,
                y2
            ])

            # Visualization
            label = (
                f"Track {track_id} | "
                f"{behavior} | "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    # Frame counter
    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    writer.write(frame)

    if frame_number % 25 == 0:
        print(
            f"Processed {frame_number}/{total_frames} "
            f"({100 * frame_number / total_frames:.1f}%)"
        )

# =========================
# CLEANUP
# =========================

cap.release()
writer.release()
csv_file.close()

print()
print("===================================")
print("FINAL TRACKING COMPLETED")
print("===================================")
print()
print(f"Video: {OUTPUT_VIDEO}")
print(f"CSV:   {OUTPUT_CSV}")