from ultralytics import YOLO
import cv2
import csv
from pathlib import Path

MODEL = r"D:\AnimalSense_AI\runs\detect\runs\enhanced\cbam_full-3\weights\best.pt"
VIDEO = r"D:\AnimalSense_AI\dataset\videos\618.mp4"

OUTPUT_VIDEO = r"D:\AnimalSense_AI\outputs\cbam_tracked.mp4"
OUTPUT_LOG = r"D:\AnimalSense_AI\outputs\cbam_behavior_log.csv"

model = YOLO(MODEL)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

Path(OUTPUT_LOG).parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_LOG, "w", newline="", encoding="utf-8") as f:

    csv_writer = csv.writer(f)

    csv_writer.writerow([
        "frame",
        "cow_id",
        "behavior",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2"
    ])

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model.track(
            frame,
            imgsz=320,
            conf=0.25,
            device=0,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )

        result = results[0]

        if result.boxes is not None:

            boxes = result.boxes

            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()

            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [-1] * len(xyxy)

            for box, class_id, confidence, cow_id in zip(
                xyxy, cls, conf, track_ids
            ):

                x1, y1, x2, y2 = map(int, box)

                class_id = int(class_id)
                cow_id = int(cow_id)

                behavior = model.names[class_id]

                csv_writer.writerow([
                    frame_number,
                    cow_id,
                    behavior,
                    round(float(confidence), 4),
                    x1,
                    y1,
                    x2,
                    y2
                ])

                label = f"Cow #{cow_id} | {behavior} | {confidence:.2f}"

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
                    0.5,
                    (0, 255, 0),
                    2
                )

        writer.write(frame)

        frame_number += 1

        if frame_number % 50 == 0:
            print(f"Processed {frame_number} frames...")

cap.release()
writer.release()

print()
print("CBAM tracking completed successfully.")
print(f"Frames processed: {frame_number}")
print(f"Tracking video: {OUTPUT_VIDEO}")
print(f"Behavior log: {OUTPUT_LOG}")