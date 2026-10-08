import csv
from pathlib import Path
from math import hypot

from ultralytics import YOLO


MODEL = r"D:\AnimalSense_AI\runs\detect\runs\enhanced\cbam_full-3\weights\best.pt"
VIDEO = r"D:\AnimalSense_AI\dataset\videos\618.mp4"

OUTPUT_VIDEO = r"D:\AnimalSense_AI\outputs\stable_cow_tracking.mp4"
OUTPUT_LOG = r"D:\AnimalSense_AI\outputs\stable_cow_behavior_log.csv"


# Maximum distance, in pixels, that a detection can move
# between consecutive frames and still be considered
# the same temporary cow track.
MAX_DISTANCE = 180

# Number of frames a track can disappear before being removed.
MAX_MISSING = 15


model = YOLO(MODEL)

import cv2

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

Path(OUTPUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)


# Our own stable temporal tracks
tracks = {}

next_cow_id = 1
frame_number = 0


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

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model.predict(
            frame,
            imgsz=320,
            conf=0.25,
            device=0,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is not None:

            boxes = result.boxes

            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()

            for box, class_id, confidence in zip(
                xyxy,
                classes,
                confidences
            ):

                x1, y1, x2, y2 = map(int, box)

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                behavior = model.names[int(class_id)]

                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": cx,
                    "cy": cy,
                    "behavior": behavior,
                    "confidence": float(confidence)
                })


        # Mark existing tracks as unmatched
        matched_tracks = set()

        # Match each detection to the closest existing track
        for detection in detections:

            best_id = None
            best_distance = MAX_DISTANCE

            for cow_id, track in tracks.items():

                if cow_id in matched_tracks:
                    continue

                distance = hypot(
                    detection["cx"] - track["cx"],
                    detection["cy"] - track["cy"]
                )

                if distance < best_distance:
                    best_distance = distance
                    best_id = cow_id


            # Existing cow
            if best_id is not None:

                cow_id = best_id

                tracks[cow_id]["cx"] = detection["cx"]
                tracks[cow_id]["cy"] = detection["cy"]
                tracks[cow_id]["missing"] = 0

                matched_tracks.add(cow_id)

            # New cow track
            else:

                cow_id = next_cow_id
                next_cow_id += 1

                tracks[cow_id] = {
                    "cx": detection["cx"],
                    "cy": detection["cy"],
                    "missing": 0
                }

                matched_tracks.add(cow_id)


            detection["cow_id"] = cow_id


        # Increase missing counter for unmatched tracks
        for cow_id in list(tracks.keys()):

            if cow_id not in matched_tracks:

                tracks[cow_id]["missing"] += 1

                if tracks[cow_id]["missing"] > MAX_MISSING:
                    del tracks[cow_id]


        # Write detections
        for detection in detections:

            cow_id = detection["cow_id"]

            csv_writer.writerow([
                frame_number,
                cow_id,
                detection["behavior"],
                round(detection["confidence"], 4),
                detection["x1"],
                detection["y1"],
                detection["x2"],
                detection["y2"]
            ])


            # Draw bounding box
            cv2.rectangle(
                frame,
                (
                    detection["x1"],
                    detection["y1"]
                ),
                (
                    detection["x2"],
                    detection["y2"]
                ),
                (0, 255, 0),
                2
            )


            label = (
                f"Cow #{cow_id} | "
                f"{detection['behavior']} | "
                f"{detection['confidence']:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    detection["x1"],
                    max(25, detection["y1"] - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


        writer.write(frame)

        frame_number += 1

        if frame_number % 25 == 0:

            print(
                f"Processed {frame_number}/250 frames..."
            )


cap.release()
writer.release()


print()
print("========================================")
print("STABLE TEMPORAL TRACKING COMPLETED")
print("========================================")
print(f"Frames processed: {frame_number}")
print(f"Output video: {OUTPUT_VIDEO}")
print(f"Output log: {OUTPUT_LOG}")
print(f"Stable IDs created: {next_cow_id - 1}")