import csv
import cv2
from collections import defaultdict

INPUT_VIDEO = r"D:\AnimalSense_AI\dataset\videos\618.mp4"
HISTORY_CSV = r"D:\AnimalSense_AI\outputs\final_behavior_history.csv"
RISK_CSV = r"D:\AnimalSense_AI\outputs\final_anomaly_risk.csv"
OUTPUT_VIDEO = r"D:\AnimalSense_AI\outputs\animalsense_final_demo.mp4"


# ---------------------------------------------------------
# Load behavioral risk information
# ---------------------------------------------------------
risk_map = {}

with open(RISK_CSV, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        track_id = int(row["track_id"])

        risk_map[track_id] = {
            "score": float(row["risk_score"]),
            "level": row["risk_level"],
            "reason": row.get("reason", ""),
        }


# ---------------------------------------------------------
# Load tracking history
# ---------------------------------------------------------
frame_data = defaultdict(list)

with open(HISTORY_CSV, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        frame = int(row["frame"])
        track_id = int(row["track_id"])

        frame_data[frame].append({
            "track_id": track_id,
            "behavior": row["behavior"],
            "confidence": float(row["confidence"]),
            "x1": int(float(row["x1"])),
            "y1": int(float(row["y1"])),
            "x2": int(float(row["x2"])),
            "y2": int(float(row["y2"])),
        })


# ---------------------------------------------------------
# Open source video
# ---------------------------------------------------------
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height}")
print(f"FPS: {fps}")
print(f"Frames: {total_frames}")


# ---------------------------------------------------------
# Output writer
# ---------------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video.")


# ---------------------------------------------------------
# Colors
# ---------------------------------------------------------
def risk_color(level):
    if level == "HIGH":
        return (0, 0, 255)

    if level == "MEDIUM":
        return (0, 165, 255)

    if level == "LOW":
        return (0, 200, 0)

    return (180, 180, 180)


# ---------------------------------------------------------
# Process video
# ---------------------------------------------------------
frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    detections = frame_data.get(frame_number, [])

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------
    cv2.rectangle(
        frame,
        (0, 0),
        (width, 85),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "AnimalSense AI",
        (25, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Continuous Cattle Behavior Monitoring",
        (25, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1
    )

    timestamp = frame_number / fps if fps > 0 else 0

    cv2.putText(
        frame,
        f"Time: {timestamp:.1f}s",
        (width - 180, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (width - 180, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # -----------------------------------------------------
    # Draw tracked animals
    # -----------------------------------------------------
    medium_or_high = []

    for det in detections:

        track_id = det["track_id"]
        behavior = det["behavior"]
        confidence = det["confidence"]

        x1 = det["x1"]
        y1 = det["y1"]
        x2 = det["x2"]
        y2 = det["y2"]

        risk = risk_map.get(track_id)

        if risk is None:
            level = "INSUFFICIENT"
            score = 0
        else:
            level = risk["level"]
            score = risk["score"]

            if level in ("MEDIUM", "HIGH"):
                medium_or_high.append((track_id, risk))

        color = risk_color(level)

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # Label
        label = (
            f"Track {track_id} | "
            f"{behavior} | "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, max(85, y1 - 30)),
            (min(width - 1, x1 + 330), y1),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 5, max(105, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        # Risk label below bounding box
        risk_text = f"Risk: {level}"

        if risk is not None:
            risk_text += f" ({score:.1f})"

        cv2.putText(
            frame,
            risk_text,
            (x1, min(height - 10, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )


    # -----------------------------------------------------
    # Alert panel
    # -----------------------------------------------------
    if medium_or_high:

        panel_height = min(
            150,
            40 + len(medium_or_high) * 32
        )

        cv2.rectangle(
            frame,
            (20, height - panel_height - 20),
            (min(width - 20, 850), height - 20),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            "BEHAVIORAL ANOMALY ALERT",
            (35, height - panel_height + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2
        )

        y = height - panel_height + 42

        for track_id, risk in medium_or_high:

            reason = risk["reason"]

            if not reason:
                reason = "Behavior deviation detected"

            text = (
                f"Track {track_id}: "
                f"{risk['level']} | "
                f"{risk['score']:.1f} | "
                f"{reason}"
            )

            # Prevent very long text
            text = text[:105]

            cv2.putText(
                frame,
                text,
                (35, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

            y += 32


    # -----------------------------------------------------
    # Footer
    # -----------------------------------------------------
    cv2.putText(
        frame,
        "YOLOv8 + ByteTrack + Temporal History + Adaptive Baseline",
        (25, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1
    )

    writer.write(frame)

    if frame_number % 25 == 0:
        print(
            f"Processed {frame_number}/{total_frames} frames"
        )


# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------
cap.release()
writer.release()

print()
print("==========================================")
print("FINAL DEMO VIDEO CREATED")
print("==========================================")
print(OUTPUT_VIDEO)
print()