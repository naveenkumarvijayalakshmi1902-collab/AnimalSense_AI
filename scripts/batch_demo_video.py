from pathlib import Path
import csv
import cv2
import numpy as np
from collections import defaultdict, Counter
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = r"D:\AnimalSense_AI\runs\detect\runs\comparison\baseline_10ep\weights\best.pt"
VIDEO_DIR = Path(r"D:\AnimalSense_AI\dataset\videos")
OUTPUT_ROOT = Path(r"D:\AnimalSense_AI\outputs\batch_demo")

TEST_VIDEOS = [
    "619.mp4",
    "620.mp4",
]

IMG_SIZE = 320
CONFIDENCE = 0.25
DEVICE = 0

WINDOW_SIZE = 25
MIN_HISTORY = 25
EMA_ALPHA = 0.10

MEDIUM_THRESHOLD = 20
HIGH_THRESHOLD = 40

BEHAVIORS = [
    "stand",
    "lying_down",
    "foraging",
    "drinking_water",
    "rumination",
]


# ============================================================
# HELPERS
# ============================================================

def behavior_name(class_id):
    if class_id is None:
        return "unknown"

    class_id = int(class_id)

    if 0 <= class_id < len(BEHAVIORS):
        return BEHAVIORS[class_id]

    return "unknown"


def risk_level(score):
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    elif score >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


def calculate_percentages(observations):
    total = len(observations)

    counts = Counter(observations)

    result = {}

    for behavior in BEHAVIORS:
        result[behavior] = (
            counts.get(behavior, 0) / total * 100
            if total > 0
            else 0.0
        )

    return result


def draw_text(frame, text, position, scale=0.55, thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):

    video_name = video_path.stem
    output_dir = OUTPUT_ROOT / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print(f"PROCESSING: {video_name}")
    print("=" * 70)

    history_csv = output_dir / "behavior_history.csv"
    baseline_csv = output_dir / "adaptive_baseline.csv"
    risk_csv = output_dir / "anomaly_risk.csv"
    output_video = output_dir / "analyzed_video.mp4"

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps:.2f}")
    print(f"Frames     : {total_frames}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    track_history = defaultdict(list)

    frame_detections = defaultdict(list)

    frame_number = 0

    # --------------------------------------------------------
    # Track video
    # --------------------------------------------------------

    print("Running YOLO + ByteTrack...")

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            device=DEVICE,
            verbose=False,
        )

        if not results:
            continue

        result = results[0]

        if result.boxes is None:
            continue

        boxes = result.boxes

        if boxes.xyxy is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()

        confs = boxes.conf.cpu().numpy()

        classes = boxes.cls.cpu().numpy()

        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [-1] * len(xyxy)

        for i in range(len(xyxy)):

            box = xyxy[i]

            confidence = float(confs[i])

            class_id = int(classes[i])

            track_id = int(track_ids[i])

            behavior = behavior_name(class_id)

            x1, y1, x2, y2 = [
                int(v) for v in box
            ]

            detection = {
                "track_id": track_id,
                "behavior": behavior,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }

            frame_detections[frame_number].append(detection)

            if track_id >= 0:
                track_history[track_id].append(
                    behavior
                )

        if frame_number % 100 == 0:

            print(
                f"Processed {frame_number}/{total_frames} frames",
                end="\r",
            )

    cap.release()

    print()
    print("Tracking completed.")

    # --------------------------------------------------------
    # Save behavior history
    # --------------------------------------------------------

    print("Saving behavior history...")

    with open(
        history_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "frame",
                "timestamp_sec",
                "track_id",
                "behavior",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )

        for frame_no in sorted(frame_detections):

            timestamp = frame_no / fps

            for detection in frame_detections[frame_no]:

                writer.writerow(
                    [
                        frame_no,
                        round(timestamp, 3),
                        detection["track_id"],
                        detection["behavior"],
                        round(
                            detection["confidence"],
                            4,
                        ),
                        detection["x1"],
                        detection["y1"],
                        detection["x2"],
                        detection["y2"],
                    ]
                )

    # --------------------------------------------------------
    # Adaptive baseline
    # --------------------------------------------------------

    print("Calculating adaptive baselines...")

    baselines = {}

    with open(
        baseline_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "track_id",
                "observations",
                "baseline_stand",
                "baseline_lying_down",
                "baseline_foraging",
                "baseline_drinking_water",
                "baseline_rumination",
                "status",
            ]
        )

        for track_id, observations in track_history.items():

            percentages = calculate_percentages(
                observations
            )

            count = len(observations)

            if count >= MIN_HISTORY:

                status = "READY"

                baselines[track_id] = percentages

            else:

                status = "INSUFFICIENT_DATA"

            writer.writerow(
                [
                    track_id,
                    count,
                    round(
                        percentages["stand"],
                        3,
                    ),
                    round(
                        percentages["lying_down"],
                        3,
                    ),
                    round(
                        percentages["foraging"],
                        3,
                    ),
                    round(
                        percentages["drinking_water"],
                        3,
                    ),
                    round(
                        percentages["rumination"],
                        3,
                    ),
                    status,
                ]
            )

    # --------------------------------------------------------
    # Anomaly / risk analysis
    # --------------------------------------------------------

    print("Calculating anomaly risk...")

    risk_results = {}

    with open(
        risk_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "track_id",
                "observations",
                "risk_score",
                "risk_level",
                "top_reason",
                "status",
            ]
        )

        for track_id, observations in track_history.items():

            count = len(observations)

            if count < MIN_HISTORY:

                risk_results[track_id] = {
                    "score": 0.0,
                    "level": "INSUFFICIENT_DATA",
                    "reason": "Not enough behavior history",
                }

                writer.writerow(
                    [
                        track_id,
                        count,
                        0.0,
                        "INSUFFICIENT_DATA",
                        "Not enough behavior history",
                        "INSUFFICIENT_DATA",
                    ]
                )

                continue

            baseline = baselines[track_id]

            recent = observations[-WINDOW_SIZE:]

            recent_percentages = calculate_percentages(
                recent
            )

            deviations = []

            for behavior in BEHAVIORS:

                difference = (
                    recent_percentages[behavior]
                    - baseline[behavior]
                )

                deviations.append(
                    (
                        behavior,
                        difference,
                        abs(difference),
                    )
                )

            deviations.sort(
                key=lambda x: x[2],
                reverse=True,
            )

            # Average absolute deviation.
            average_deviation = sum(
                item[2] for item in deviations
            ) / len(deviations)

            score = min(
                100.0,
                average_deviation * 2.0,
            )

            level = risk_level(score)

            biggest = deviations[0]

            behavior = biggest[0]

            difference = biggest[1]

            if difference > 0:

                reason = (
                    f"{behavior} increased by "
                    f"{difference:.1f}%"
                )

            else:

                reason = (
                    f"{behavior} decreased by "
                    f"{abs(difference):.1f}%"
                )

            risk_results[track_id] = {
                "score": score,
                "level": level,
                "reason": reason,
            }

            writer.writerow(
                [
                    track_id,
                    count,
                    round(score, 2),
                    level,
                    reason,
                    "READY",
                ]
            )

    # --------------------------------------------------------
    # Render final analyzed video
    # --------------------------------------------------------

    print("Rendering analyzed video...")

    cap = cv2.VideoCapture(str(video_path))

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer_video = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height),
    )

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        detections = frame_detections.get(
            frame_number,
            [],
        )

        high_count = 0
        medium_count = 0
        low_count = 0

        for detection in detections:

            track_id = detection["track_id"]

            behavior = detection["behavior"]

            confidence = detection["confidence"]

            x1 = detection["x1"]
            y1 = detection["y1"]
            x2 = detection["x2"]
            y2 = detection["y2"]

            info = risk_results.get(
                track_id,
                {
                    "score": 0.0,
                    "level": "INSUFFICIENT_DATA",
                    "reason": "",
                },
            )

            level = info["level"]

            score = info["score"]

            if level == "HIGH":
                high_count += 1

            elif level == "MEDIUM":
                medium_count += 1

            elif level == "LOW":
                low_count += 1

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            label = (
                f"ID {track_id} | "
                f"{behavior} | "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                frame,
                (
                    x1,
                    max(0, y1 - 28),
                ),
                (
                    min(
                        width - 1,
                        x1 + 300,
                    ),
                    y1,
                ),
                (0, 0, 0),
                -1,
            )

            draw_text(
                frame,
                label,
                (
                    x1 + 4,
                    max(20, y1 - 8),
                ),
                0.48,
                1,
            )

            # ------------------------------------------------
            # Risk label
            # ------------------------------------------------

            risk_text = (
                f"Risk: {level} "
                f"({score:.1f})"
            )

            draw_text(
                frame,
                risk_text,
                (
                    x1,
                    min(
                        height - 10,
                        y2 + 20,
                    ),
                ),
                0.50,
                2,
            )

        # ----------------------------------------------------
        # Information panel
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (360, 125),
            (0, 0, 0),
            -1,
        )

        draw_text(
            frame,
            "AnimalSense AI",
            (20, 35),
            0.75,
            2,
        )

        draw_text(
            frame,
            f"Video: {video_name}",
            (20, 60),
            0.50,
            1,
        )

        draw_text(
            frame,
            f"Frame: {frame_number}/{total_frames}",
            (20, 82),
            0.50,
            1,
        )

        draw_text(
            frame,
            (
                f"High: {high_count}  "
                f"Medium: {medium_count}  "
                f"Low: {low_count}"
            ),
            (20, 105),
            0.50,
            1,
        )

        writer_video.write(frame)

        if frame_number % 100 == 0:

            print(
                f"Rendering {frame_number}/{total_frames}",
                end="\r",
            )

    cap.release()
    writer_video.release()

    print()
    print()
    print(f"Completed: {video_name}")
    print(f"Video    : {output_video}")
    print(f"History  : {history_csv}")
    print(f"Baseline : {baseline_csv}")
    print(f"Risk     : {risk_csv}")


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for video_name in TEST_VIDEOS:

        video_path = VIDEO_DIR / video_name

        if not video_path.exists():

            print(
                f"WARNING: Missing video: "
                f"{video_path}"
            )

            continue

        process_video(video_path)

    print()
    print("=" * 70)
    print("ALL REQUESTED VIDEOS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()