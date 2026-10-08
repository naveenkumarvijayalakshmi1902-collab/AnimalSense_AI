import csv
import os
from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\AnimalSense_AI")

VIDEO_DIR = PROJECT_ROOT / "dataset" / "videos"

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "comparison"
    / "baseline_10ep"
    / "weights"
    / "best.pt"
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "batch_analysis"

# Processing settings
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
# UTILITY
# ============================================================

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


# ============================================================
# STEP 1 — TRACKING
# ============================================================

def process_video(video_path, output_dir, model):

    print()
    print("=" * 70)
    print(f"PROCESSING: {video_path.name}")
    print("=" * 70)

    ensure_dir(output_dir)

    history_csv = output_dir / "behavior_history.csv"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")

    history_file = open(
        history_csv,
        "w",
        newline="",
        encoding="utf-8"
    )

    writer = csv.writer(history_file)

    writer.writerow([
        "frame",
        "timestamp_sec",
        "track_id",
        "behavior",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ])

    frame_number = 0

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
            verbose=False,
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for box, track_id, cls, conf in zip(
                boxes,
                ids,
                classes,
                confs
            ):

                x1, y1, x2, y2 = box

                if cls < len(BEHAVIORS):
                    behavior = BEHAVIORS[cls]
                else:
                    behavior = "unknown"

                timestamp = (
                    frame_number / fps
                    if fps > 0
                    else 0
                )

                writer.writerow([
                    frame_number,
                    f"{timestamp:.3f}",
                    int(track_id),
                    behavior,
                    f"{float(conf):.4f}",
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ])

        if frame_number % 100 == 0:

            percent = (
                frame_number / total_frames * 100
                if total_frames > 0
                else 0
            )

            print(
                f"  {frame_number}/{total_frames}"
                f" ({percent:.1f}%)"
            )

    cap.release()
    history_file.close()

    print(f"History saved: {history_csv}")

    return True


# ============================================================
# STEP 2 — LOAD HISTORY
# ============================================================

def load_history(history_csv):

    track_history = defaultdict(list)

    with open(
        history_csv,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            track_id = int(row["track_id"])

            track_history[track_id].append({
                "frame": int(row["frame"]),
                "behavior": row["behavior"],
                "confidence": safe_float(row["confidence"]),
            })

    return track_history


# ============================================================
# STEP 3 — ADAPTIVE BASELINE
# ============================================================

def calculate_baselines(track_history):

    baselines = {}

    for track_id, observations in track_history.items():

        counts = defaultdict(float)

        for behavior in BEHAVIORS:
            counts[behavior] = 0.0

        ema = {
            behavior: 0.0
            for behavior in BEHAVIORS
        }

        for obs in observations:

            behavior = obs["behavior"]

            if behavior in counts:
                counts[behavior] += 1

            for b in BEHAVIORS:

                current = (
                    1.0
                    if behavior == b
                    else 0.0
                )

                ema[b] = (
                    EMA_ALPHA * current
                    + (1 - EMA_ALPHA) * ema[b]
                )

        total = len(observations)

        ready = total >= MIN_HISTORY

        baselines[track_id] = {
            "observations": total,
            "ready": ready,
            "ema": ema,
            "counts": counts,
        }

    return baselines


# ============================================================
# STEP 4 — ANOMALY / RISK
# ============================================================

def calculate_risk(track_history, baselines):

    risk_results = []

    for track_id, observations in track_history.items():

        baseline = baselines[track_id]

        total = len(observations)

        if not baseline["ready"]:

            risk_results.append({
                "track_id": track_id,
                "observations": total,
                "risk_score": "",
                "risk_level": "INSUFFICIENT_DATA",
                "reason": "Insufficient tracking history",
            })

            continue

        # Last temporal window
        window = observations[-WINDOW_SIZE:]

        window_counts = {
            behavior: 0
            for behavior in BEHAVIORS
        }

        for obs in window:

            behavior = obs["behavior"]

            if behavior in window_counts:
                window_counts[behavior] += 1

        window_total = len(window)

        current_distribution = {}

        for behavior in BEHAVIORS:

            if window_total > 0:
                current_distribution[behavior] = (
                    window_counts[behavior]
                    / window_total
                    * 100
                )
            else:
                current_distribution[behavior] = 0

        deviations = {}

        for behavior in BEHAVIORS:

            baseline_value = (
                baseline["ema"][behavior] * 100
            )

            current_value = (
                current_distribution[behavior]
            )

            deviations[behavior] = (
                abs(current_value - baseline_value)
            )

        average_deviation = (
            sum(deviations.values())
            / len(BEHAVIORS)
        )

        risk_score = min(
            100,
            average_deviation * 2
        )

        if risk_score >= HIGH_THRESHOLD:
            risk_level = "HIGH"

        elif risk_score >= MEDIUM_THRESHOLD:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

        # Explainable reason
        sorted_changes = sorted(
            deviations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        reasons = []

        for behavior, deviation in sorted_changes[:3]:

            if deviation >= 5:

                baseline_value = (
                    baseline["ema"][behavior] * 100
                )

                current_value = (
                    current_distribution[behavior]
                )

                direction = (
                    "increased"
                    if current_value > baseline_value
                    else "decreased"
                )

                reasons.append(
                    f"{behavior} {direction} "
                    f"by {deviation:.1f}%"
                )

        reason = "; ".join(reasons)

        if not reason:
            reason = "No major behavioral deviation"

        risk_results.append({
            "track_id": track_id,
            "observations": total,
            "risk_score": f"{risk_score:.2f}",
            "risk_level": risk_level,
            "reason": reason,
        })

    return risk_results


# ============================================================
# STEP 5 — SAVE RESULTS
# ============================================================

def save_results(output_dir, baselines, risk_results):

    baseline_csv = (
        output_dir / "adaptive_baseline.csv"
    )

    risk_csv = (
        output_dir / "anomaly_risk.csv"
    )

    # Baseline
    with open(
        baseline_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "track_id",
            "observations",
            "baseline_ready",
            "stand",
            "lying_down",
            "foraging",
            "drinking_water",
            "rumination",
        ])

        for track_id, data in baselines.items():

            writer.writerow([
                track_id,
                data["observations"],
                data["ready"],
                *[
                    f"{data['ema'][b] * 100:.2f}"
                    for b in BEHAVIORS
                ],
            ])

    # Risk
    with open(
        risk_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "track_id",
            "observations",
            "risk_score",
            "risk_level",
            "reason",
        ])

        for result in risk_results:

            writer.writerow([
                result["track_id"],
                result["observations"],
                result["risk_score"],
                result["risk_level"],
                result["reason"],
            ])

    return baseline_csv, risk_csv


# ============================================================
# STEP 6 — MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ANIMALSENSE AI — BATCH VIDEO ANALYSIS")
    print("=" * 70)

    if not VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"Video directory not found: {VIDEO_DIR}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    ensure_dir(OUTPUT_ROOT)

    videos = sorted(
        VIDEO_DIR.glob("*.mp4")
    )[:3]

    print(f"Videos found: {len(videos)}")

    if not videos:
        print("No MP4 videos found.")
        return

    print()
    print("Loading YOLOv8 model...")
    model = YOLO(str(MODEL_PATH))

    overall_summary = []

    for index, video_path in enumerate(
        videos,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(videos)}] "
            f"{video_path.name}"
        )

        video_name = video_path.stem

        output_dir = (
            OUTPUT_ROOT / video_name
        )

        ensure_dir(output_dir)

        try:

            success = process_video(
                video_path,
                output_dir,
                model
            )

            if not success:
                continue

            history_csv = (
                output_dir
                / "behavior_history.csv"
            )

            track_history = load_history(
                history_csv
            )

            baselines = calculate_baselines(
                track_history
            )

            risk_results = calculate_risk(
                track_history,
                baselines
            )

            baseline_csv, risk_csv = save_results(
                output_dir,
                baselines,
                risk_results
            )

            high = sum(
                1
                for r in risk_results
                if r["risk_level"] == "HIGH"
            )

            medium = sum(
                1
                for r in risk_results
                if r["risk_level"] == "MEDIUM"
            )

            low = sum(
                1
                for r in risk_results
                if r["risk_level"] == "LOW"
            )

            insufficient = sum(
                1
                for r in risk_results
                if r["risk_level"]
                == "INSUFFICIENT_DATA"
            )

            overall_summary.append([
                video_path.name,
                len(track_history),
                low,
                medium,
                high,
                insufficient,
            ])

            print()
            print(
                f"Completed {video_path.name}"
            )
            print(
                f"Tracks: {len(track_history)}"
            )
            print(f"LOW: {low}")
            print(f"MEDIUM: {medium}")
            print(f"HIGH: {high}")
            print(
                f"INSUFFICIENT: {insufficient}"
            )

        except Exception as e:

            print()
            print(
                f"ERROR processing "
                f"{video_path.name}:"
            )
            print(e)

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    summary_csv = (
        OUTPUT_ROOT
        / "overall_anomaly_summary.csv"
    )

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "video",
            "tracks",
            "low",
            "medium",
            "high",
            "insufficient_data",
        ])

        writer.writerows(
            overall_summary
        )

    print()
    print("=" * 70)
    print("BATCH PROCESSING COMPLETED")
    print("=" * 70)

    print(
        f"Processed videos: "
        f"{len(overall_summary)}"
    )

    print(
        f"Summary:"
        f"\n{summary_csv}"
    )

    print()
    print(
        f"All results:"
        f"\n{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()