from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

try:
    from .alerts import alert_record, explain_reasons
    from .baseline import IndividualBaseline
    from .behaviour import BehaviourEngine
    from .common import ROOT, ensure_dirs, load_config, resolve_path, select_device
    from .deviation import DeviationEngine
except ImportError:
    from alerts import alert_record, explain_reasons
    from baseline import IndividualBaseline
    from behaviour import BehaviourEngine
    from common import ROOT, ensure_dirs, load_config, resolve_path, select_device
    from deviation import DeviationEngine


def _animal_id(track_id: int | None) -> str:
    return f"Cow-A{int(track_id or 0):03d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="AnimalSense AI continuous video pipeline")
    ap.add_argument("--source", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs()
    source = resolve_path(args.source or cfg["video"]["source"])
    output = resolve_path(args.output or cfg["video"]["output"])
    model_arg = args.model or cfg["model"]["enhanced_weights"]
    model_path = resolve_path(model_arg)
    if not model_path.exists():
        baseline = ROOT / "runs/yolov8_baseline/weights/best.pt"
        model_path = baseline if baseline.exists() else Path(cfg["model"]["baseline_weights"])
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    device = select_device(cfg["model"].get("device", "auto"))
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    skip = max(1, int(cfg["video"].get("frame_skip", 1)))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps / skip, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output}")

    behaviours = list(cfg["behaviour"]["allowed"])
    bcfg = cfg["baseline"]
    acfg = cfg["anomaly"]
    engines: dict[str, BehaviourEngine] = {}
    baselines: dict[str, IndividualBaseline] = {}
    first_seen: dict[str, float] = {}
    last_seen: dict[str, float] = {}
    dev = DeviationEngine(float(acfg["deviation_threshold"]), float(acfg["warning_threshold"]), float(acfg["critical_threshold"]), int(acfg["persistence_windows"]))
    logs: list[dict] = []
    alerts: list[dict] = []
    frame_no = processed = 0
    t0 = time.perf_counter()

    print("AnimalSense AI")
    print("-------------------------------")
    print(f"Video: {source}")
    print(f"Model: {model_path}")
    print(f"FPS: {fps:.2f} | Resolution: {width}x{height} | Frames: {frame_count}")
    print(f"Device: {device}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % skip != 0:
            continue
        timestamp = frame_no / fps
        runtime_bcfg = dict(cfg["behaviour"], frame_width=width, frame_height=height)
        tracked = model.track(frame, persist=True, tracker=cfg["model"].get("tracker", "bytetrack.yaml"), conf=float(cfg["model"]["confidence"]), iou=float(cfg["model"]["iou"]), imgsz=int(cfg["model"]["imgsz"]), device=device, verbose=False, classes=[0])
        result = tracked[0]
        canvas = frame.copy()
        active = set()

        if result.boxes is not None and len(result.boxes):
            ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else list(range(len(result.boxes)))
            boxes = result.boxes.xyxy.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            for tid, bbox, conf in zip(ids, boxes, confs):
                aid = _animal_id(tid)
                active.add(aid)
                first_seen.setdefault(aid, timestamp)
                last_seen[aid] = timestamp
                if aid not in engines:
                    engines[aid] = BehaviourEngine(runtime_bcfg)
                    baselines[aid] = IndividualBaseline(behaviours, float(bcfg["ema_alpha"]))
                else:
                    engines[aid].config.update(runtime_bcfg)
                behaviour, bconf, feat = engines[aid].infer(frame_no, timestamp, tuple(map(float, bbox)), float(conf))
                current = engines[aid].distribution(timestamp, float(acfg["monitor_window_seconds"]), behaviours)
                baselines[aid].update(current, timestamp, float(bcfg["warmup_seconds"]), int(bcfg["min_observations"]))
                ready = baselines[aid].ready()
                score, per = dev.score(current, baselines[aid].distribution)
                status, fire = dev.status(aid, score, ready)
                reasons = explain_reasons(per, current, baselines[aid].distribution)

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(canvas, f"{aid} | {behaviour} | {bconf:.2f} | D:{score:.1f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(canvas, status, (x1, min(height - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                logs.append({
                    "timestamp": round(timestamp, 3), "frame_number": frame_no, "animal_id": aid,
                    "behaviour": behaviour, "confidence": round(bconf, 4),
                    "movement_score": round(float(feat["movement"]), 6), "aspect_ratio": round(float(feat["aspect_ratio"]), 4),
                    "baseline_ready": ready, "deviation_score": round(score, 4), "risk_level": status,
                })
                if fire:
                    prev = next((a for a in reversed(alerts) if a["animal_id"] == aid), None)
                    if prev is None or timestamp - prev["timestamp_seconds"] >= 30 or prev["status"] != status:
                        alerts.append(alert_record(timestamp, aid, status, score, reasons))

        cv2.putText(canvas, "AnimalSense AI | Continuous Video", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Frame: {frame_no} | Active animals: {len(active)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(canvas)
        processed += 1

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t0
    out_dir = ROOT / "outputs/logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_csv(out_dir / "behaviour_log.csv", index=False)

    summary = []
    for aid, engine in engines.items():
        now = last_seen[aid]
        current = engine.distribution(now, float(acfg["monitor_window_seconds"]), behaviours)
        score, _ = dev.score(current, baselines[aid].distribution)
        status, _ = dev.status(aid, score, baselines[aid].ready())
        summary.append({
            "animal_id": aid, "total_observations": len(engine.behaviour_history),
            "baseline_ready": baselines[aid].ready(), "baseline": json.dumps(baselines[aid].distribution),
            "current_distribution": json.dumps(current), "deviation_score": round(score, 3), "risk_level": status,
            "first_seen_seconds": round(first_seen[aid], 3), "last_seen_seconds": round(now, 3),
        })
    pd.DataFrame(summary).to_csv(out_dir / "animal_summary.csv", index=False)
    (out_dir / "alerts.json").write_text(json.dumps(alerts, indent=2), encoding="utf-8")

    print("\nProcessing completed.")
    print(f"Animals observed: {len(engines)}")
    print(f"Processed frames: {processed}")
    print(f"Processing FPS: {processed / max(elapsed, 1e-9):.2f}")
    print(f"Output video: {output}")
    print(f"Logs: {out_dir}")


if __name__ == "__main__":
    main()
