from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

from common import ROOT, load_config, select_device


def metric_row(name: str, weights: Path, cfg: dict) -> dict:
    model = YOLO(str(weights))
    m = model.val(data=str(ROOT / "configs/cattle.yaml"), imgsz=int(cfg["model"]["imgsz"]), device=select_device(cfg["model"].get("device", "auto")), plots=False, verbose=False)
    p, r = float(m.box.mp), float(m.box.mr)
    return {"Model": name, "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "Precision": p, "Recall": r, "F1": 2*p*r/(p+r) if p+r else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="runs/yolov8_baseline/weights/best.pt")
    ap.add_argument("--enhanced", default="runs/yolov8_cbam/weights/best.pt")
    args = ap.parse_args()
    cfg = load_config()
    rows = []
    for name, raw in (("YOLOv8 baseline", args.baseline), ("YOLOv8 + CBAM", args.enhanced)):
        p = Path(raw)
        if not p.is_absolute(): p = ROOT / p
        if p.exists(): rows.append(metric_row(name, p, cfg))
    if not rows: raise FileNotFoundError("No checkpoints found.")
    df = pd.DataFrame(rows)
    out = ROOT / "outputs/logs/model_comparison.csv"
    df.to_csv(out, index=False)
    for metric in ["mAP50", "mAP50-95", "Precision", "Recall", "F1"]:
        plt.figure(figsize=(8, 5))
        plt.bar(df["Model"], df[metric])
        plt.ylabel(metric); plt.title(f"AnimalSense AI - {metric}")
        plt.xticks(rotation=15, ha="right"); plt.tight_layout()
        plt.savefig(ROOT / "outputs/graphs" / f"{metric.lower().replace('-', '_')}.png", dpi=180)
        plt.close()
    print(df.to_string(index=False)); print(f"Saved: {out}")


if __name__ == "__main__":
    main()
