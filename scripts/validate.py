from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from common import ROOT, load_config, select_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/yolov8_baseline/weights/best.pt")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = load_config()
    p = Path(args.model)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    m = YOLO(str(p))
    r = m.val(data=str(ROOT / "configs/cattle.yaml"), imgsz=int(cfg["model"]["imgsz"]), device=select_device(cfg["model"].get("device", "auto")), plots=True)
    row = {
        "model": str(p), "mAP50": float(r.box.map50), "mAP50-95": float(r.box.map),
        "precision": float(r.box.mp), "recall": float(r.box.mr),
    }
    print(row)
    out = ROOT / "outputs/logs" / f"{args.name or p.stem}_metrics.csv"
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
