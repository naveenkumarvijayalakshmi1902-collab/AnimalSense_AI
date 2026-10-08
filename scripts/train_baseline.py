from __future__ import annotations

from ultralytics import YOLO

from common import ROOT, ensure_dirs, load_config, select_device


def main() -> None:
    cfg = load_config()
    ensure_dirs()
    t = cfg["training"]
    device = select_device(cfg["model"].get("device", "auto"))
    data = ROOT / "configs" / "cattle.yaml"
    print("=== AnimalSense AI | YOLOv8 baseline ===")
    print(f"Device: {device}")
    model = YOLO(cfg["model"]["baseline_weights"])
    model.train(
        data=str(data), epochs=int(t["epochs"]), imgsz=int(t["imgsz"]), batch=int(t["batch"]),
        workers=int(t["workers"]), patience=int(t["patience"]), project=str(ROOT / t["project"]),
        name=t["baseline_name"], pretrained=True, device=device, exist_ok=True,
    )
    print("Baseline training finished.")


if __name__ == "__main__":
    main()
