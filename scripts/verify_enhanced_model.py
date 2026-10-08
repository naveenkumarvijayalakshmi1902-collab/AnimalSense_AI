from __future__ import annotations

from ultralytics import YOLO

from common import ROOT


def main() -> None:
    path = ROOT / "configs" / "yolov8_cbam.yaml"
    model = YOLO(str(path))
    model.info(verbose=True)
    print("Enhanced YOLOv8 + CBAM YAML parsed successfully.")


if __name__ == "__main__":
    main()
