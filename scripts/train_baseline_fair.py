from ultralytics import YOLO

print("Loading YOLOv8n baseline...")

model = YOLO("yolov8n.pt")

print("Starting fair baseline training...")

model.train(
    data=r"configs\cattle.yaml",
    epochs=10,
    imgsz=320,
    batch=4,
    device=0,
    workers=0,
    project=r"runs\comparison",
    name="baseline_10ep",
    pretrained=True,
    save=True,
    val=True,
    patience=10,
)

print()
print("Fair baseline training completed.")
print(r"Check: runs\comparison\baseline_10ep\weights\best.pt")