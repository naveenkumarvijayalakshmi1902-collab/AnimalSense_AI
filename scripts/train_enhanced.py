from ultralytics import YOLO

print("Loading enhanced YOLOv8 + CBAM model...")

model = YOLO(r"models\enhanced\yolov8n_cbam.yaml")

print("Starting GPU training...")

results = model.train(
    data=r"configs\cattle.yaml",
    epochs=10,
    imgsz=320,
    batch=4,
    device=0,
    workers=0,
    project=r"runs\enhanced",
    name="cbam_full",
    pretrained=False,
    save=True,
    val=True,
    patience=5
)

print()
print("Full CBAM training completed.")
print(r"Best model: runs\detect\runs\enhanced\cbam_full\weights\best.pt")