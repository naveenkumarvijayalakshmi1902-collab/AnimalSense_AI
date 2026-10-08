from ultralytics import YOLO

print("Loading YOLOv8n...")

model = YOLO("yolov8n.pt")

print("Starting AnimalSense AI training...")

results = model.train(
    data="configs/cattle.yaml",

    # Fast CPU training
    epochs=5,
    imgsz=320,
    batch=2,
    device="cpu",
    workers=0,

    # Project output
    project="runs",
    name="animalsense_fast",

    # Use pretrained YOLO weights
    pretrained=True,

    # Save best model
    save=True,

    # Validation
    val=True
)

print()
print("Training completed.")
print("Best model:")
print("runs/animalsense_fast/weights/best.pt")