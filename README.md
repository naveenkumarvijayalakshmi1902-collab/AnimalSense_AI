# AnimalSense AI

AnimalSense AI is a computer-vision-based cattle behavior monitoring system.

## Behaviors

The system recognizes:

- Standing
- Lying Down
- Foraging
- Drinking Water
- Rumination

## Pipeline

Video
↓
YOLOv8 Behavior Detection
↓
ByteTrack Tracking
↓
Temporal Behavior History
↓
Adaptive Behavioral Baseline
↓
Behavioral Anomaly Detection
↓
Risk Score
↓
Explainable Alert

## Technologies

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- ByteTrack
- NumPy
- Pandas

## Dataset

The project uses cattle behavior videos with AVA-style annotations converted to YOLO format.

## Model Evaluation

The baseline YOLOv8n model achieved:

- Precision: 70.0%
- Recall: 60.0%
- F1-score: approximately 64.6%
- mAP@50: 64.6%
- mAP@50–95: 34.7%

Further experiments are being conducted to improve these metrics.