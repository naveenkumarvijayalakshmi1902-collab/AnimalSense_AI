from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import multiprocessing
multiprocessing.freeze_support()


# ============================================================
# ANIMALSENSE AI - RESULTS & CONFUSION MATRIX GENERATOR
# ============================================================

# ---------- PATHS ----------
PROJECT_ROOT = Path(r"D:\AnimalSense_AI")

# Your trained YOLOv8 baseline
MODEL_PATH = PROJECT_ROOT / r"runs\detect\runs\comparison\baseline_10ep\weights\best.pt"

# Training results.csv
RESULTS_CSV = PROJECT_ROOT / r"runs\detect\runs\comparison\baseline_10ep\results.csv"

# Dataset configuration
DATA_YAML = PROJECT_ROOT / r"configs\cattle.yaml"

# Output folder
OUTPUT_DIR = PROJECT_ROOT / r"outputs\results_graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. READ TRAINING RESULTS
# ============================================================

print("\n========================================")
print("AnimalSense AI Results Generator")
print("========================================")

if not RESULTS_CSV.exists():
    raise FileNotFoundError(
        f"results.csv not found:\n{RESULTS_CSV}"
    )

print(f"\nReading:\n{RESULTS_CSV}")

df = pd.read_csv(RESULTS_CSV)

# Remove spaces from column names
df.columns = df.columns.str.strip()

print("\nAvailable columns:")
print(df.columns.tolist())


# ============================================================
# 2. GET FINAL METRICS
# ============================================================

def get_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


precision_col = get_column(
    df,
    ["metrics/precision(B)", "metrics/precision"]
)

recall_col = get_column(
    df,
    ["metrics/recall(B)", "metrics/recall"]
)

map50_col = get_column(
    df,
    ["metrics/mAP50(B)", "metrics/mAP50"]
)

map5095_col = get_column(
    df,
    ["metrics/mAP50-95(B)", "metrics/mAP50-95"]
)

# Final epoch
last = df.iloc[-1]

precision = float(last[precision_col]) if precision_col else 0
recall = float(last[recall_col]) if recall_col else 0
map50 = float(last[map50_col]) if map50_col else 0
map5095 = float(last[map5095_col]) if map5095_col else 0

# F1 score
if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0

print("\n========================================")
print("FINAL PERFORMANCE")
print("========================================")

print(f"Precision     : {precision * 100:.2f}%")
print(f"Recall        : {recall * 100:.2f}%")
print(f"F1 Score      : {f1 * 100:.2f}%")
print(f"mAP@50        : {map50 * 100:.2f}%")
print(f"mAP@50-95     : {map5095 * 100:.2f}%")


# ============================================================
# 3. CREATE PERFORMANCE METRICS TABLE
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

ax.axis("off")

table_data = [
    ["Precision (%)", f"{precision * 100:.2f}"],
    ["Recall (%)", f"{recall * 100:.2f}"],
    ["F1-Score (%)", f"{f1 * 100:.2f}"],
    ["mAP@50 (%)", f"{map50 * 100:.2f}"],
    ["mAP@50:95 (%)", f"{map5095 * 100:.2f}"],
]

table = ax.table(
    cellText=table_data,
    colLabels=["Performance Metrics", "Value"],
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 2.0)

ax.set_title(
    "Quantitative Performance of the AnimalSense AI System",
    fontsize=17,
    fontweight="bold",
    pad=20
)

plt.tight_layout()

metrics_table_path = OUTPUT_DIR / "performance_metrics_table.png"

plt.savefig(
    metrics_table_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nSaved: {metrics_table_path}")


# ============================================================
# 4. TRAINING & VALIDATION LOSS GRAPHS
# ============================================================

loss_columns = [
    ("train/box_loss", "Training Box Loss"),
    ("train/cls_loss", "Training Classification Loss"),
    ("train/dfl_loss", "Training DFL Loss"),
    ("val/box_loss", "Validation Box Loss"),
    ("val/cls_loss", "Validation Classification Loss"),
    ("val/dfl_loss", "Validation DFL Loss"),
]

available_loss_columns = [
    item for item in loss_columns
    if item[0] in df.columns
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9)
)

axes = axes.flatten()

for i, (column, title) in enumerate(available_loss_columns):

    axes[i].plot(
        df["epoch"],
        df[column],
        linewidth=2
    )

    axes[i].set_title(
        title,
        fontsize=12,
        fontweight="bold"
    )

    axes[i].set_xlabel("Epoch")
    axes[i].set_ylabel("Loss")
    axes[i].grid(True, alpha=0.3)

# Hide unused axes
for j in range(len(available_loss_columns), len(axes)):
    axes[j].axis("off")

fig.suptitle(
    "Training and Validation Performance of AnimalSense AI",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

loss_path = OUTPUT_DIR / "training_validation_loss.png"

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {loss_path}")


# ============================================================
# 5. PRECISION / RECALL / mAP GRAPHS
# ============================================================

metric_columns = [
    ("metrics/precision(B)", "Precision"),
    ("metrics/recall(B)", "Recall"),
    ("metrics/mAP50(B)", "mAP@50"),
    ("metrics/mAP50-95(B)", "mAP@50:95"),
]

available_metric_columns = [
    item for item in metric_columns
    if item[0] in df.columns
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9)
)

axes = axes.flatten()

for i, (column, title) in enumerate(available_metric_columns):

    axes[i].plot(
        df["epoch"],
        df[column] * 100,
        linewidth=2
    )

    axes[i].set_title(
        title,
        fontsize=13,
        fontweight="bold"
    )

    axes[i].set_xlabel("Epoch")
    axes[i].set_ylabel("Percentage (%)")

    axes[i].set_ylim(0, 100)

    axes[i].grid(True, alpha=0.3)

# Hide unused axes
for j in range(len(available_metric_columns), len(axes)):
    axes[j].axis("off")

fig.suptitle(
    "Training and Validation Metrics of AnimalSense AI",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

metric_graph_path = OUTPUT_DIR / "precision_recall_map_graphs.png"

plt.savefig(
    metric_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {metric_graph_path}")


# ============================================================
# 6. GENERATE CONFUSION MATRIX USING YOLO
# ============================================================

print("\n========================================")
print("Generating Confusion Matrix")
print("========================================")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print("\nRunning validation...")

validation_results = model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=320,
    batch=4,
    device=0,
    workers=0,
    plots=True,
    save_json=False
)

print("\nValidation complete.")


# ============================================================
# 7. FIND YOLO CONFUSION MATRIX
# ============================================================

possible_cm_files = [
    Path(validation_results.save_dir) / "confusion_matrix.png",
    Path(validation_results.save_dir) / "confusion_matrix_normalized.png",
]

for cm_file in possible_cm_files:

    if cm_file.exists():

        destination = OUTPUT_DIR / cm_file.name

        shutil.copy2(
            cm_file,
            destination
        )

        print(f"Saved: {destination}")


# ============================================================
# 8. CREATE A SIMPLE METRICS BAR GRAPH
# ============================================================

metric_names = [
    "Precision",
    "Recall",
    "F1-Score",
    "mAP@50",
    "mAP@50:95"
]

metric_values = [
    precision * 100,
    recall * 100,
    f1 * 100,
    map50 * 100,
    map5095 * 100
]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    metric_names,
    metric_values
)

ax.set_ylim(0, 100)

ax.set_ylabel("Performance (%)")

ax.set_title(
    "AnimalSense AI Performance Metrics",
    fontsize=17,
    fontweight="bold"
)

ax.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(bars, metric_values):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.1f}%",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.tight_layout()

bar_path = OUTPUT_DIR / "performance_comparison.png"

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {bar_path}")


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("ALL RESULTS GENERATED")
print("========================================")

print(f"\nOutput folder:")
print(OUTPUT_DIR)

print("\nFiles generated:")

for file in OUTPUT_DIR.iterdir():
    print(" -", file.name)

print("\nDone.")
