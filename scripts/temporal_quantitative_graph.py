from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# AnimalSense AI
# Temporal Quantitative Behavior Analysis
# Research-paper-style waveform graph
# ============================================================

PROJECT_ROOT = Path(r"D:\AnimalSense_AI")

# Change this if you want to analyze another file
CSV_FILE = PROJECT_ROOT / "outputs" / "final_behavior_history.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "quantitative_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

# Track to analyze
TRACK_ID = None
# None = automatically select the track with the most observations

MEDIAN_WINDOW = 9
LOWPASS_WINDOW = 25

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"CSV file not found:\n{CSV_FILE}"
    )

df = pd.read_csv(CSV_FILE)

print("Loaded:")
print(CSV_FILE)

print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required = [
    "frame",
    "track_id",
    "behavior",
    "confidence"
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

# ------------------------------------------------------------
# SELECT TRACK
# ------------------------------------------------------------

track_counts = (
    df.groupby("track_id")
    .size()
    .sort_values(ascending=False)
)

if track_counts.empty:
    raise ValueError("No track data found.")

if TRACK_ID is None:
    TRACK_ID = track_counts.index[0]

print("\nSelected Track ID:", TRACK_ID)
print("Number of observations:", track_counts.loc[TRACK_ID])

track = df[
    df["track_id"] == TRACK_ID
].copy()

track = track.sort_values("frame")

# ------------------------------------------------------------
# NUMERIC CONVERSION
# ------------------------------------------------------------

track["confidence"] = pd.to_numeric(
    track["confidence"],
    errors="coerce"
)

track = track.dropna(
    subset=["frame", "confidence"]
)

# Convert confidence to percentage
track["confidence_percent"] = (
    track["confidence"] * 100
)

# ------------------------------------------------------------
# ORIGINAL SIGNAL
# ------------------------------------------------------------

original = track["confidence_percent"]

# ------------------------------------------------------------
# MEDIAN FILTER
# ------------------------------------------------------------

median_filtered = (
    original
    .rolling(
        window=MEDIAN_WINDOW,
        center=True,
        min_periods=1
    )
    .median()
)

# ------------------------------------------------------------
# LOW-PASS / MOVING AVERAGE
# ------------------------------------------------------------

lowpass_filtered = (
    median_filtered
    .rolling(
        window=LOWPASS_WINDOW,
        center=True,
        min_periods=1
    )
    .mean()
)

# ------------------------------------------------------------
# SAVE NUMERICAL DATA
# ------------------------------------------------------------

result = pd.DataFrame({
    "frame": track["frame"].values,
    "behavior": track["behavior"].values,
    "original_confidence": original.values,
    "median_filtered": median_filtered.values,
    "lowpass_filtered": lowpass_filtered.values
})

result.to_csv(
    OUTPUT_DIR / f"track_{TRACK_ID}_temporal_signal.csv",
    index=False
)

# ------------------------------------------------------------
# FIGURE 1
# THREE RESEARCH-PAPER-STYLE SIGNALS
# ------------------------------------------------------------

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 9),
    sharex=True
)

# Original
axes[0].plot(
    track["frame"],
    original,
    linewidth=1.0
)

axes[0].set_title(
    "Original Behavioral Confidence Signal"
)

axes[0].set_ylabel(
    "Confidence (%)"
)

axes[0].grid(
    True,
    alpha=0.25
)

# Median
axes[1].plot(
    track["frame"],
    median_filtered,
    linewidth=1.2
)

axes[1].set_title(
    "Behavioral Confidence After Median Filtering"
)

axes[1].set_ylabel(
    "Confidence (%)"
)

axes[1].grid(
    True,
    alpha=0.25
)

# Low-pass
axes[2].plot(
    track["frame"],
    lowpass_filtered,
    linewidth=1.4
)

axes[2].set_title(
    "Behavioral Confidence After Low-Pass Smoothing"
)

axes[2].set_xlabel(
    "Frame"
)

axes[2].set_ylabel(
    "Confidence (%)"
)

axes[2].grid(
    True,
    alpha=0.25
)

fig.suptitle(
    f"Temporal Behavioral Analysis — Track {TRACK_ID}",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

output_file = (
    OUTPUT_DIR /
    f"track_{TRACK_ID}_temporal_analysis.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# FIGURE 2
# ORIGINAL VS SMOOTHED
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    track["frame"],
    original,
    linewidth=0.8,
    alpha=0.45,
    label="Original signal"
)

plt.plot(
    track["frame"],
    median_filtered,
    linewidth=1.2,
    label="Median filtered"
)

plt.plot(
    track["frame"],
    lowpass_filtered,
    linewidth=2.0,
    label="Low-pass filtered"
)

plt.xlabel("Frame")
plt.ylabel("Behavior confidence (%)")

plt.title(
    f"Temporal Behavior Confidence Curve — Track {TRACK_ID}"
)

plt.ylim(0, 105)

plt.grid(
    True,
    alpha=0.25
)

plt.legend()

plt.tight_layout()

output_file_2 = (
    OUTPUT_DIR /
    f"track_{TRACK_ID}_comparison_curve.png"
)

plt.savefig(
    output_file_2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# BEHAVIOR TRANSITION ANALYSIS
# ------------------------------------------------------------

track["previous_behavior"] = (
    track["behavior"].shift(1)
)

transitions = track[
    track["behavior"] !=
    track["previous_behavior"]
].copy()

transitions = transitions[
    ["frame", "behavior", "confidence"]
]

transitions.to_csv(
    OUTPUT_DIR /
    f"track_{TRACK_ID}_behavior_transitions.csv",
    index=False
)

# ------------------------------------------------------------
# SUMMARY STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TEMPORAL QUANTITATIVE ANALYSIS")
print("=" * 60)

print(f"\nTrack ID: {TRACK_ID}")

print(
    f"Observations: {len(track)}"
)

print(
    f"Mean confidence: "
    f"{original.mean():.2f}%"
)

print(
    f"Maximum confidence: "
    f"{original.max():.2f}%"
)

print(
    f"Minimum confidence: "
    f"{original.min():.2f}%"
)

print(
    f"Standard deviation: "
    f"{original.std():.2f}%"
)

print(
    f"Behavior transitions: "
    f"{max(len(transitions) - 1, 0)}"
)

print("\nBehavior distribution:")

behavior_distribution = (
    track["behavior"]
    .value_counts()
)

for behavior, count in behavior_distribution.items():

    percentage = (
        count /
        len(track) *
        100
    )

    print(
        f"{behavior:20s} "
        f"{count:5d} frames "
        f"({percentage:.2f}%)"
    )

# ------------------------------------------------------------
# FINAL OUTPUTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FILES CREATED")
print("=" * 60)

print(
    output_file
)

print(
    output_file_2
)

print(
    OUTPUT_DIR /
    f"track_{TRACK_ID}_temporal_signal.csv"
)

print(
    OUTPUT_DIR /
    f"track_{TRACK_ID}_behavior_transitions.csv"
)

print("\nAnalysis completed successfully.")