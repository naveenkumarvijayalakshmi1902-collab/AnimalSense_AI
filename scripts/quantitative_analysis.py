from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# AnimalSense AI - Quantitative Analysis
# ============================================================

PROJECT_ROOT = Path(r"D:\AnimalSense_AI")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "quantitative_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. MODEL PERFORMANCE DATA
# ------------------------------------------------------------

models = [
    "YOLOv8n",
    "YOLOv8n + CBAM"
]

precision = [
    70.0,
    52.2
]

recall = [
    60.0,
    47.1
]

map50 = [
    64.6,
    48.9
]

map50_95 = [
    34.7,
    23.6
]


performance_df = pd.DataFrame({
    "Model": models,
    "Precision": precision,
    "Recall": recall,
    "mAP50": map50,
    "mAP50-95": map50_95
})

performance_df.to_csv(
    OUTPUT_DIR / "model_performance.csv",
    index=False
)


# ------------------------------------------------------------
# 2. GRAPH: MODEL PERFORMANCE COMPARISON
# ------------------------------------------------------------

x = range(len(models))
width = 0.2

plt.figure(figsize=(10, 6))

plt.bar(
    [i - 1.5 * width for i in x],
    precision,
    width,
    label="Precision"
)

plt.bar(
    [i - 0.5 * width for i in x],
    recall,
    width,
    label="Recall"
)

plt.bar(
    [i + 0.5 * width for i in x],
    map50,
    width,
    label="mAP@50"
)

plt.bar(
    [i + 1.5 * width for i in x],
    map50_95,
    width,
    label="mAP@50-95"
)

plt.xticks(x, models)
plt.ylabel("Percentage (%)")
plt.xlabel("Model")
plt.title("YOLOv8 Model Performance Comparison")
plt.ylim(0, 100)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "01_model_performance_comparison.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 3. PER-BEHAVIOR mAP@50
# ------------------------------------------------------------

behaviors = [
    "Standing",
    "Lying Down",
    "Foraging",
    "Drinking Water",
    "Rumination"
]

behavior_map50 = [
    83.5,
    83.4,
    81.5,
    25.2,
    49.4
]

behavior_df = pd.DataFrame({
    "Behavior": behaviors,
    "mAP50": behavior_map50
})

behavior_df.to_csv(
    OUTPUT_DIR / "behavior_map50.csv",
    index=False
)


# ------------------------------------------------------------
# 4. GRAPH: PER-BEHAVIOR mAP@50
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

bars = plt.bar(
    behaviors,
    behavior_map50
)

plt.ylabel("mAP@50 (%)")
plt.xlabel("Cattle Behavior")
plt.title("Behavior-wise Detection Performance")
plt.ylim(0, 100)
plt.xticks(rotation=20)
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, behavior_map50):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "02_behavior_wise_map50.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 5. PER-BEHAVIOR PRECISION AND RECALL
# ------------------------------------------------------------

behavior_precision = [
    78.7,
    73.7,
    72.3,
    62.2,
    62.9
]

behavior_recall = [
    76.8,
    79.4,
    80.8,
    16.2,
    46.7
]

behavior_pr_df = pd.DataFrame({
    "Behavior": behaviors,
    "Precision": behavior_precision,
    "Recall": behavior_recall
})

behavior_pr_df.to_csv(
    OUTPUT_DIR / "behavior_precision_recall.csv",
    index=False
)


# ------------------------------------------------------------
# 6. GRAPH: BEHAVIOR PRECISION VS RECALL
# ------------------------------------------------------------

x = range(len(behaviors))
width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    [i - width / 2 for i in x],
    behavior_precision,
    width,
    label="Precision"
)

plt.bar(
    [i + width / 2 for i in x],
    behavior_recall,
    width,
    label="Recall"
)

plt.xticks(x, behaviors, rotation=20)
plt.ylabel("Percentage (%)")
plt.xlabel("Behavior")
plt.title("Behavior-wise Precision and Recall")
plt.ylim(0, 100)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "03_behavior_precision_recall.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 7. BEHAVIOR DISTRIBUTION
# ------------------------------------------------------------

# These are example aggregate proportions from the analyzed
# behavior histories. Replace them with your complete batch
# values when available.

behavior_distribution = {
    "Standing": 45,
    "Foraging": 35,
    "Lying Down": 10,
    "Rumination": 7,
    "Drinking Water": 3
}

distribution_df = pd.DataFrame({
    "Behavior": list(behavior_distribution.keys()),
    "Percentage": list(behavior_distribution.values())
})

distribution_df.to_csv(
    OUTPUT_DIR / "behavior_distribution.csv",
    index=False
)


# ------------------------------------------------------------
# 8. GRAPH: BEHAVIOR DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 8))

plt.pie(
    behavior_distribution.values(),
    labels=behavior_distribution.keys(),
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Observed Cattle Behavior Distribution")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "04_behavior_distribution.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 9. ANOMALY RISK ANALYSIS
# ------------------------------------------------------------

risk_file = (
    PROJECT_ROOT
    / "outputs"
    / "final_anomaly_risk.csv"
)

if risk_file.exists():

    risk_df = pd.read_csv(risk_file)

    print("\nLoaded anomaly file:")
    print(risk_file)

    print("\nColumns:")
    print(list(risk_df.columns))

    # Try to locate risk score column automatically
    score_column = None

    possible_score_columns = [
        "risk_score",
        "score",
        "anomaly_score",
        "risk"
    ]

    for col in possible_score_columns:
        if col in risk_df.columns:
            score_column = col
            break

    if score_column is not None:

        # Convert to numeric
        risk_df[score_column] = pd.to_numeric(
            risk_df[score_column],
            errors="coerce"
        )

        risk_df = risk_df.dropna(
            subset=[score_column]
        )

        # ----------------------------------------------------
        # GRAPH: RISK SCORE BY TRACK
        # ----------------------------------------------------

        plt.figure(figsize=(12, 6))

        plt.bar(
            risk_df.index.astype(str),
            risk_df[score_column]
        )

        plt.xlabel("Track")
        plt.ylabel("Risk Score")
        plt.title("Behavioral Anomaly Risk Score by Track")
        plt.xticks(rotation=90)

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "05_anomaly_risk_by_track.png",
            dpi=300
        )

        plt.close()

        # ----------------------------------------------------
        # SAVE SORTED RISK DATA
        # ----------------------------------------------------

        risk_df.sort_values(
            score_column,
            ascending=False
        ).to_csv(
            OUTPUT_DIR / "risk_scores_sorted.csv",
            index=False
        )

    else:
        print(
            "\nWARNING: No recognized risk-score column found."
        )

else:

    print(
        "\nWARNING: final_anomaly_risk.csv was not found."
    )


# ------------------------------------------------------------
# 10. RISK LEVEL DISTRIBUTION
# ------------------------------------------------------------

if risk_file.exists():

    risk_df = pd.read_csv(risk_file)

    level_column = None

    possible_level_columns = [
        "risk_level",
        "level",
        "risk"
    ]

    for col in possible_level_columns:
        if col in risk_df.columns:
            # Only use it if it contains categorical risk levels
            values = risk_df[col].astype(str).str.upper()

            if values.isin(
                ["LOW", "MEDIUM", "HIGH"]
            ).any():
                level_column = col
                break

    if level_column is not None:

        risk_levels = (
            risk_df[level_column]
            .astype(str)
            .str.upper()
            .value_counts()
        )

        # Ensure all categories exist
        low = risk_levels.get("LOW", 0)
        medium = risk_levels.get("MEDIUM", 0)
        high = risk_levels.get("HIGH", 0)

        risk_distribution = {
            "LOW": low,
            "MEDIUM": medium,
            "HIGH": high
        }

        plt.figure(figsize=(8, 6))

        bars = plt.bar(
            risk_distribution.keys(),
            risk_distribution.values()
        )

        plt.xlabel("Risk Level")
        plt.ylabel("Number of Tracks")
        plt.title("Behavioral Risk Level Distribution")

        for bar, value in zip(
            bars,
            risk_distribution.values()
        ):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.1,
                str(value),
                ha="center",
                va="bottom",
                fontweight="bold"
            )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "06_risk_level_distribution.png",
            dpi=300
        )

        plt.close()

        pd.DataFrame({
            "Risk Level": list(risk_distribution.keys()),
            "Count": list(risk_distribution.values())
        }).to_csv(
            OUTPUT_DIR / "risk_level_distribution.csv",
            index=False
        )


# ------------------------------------------------------------
# 11. SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("QUANTITATIVE ANALYSIS COMPLETED")
print("=" * 60)

print(f"\nOutput folder:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file.name)

print("\nImportant presentation metrics:")
print("--------------------------------")
print("Baseline YOLOv8n Precision : 70.0%")
print("Baseline YOLOv8n Recall    : 60.0%")
print("Baseline YOLOv8n mAP@50    : 64.6%")
print("Baseline YOLOv8n mAP@50-95 : 34.7%")

print("\nBehavior-wise mAP@50:")
for behavior, score in zip(
    behaviors,
    behavior_map50
):
    print(f"{behavior:20s}: {score:.1f}%")

print("\nAnalysis finished.")