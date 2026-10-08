import csv
from collections import defaultdict, Counter

INPUT_CSV = r"D:\AnimalSense_AI\outputs\final_behavior_history.csv"
BASELINE_CSV = r"D:\AnimalSense_AI\outputs\adaptive_baseline_final.csv"
OUTPUT_CSV = r"D:\AnimalSense_AI\outputs\final_anomaly_risk.csv"

BEHAVIORS = [
    "stand",
    "lying_down",
    "foraging",
    "drinking_water",
    "rumination"
]

WINDOW_SIZE = 25
MIN_HISTORY = 25

# Deviation thresholds
LOW_THRESHOLD = 20
MEDIUM_THRESHOLD = 40


# ============================================================
# LOAD ADAPTIVE BASELINES
# ============================================================

baselines = {}

with open(BASELINE_CSV, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        track_id = row["track_id"]

        baselines[track_id] = {
            "ready": row["baseline_ready"] == "True",

            "stand": float(row["stand_baseline_pct"]),
            "lying_down": float(row["lying_down_baseline_pct"]),
            "foraging": float(row["foraging_baseline_pct"]),
            "drinking_water": float(row["drinking_water_baseline_pct"]),
            "rumination": float(row["rumination_baseline_pct"])
        }


# ============================================================
# LOAD TRACK HISTORY
# ============================================================

tracks = defaultdict(list)

with open(INPUT_CSV, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        track_id = row["track_id"]

        if track_id == "-1":
            continue

        tracks[track_id].append({
            "frame": int(row["frame"]),
            "time": float(row["timestamp_sec"]),
            "behavior": row["behavior"]
        })


# ============================================================
# ANALYSIS
# ============================================================

results = []

print()
print("======================================")
print("FINAL ANOMALY / RISK ANALYSIS")
print("======================================")


for track_id, observations in tracks.items():

    observations.sort(key=lambda x: x["frame"])

    if track_id not in baselines:
        continue

    baseline = baselines[track_id]

    if not baseline["ready"]:
        print(
            f"Track {track_id}: "
            f"insufficient baseline data"
        )
        continue

    # Need enough data for a current window
    if len(observations) < MIN_HISTORY:
        continue

    # Use the most recent window
    current_window = observations[-WINDOW_SIZE:]

    counts = Counter(
        x["behavior"]
        for x in current_window
    )

    current = {}

    for behavior in BEHAVIORS:

        current[behavior] = (
            counts.get(behavior, 0)
            / len(current_window)
            * 100
        )

    # --------------------------------------------------------
    # Calculate deviation
    # --------------------------------------------------------

    deviations = {}

    for behavior in BEHAVIORS:

        deviations[behavior] = abs(
            current[behavior]
            - baseline[behavior]
        )

    # Average behavioral deviation
    average_deviation = (
        sum(deviations.values())
        / len(BEHAVIORS)
    )

    # Convert deviation to 0-100 risk score
    risk_score = min(
        100,
        average_deviation * 2
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if risk_score >= MEDIUM_THRESHOLD:

        risk_level = "HIGH"

    elif risk_score >= LOW_THRESHOLD:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    # --------------------------------------------------------
    # Find strongest changes
    # --------------------------------------------------------

    ranked = sorted(
        BEHAVIORS,
        key=lambda b: deviations[b],
        reverse=True
    )

    explanations = []

    for behavior in ranked[:3]:

        difference = (
            current[behavior]
            - baseline[behavior]
        )

        if abs(difference) < 5:
            continue

        direction = (
            "increased"
            if difference > 0
            else "decreased"
        )

        explanations.append(
            f"{behavior} {direction} "
            f"by {abs(difference):.1f}%"
        )

    explanation = "; ".join(explanations)

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print(f"Track {track_id}")
    print(f"Risk score: {risk_score:.2f}")
    print(f"Risk level: {risk_level}")

    if explanation:
        print(f"Reason: {explanation}")
    else:
        print("Reason: no major behavioral deviation")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.append([
        track_id,
        len(observations),
        round(risk_score, 2),
        risk_level,

        round(baseline["stand"], 2),
        round(current["stand"], 2),

        round(baseline["lying_down"], 2),
        round(current["lying_down"], 2),

        round(baseline["foraging"], 2),
        round(current["foraging"], 2),

        round(baseline["drinking_water"], 2),
        round(current["drinking_water"], 2),

        round(baseline["rumination"], 2),
        round(current["rumination"], 2),

        explanation
    ])


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "track_id",
        "total_observations",
        "risk_score",
        "risk_level",

        "baseline_stand",
        "current_stand",

        "baseline_lying_down",
        "current_lying_down",

        "baseline_foraging",
        "current_foraging",

        "baseline_drinking_water",
        "current_drinking_water",

        "baseline_rumination",
        "current_rumination",

        "explanation"
    ])

    writer.writerows(results)


print()
print("======================================")
print("ANOMALY ANALYSIS COMPLETED")
print("======================================")
print()
print(f"Saved: {OUTPUT_CSV}")