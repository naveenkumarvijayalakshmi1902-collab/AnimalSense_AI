import csv
from collections import defaultdict

INPUT_CSV = r"D:\AnimalSense_AI\outputs\final_behavior_history.csv"
OUTPUT_CSV = r"D:\AnimalSense_AI\outputs\adaptive_baseline_final.csv"

BEHAVIORS = [
    "stand",
    "lying_down",
    "foraging",
    "drinking_water",
    "rumination"
]

# Number of observations required before a baseline is considered usable
MIN_OBSERVATIONS = 25

# EMA smoothing factor
ALPHA = 0.10

# Track -> behavior -> adaptive probability
baseline = defaultdict(
    lambda: {behavior: 0.0 for behavior in BEHAVIORS}
)

counts = defaultdict(int)

with open(INPUT_CSV, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        track_id = row["track_id"]

        if track_id == "-1":
            continue

        behavior = row["behavior"]

        if behavior not in BEHAVIORS:
            continue

        counts[track_id] += 1

        # Current observation as one-hot vector
        for b in BEHAVIORS:

            observed = 1.0 if b == behavior else 0.0

            old_value = baseline[track_id][b]

            baseline[track_id][b] = (
                ALPHA * observed
                + (1 - ALPHA) * old_value
            )

print()
print("======================================")
print("ADAPTIVE BASELINE")
print("======================================")

rows = []

for track_id in sorted(
    baseline.keys(),
    key=lambda x: int(x)
):

    total = counts[track_id]

    ready = total >= MIN_OBSERVATIONS

    print()
    print(f"Track {track_id}")
    print(f"Observations: {total}")
    print(f"Baseline ready: {ready}")

    values = []

    for behavior in BEHAVIORS:

        value = baseline[track_id][behavior]

        values.append(value)

        print(
            f"  {behavior:16s}: "
            f"{value * 100:6.2f}%"
        )

    rows.append([
        track_id,
        total,
        ready,
        round(values[0] * 100, 2),
        round(values[1] * 100, 2),
        round(values[2] * 100, 2),
        round(values[3] * 100, 2),
        round(values[4] * 100, 2)
    ])

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "track_id",
        "observations",
        "baseline_ready",
        "stand_baseline_pct",
        "lying_down_baseline_pct",
        "foraging_baseline_pct",
        "drinking_water_baseline_pct",
        "rumination_baseline_pct"
    ])

    writer.writerows(rows)

print()
print("======================================")
print("ADAPTIVE BASELINE COMPLETED")
print("======================================")
print()
print(f"Saved: {OUTPUT_CSV}")