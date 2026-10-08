import csv
from collections import defaultdict, Counter

INPUT_CSV = r"D:\AnimalSense_AI\outputs\final_behavior_history.csv"
OUTPUT_CSV = r"D:\AnimalSense_AI\outputs\temporal_behavior_summary.csv"

BEHAVIORS = [
    "stand",
    "lying_down",
    "foraging",
    "drinking_water",
    "rumination"
]

# track_id -> list of observations
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
            "behavior": row["behavior"],
            "confidence": float(row["confidence"])
        })

print()
print("======================================")
print("TEMPORAL BEHAVIOR ANALYSIS")
print("======================================")

rows = []

for track_id, observations in tracks.items():

    observations.sort(key=lambda x: x["frame"])

    total = len(observations)

    counter = Counter(
        x["behavior"] for x in observations
    )

    print()
    print(f"Track {track_id}")
    print(f"Observations: {total}")

    percentages = {}

    for behavior in BEHAVIORS:

        count = counter.get(behavior, 0)

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        percentages[behavior] = percentage

        print(
            f"  {behavior:16s}: "
            f"{percentage:6.2f}%"
        )

    # Behavior transitions
    transitions = 0

    for i in range(1, len(observations)):

        previous = observations[i - 1]["behavior"]
        current = observations[i]["behavior"]

        if previous != current:
            transitions += 1

    duration = 0

    if observations:
        duration = (
            observations[-1]["time"]
            - observations[0]["time"]
        )

    rows.append([
        track_id,
        total,
        round(duration, 2),
        round(percentages["stand"], 2),
        round(percentages["lying_down"], 2),
        round(percentages["foraging"], 2),
        round(percentages["drinking_water"], 2),
        round(percentages["rumination"], 2),
        transitions
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
        "duration_sec",
        "stand_pct",
        "lying_down_pct",
        "foraging_pct",
        "drinking_water_pct",
        "rumination_pct",
        "behavior_transitions"
    ])

    writer.writerows(rows)

print()
print("======================================")
print("TEMPORAL ANALYSIS COMPLETED")
print("======================================")
print()
print(f"Saved: {OUTPUT_CSV}")