import csv
from collections import defaultdict, Counter
from pathlib import Path


INPUT = Path(
    r"D:\AnimalSense_AI\outputs\stable_cow_behavior_log.csv"
)

OUTPUT = Path(
    r"D:\AnimalSense_AI\outputs\adaptive_baseline.csv"
)

BEHAVIORS = [
    "stand",
    "lying_down",
    "foraging",
    "drinking_water",
    "rumination"
]

# Number of observations required before creating
# an individual baseline.
MIN_OBSERVATIONS = 20

# EMA learning rate.
# Smaller = more stable baseline.
ALPHA = 0.10


# --------------------------------------------------
# Load observations
# --------------------------------------------------

cow_history = defaultdict(list)

with INPUT.open("r", newline="", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        cow_history[row["cow_id"]].append({
            "frame": int(row["frame"]),
            "behavior": row["behavior"],
            "confidence": float(row["confidence"])
        })


# --------------------------------------------------
# Build adaptive baseline
# --------------------------------------------------

results = []

for cow_id, observations in cow_history.items():

    observations.sort(
        key=lambda x: x["frame"]
    )

    # Initial baseline
    baseline = {
        behavior: 0.0
        for behavior in BEHAVIORS
    }

    total = 0

    for observation in observations:

        behavior = observation["behavior"]

        if behavior not in baseline:
            continue

        total += 1

        # Current one-hot behavior vector
        current = {
            b: 1.0 if b == behavior else 0.0
            for b in BEHAVIORS
        }

        # Adaptive exponential moving average
        for b in BEHAVIORS:

            baseline[b] = (
                ALPHA * current[b]
                +
                (1 - ALPHA) * baseline[b]
            )


    if total < MIN_OBSERVATIONS:
        status = "INSUFFICIENT_DATA"
    else:
        status = "BASELINE_READY"


    # Convert to percentages
    baseline_percent = {
        b: round(baseline[b] * 100, 2)
        for b in BEHAVIORS
    }


    results.append({
        "cow_id": cow_id,
        "observations": total,
        "status": status,
        "stand_pct": baseline_percent["stand"],
        "lying_down_pct": baseline_percent["lying_down"],
        "foraging_pct": baseline_percent["foraging"],
        "drinking_water_pct": baseline_percent["drinking_water"],
        "rumination_pct": baseline_percent["rumination"]
    })


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "cow_id",
        "observations",
        "status",
        "stand_pct",
        "lying_down_pct",
        "foraging_pct",
        "drinking_water_pct",
        "rumination_pct"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(results)


# --------------------------------------------------
# Display
# --------------------------------------------------

print()
print("==========================================")
print("ADAPTIVE INDIVIDUAL-COW BASELINE")
print("==========================================")

for row in results:

    print()
    print(
        f"Cow #{row['cow_id']} "
        f"({row['observations']} observations)"
    )

    print(
        f"Status: {row['status']}"
    )

    print(
        f"  Stand:          {row['stand_pct']}%"
    )

    print(
        f"  Lying down:     {row['lying_down_pct']}%"
    )

    print(
        f"  Foraging:       {row['foraging_pct']}%"
    )

    print(
        f"  Drinking water: {row['drinking_water_pct']}%"
    )

    print(
        f"  Rumination:     {row['rumination_pct']}%"
    )


print()
print(f"Output: {OUTPUT}")