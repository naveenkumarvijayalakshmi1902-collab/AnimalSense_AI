import csv
from collections import defaultdict, Counter
from pathlib import Path

INPUT_CSV = Path(
    r"D:\AnimalSense_AI\outputs\stage4_behavior_log.csv"
)

OUTPUT_CSV = Path(
    r"D:\AnimalSense_AI\outputs\stage5_anomaly_report.csv"
)

# Read tracking results
cow_behaviors = defaultdict(list)

with open(INPUT_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        cow_id = row["cow_id"]
        behavior = row["behavior"]

        cow_behaviors[cow_id].append(behavior)


def calculate_risk(counts):
    total = sum(counts.values())

    if total == 0:
        return "LOW", 0

    lying = counts.get("lying_down", 0) / total
    foraging = counts.get("foraging", 0) / total
    rumination = counts.get("rumination", 0) / total
    drinking = counts.get("drinking_water", 0) / total

    # Simple behavioral anomaly score.
    # This is NOT a disease diagnosis.

    score = 0

    if lying > 0.70:
        score += 40

    if foraging < 0.05:
        score += 25

    if rumination < 0.05:
        score += 25

    if drinking < 0.02:
        score += 10

    if score >= 60:
        risk = "HIGH"
    elif score >= 30:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return risk, score


with open(OUTPUT_CSV, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "cow_id",
        "total_observations",
        "stand_percent",
        "lying_percent",
        "foraging_percent",
        "drinking_percent",
        "rumination_percent",
        "risk",
        "risk_score"
    ])

    for cow_id, behaviors in cow_behaviors.items():

        counts = Counter(behaviors)

        total = len(behaviors)

        stand_pct = counts["stand"] / total * 100
        lying_pct = counts["lying_down"] / total * 100
        foraging_pct = counts["foraging"] / total * 100
        drinking_pct = counts["drinking_water"] / total * 100
        rumination_pct = counts["rumination"] / total * 100

        risk, score = calculate_risk(counts)

        writer.writerow([
            cow_id,
            total,
            round(stand_pct, 2),
            round(lying_pct, 2),
            round(foraging_pct, 2),
            round(drinking_pct, 2),
            round(rumination_pct, 2),
            risk,
            score
        ])

        print(
            f"Cow #{cow_id}: "
            f"risk={risk}, "
            f"score={score}"
        )

print()
print("STAGE 5 COMPLETE")
print(f"Report: {OUTPUT_CSV}")