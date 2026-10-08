import csv
from collections import defaultdict, Counter
from pathlib import Path

INPUT = Path(r"D:\AnimalSense_AI\outputs\cbam_behavior_log.csv")
OUTPUT = Path(r"D:\AnimalSense_AI\outputs\cbam_temporal_history.csv")

history = defaultdict(list)

with INPUT.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        cow_id = row["cow_id"]

        history[cow_id].append({
            "frame": int(row["frame"]),
            "behavior": row["behavior"],
            "confidence": float(row["confidence"])
        })

rows = []

for cow_id, observations in history.items():

    observations.sort(key=lambda x: x["frame"])

    counts = Counter(
        item["behavior"] for item in observations
    )

    total = len(observations)

    for behavior, count in counts.items():

        percentage = (count / total) * 100

        rows.append({
            "cow_id": cow_id,
            "behavior": behavior,
            "observations": count,
            "percentage": round(percentage, 2)
        })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as f:

    fieldnames = [
        "cow_id",
        "behavior",
        "observations",
        "percentage"
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(rows)

print()
print("======================================")
print("CBAM TEMPORAL ANALYSIS COMPLETED")
print("======================================")
print(f"Individual cows tracked: {len(history)}")
print(f"Output file: {OUTPUT}")
print()

for cow_id, observations in sorted(history.items()):
    counts = Counter(x["behavior"] for x in observations)
    print(f"Cow #{cow_id}: {len(observations)} observations")

    for behavior, count in counts.items():
        percentage = (count / len(observations)) * 100
        print(f"   {behavior}: {percentage:.1f}%")

    print()