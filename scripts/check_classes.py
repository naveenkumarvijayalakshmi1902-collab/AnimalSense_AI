from pathlib import Path
from collections import Counter

LABEL_DIR = Path(r"dataset/labels/train")

class_names = {
    0: "stand",
    1: "lying_down",
    2: "foraging",
    3: "drinking_water",
    4: "rumination",
}

counts = Counter()

for file in LABEL_DIR.glob("*.txt"):
    for line in file.read_text().splitlines():
        if line.strip():
            class_id = int(line.split()[0])
            counts[class_id] += 1

print("\n===== TRAINING CLASS DISTRIBUTION =====")

for class_id, name in class_names.items():
    print(f"{class_id}: {name:15s} = {counts[class_id]}")

print("\nTotal annotations:", sum(counts.values()))