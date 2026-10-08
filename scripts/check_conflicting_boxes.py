from pathlib import Path
from collections import defaultdict

LABEL_DIR = Path(r"dataset/labels/train")

conflicting_images = 0
total_conflicts = 0

for label_file in LABEL_DIR.glob("*.txt"):
    boxes = defaultdict(set)

    for line in label_file.read_text().splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        cls = int(parts[0])
        coords = tuple(round(float(x), 5) for x in parts[1:5])

        boxes[coords].add(cls)

    conflicts = {
        box: classes
        for box, classes in boxes.items()
        if len(classes) > 1
    }

    if conflicts:
        conflicting_images += 1
        total_conflicts += len(conflicts)

print("\n===== CONFLICTING BOUNDING BOX ANALYSIS =====")
print("Images with conflicting boxes:", conflicting_images)
print("Total conflicting boxes:", total_conflicts)