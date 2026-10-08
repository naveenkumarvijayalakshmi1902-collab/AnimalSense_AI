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

combination_counts = Counter()

for label_file in LABEL_DIR.glob("*.txt"):
    classes = set()

    for line in label_file.read_text().splitlines():
        if line.strip():
            class_id = int(line.split()[0])
            classes.add(class_id)

    if classes:
        combination = tuple(sorted(classes))
        combination_counts[combination] += 1

print("\n===== LABEL COMBINATIONS PER IMAGE =====\n")

for combination, count in combination_counts.most_common():
    names = [class_names[c] for c in combination]
    print(f"{count:5d} images : {', '.join(names)}")