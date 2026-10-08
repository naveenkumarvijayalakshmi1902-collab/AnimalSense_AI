from pathlib import Path
from collections import defaultdict

LABEL_DIR = Path(r"dataset/labels/train")

class_names = {
    0: "stand",
    1: "lying_down",
    2: "foraging",
    3: "drinking_water",
    4: "rumination",
}

files_by_class = defaultdict(list)

for label_file in LABEL_DIR.glob("*.txt"):
    for line in label_file.read_text().splitlines():
        if line.strip():
            class_id = int(line.split()[0])
            files_by_class[class_id].append(label_file.stem)
            
print("\n===== IMAGES CONTAINING EACH CLASS =====")

for class_id, name in class_names.items():
    unique_files = sorted(set(files_by_class[class_id]))
    print(f"{name:15s}: {len(unique_files)} images")

print("\n===== RARE CLASS SAMPLE FILES =====")

for class_id in [3, 4]:
    unique_files = sorted(set(files_by_class[class_id]))

    print(f"\n{name if False else class_names[class_id]}:")
    for file in unique_files[:20]:
        print(" ", file)