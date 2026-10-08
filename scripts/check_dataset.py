from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

print("AnimalSense AI dataset check")

for split in ["train", "val", "test"]:

    image_dir = ROOT / "dataset" / "images" / split
    label_dir = ROOT / "dataset" / "labels" / split

    images = list(image_dir.glob("*.jpg")) if image_dir.exists() else []

    missing = 0
    invalid = 0

    for image in images:

        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            missing += 1
            continue

        for line in label.read_text().splitlines():

            parts = line.strip().split()

            if len(parts) != 5:
                invalid += 1
                continue

            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])

                if not 0 <= cls <= 4:
                    invalid += 1
                    continue

                if not all(0 <= x <= 1 for x in [cx, cy, w, h]):
                    invalid += 1
                    continue

                if w <= 0 or h <= 0:
                    invalid += 1

            except ValueError:
                invalid += 1

    print(
        f"{split:>5}: images={len(images):6d} "
        f"missing_labels={missing:5d} "
        f"invalid_labels={invalid:5d}"
    )

print("\nExpected YOLO label:")
print("class_id cx cy width height")
print("class IDs: 0=stand, 1=lying, 2=foraging, 3=drinking, 4=rumination")