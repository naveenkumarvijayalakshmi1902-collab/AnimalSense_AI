from pathlib import Path
import shutil
import random

ROOT = Path("dataset")
IMG_SRC = ROOT / "images" / "train"
LBL_SRC = ROOT / "labels" / "train"

IMG_TRAIN = ROOT / "images" / "train"
IMG_VAL = ROOT / "images" / "val"
IMG_TEST = ROOT / "images" / "test"

LBL_TRAIN = ROOT / "labels" / "train"
LBL_VAL = ROOT / "labels" / "val"
LBL_TEST = ROOT / "labels" / "test"

for p in [IMG_VAL, IMG_TEST, LBL_VAL, LBL_TEST]:
    p.mkdir(parents=True, exist_ok=True)

images = list(IMG_SRC.glob("*.jpg"))

# Group frames by video ID.
# Expected filename pattern: videoID_timestamp.jpg
groups = {}

for img in images:
    video_id = img.stem.split("_")[0]
    groups.setdefault(video_id, []).append(img)

video_ids = list(groups.keys())

random.seed(42)
random.shuffle(video_ids)

n = len(video_ids)

val_end = int(n * 0.10)
test_end = int(n * 0.20)

val_ids = set(video_ids[:val_end])
test_ids = set(video_ids[val_end:test_end])

print(f"Total videos: {n}")
print(f"Validation videos: {len(val_ids)}")
print(f"Test videos: {len(test_ids)}")

val_count = 0
test_count = 0

for video_id, imgs in groups.items():

    if video_id in val_ids:
        dst_img = IMG_VAL
        dst_lbl = LBL_VAL
        val_count += len(imgs)

    elif video_id in test_ids:
        dst_img = IMG_TEST
        dst_lbl = LBL_TEST
        test_count += len(imgs)

    else:
        continue

    for img in imgs:
        label = LBL_SRC / f"{img.stem}.txt"

        shutil.copy2(img, dst_img / img.name)

        if label.exists():
            shutil.copy2(label, dst_lbl / label.name)

print()
print(f"Validation images: {val_count}")
print(f"Test images: {test_count}")
print("Dataset split completed.")