"""
Step 1: Verify dataset_slake structure before training
Checks that PathVQA, SLAKE, and Reject images are in place.

Expected layout inside this folder:
    dataset_pathvqa/
        train/train.csv
        validation/validation.csv
    dataset_slake/
        train/train.csv
        val/val.csv
    dataset_reject/
        images/   (non-medical images for reject class)

Copy your datasets here before running 2_train.py.
"""
from pathlib import Path
import csv
import os

PROJECT_ROOT = Path(__file__).parent.resolve()

PATHVQA_TRAIN_CSV = PROJECT_ROOT / "dataset_pathvqa" / "train"      / "train.csv"
PATHVQA_VAL_CSV   = PROJECT_ROOT / "dataset_pathvqa" / "validation" / "validation.csv"
SLAKE_TRAIN_CSV   = PROJECT_ROOT / "dataset_slake"   / "train"      / "train.csv"
SLAKE_VAL_CSV     = PROJECT_ROOT / "dataset_slake"   / "val"        / "val.csv"
REJECT_DIR        = PROJECT_ROOT / "dataset_reject"   / "images"

print("=" * 60)
print("MODALITY CLASSIFIER — STEP 1: VERIFY DATASETS")
print("=" * 60)


def count_unique_images(csv_path: Path, label: str) -> int:
    if not csv_path.exists():
        print(f"  [MISSING] {csv_path}")
        return 0

    seen = set()
    missing = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            path = row["image_path"]
            if path not in seen:
                seen.add(path)
                if not os.path.exists(path):
                    missing += 1

    print(f"  {label}: {len(seen)} unique images  ({missing} missing files)")
    return len(seen)


def count_reject_images(reject_dir: Path) -> int:
    if not reject_dir.exists():
        print(f"  [MISSING] {reject_dir}")
        return 0

    exts = {".jpg", ".jpeg", ".png"}
    images = [f for f in reject_dir.iterdir() if f.suffix.lower() in exts]
    print(f"  Reject: {len(images)} images")
    return len(images)


print("\n--- Training split ---")
p_train = count_unique_images(PATHVQA_TRAIN_CSV, "PathVQA (pathology)")
s_train = count_unique_images(SLAKE_TRAIN_CSV,   "SLAKE   (radiology)")

print("\n--- Validation split ---")
p_val = count_unique_images(PATHVQA_VAL_CSV, "PathVQA (pathology)")
s_val = count_unique_images(SLAKE_VAL_CSV,   "SLAKE   (radiology)")

print("\n--- Reject images (shared) ---")
r_count = count_reject_images(REJECT_DIR)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Train — Pathology: {p_train}  |  Radiology: {s_train}  |  Reject: {int(r_count * 0.9):.0f} (90%)")
print(f"  Val   — Pathology: {p_val}  |  Radiology: {s_val}  |  Reject: {int(r_count * 0.1):.0f} (10%)")

all_present = all([
    PATHVQA_TRAIN_CSV.exists(), PATHVQA_VAL_CSV.exists(),
    SLAKE_TRAIN_CSV.exists(), SLAKE_VAL_CSV.exists(),
    REJECT_DIR.exists(),
])

print()
if all_present:
    print("All datasets found. Ready to run 2_train.py")
else:
    print("Some datasets are missing. Copy them into this folder before training.")
print("=" * 60)
