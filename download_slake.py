"""
download_slake.py
Downloads SLAKE-vqa-english and saves to disk in same format as VQA-RAD.

Output structure:
    dataset_slake/
        train/
            images/
            train.csv
        validation/
            images/
            validation.csv
        test/
            images/
            test.csv

Run: python download_slake.py
"""

import os
import csv
from datasets import load_dataset
from PIL import Image

SPLITS = {
    "train":      "dataset_slake/train",
    "validation": "dataset_slake/validation",
    "test":       "dataset_slake/test",
}

# Create output folders
for split, folder in SPLITS.items():
    os.makedirs(os.path.join(folder, "images"), exist_ok=True)

ds = load_dataset("mdwiratathya/SLAKE-vqa-english")

for split, folder in SPLITS.items():
    print(f"\nProcessing {split}...")

    images_dir = os.path.join(folder, "images")
    csv_path   = os.path.join(folder, f"{split}.csv")

    # Track saved images to avoid duplicates (same image, multiple QA pairs)
    saved_images = {}
    rows = []
    img_counter = 0

    for item in ds[split]:
        img   = item["image"]
        question = item["question"]
        answer   = item["answer"]

        # Use hash of image to detect duplicates
        img_hash = hash(img.tobytes())

        if img_hash not in saved_images:
            img_filename = f"slake_{img_counter:05d}.jpg"
            img_path = os.path.join(images_dir, img_filename)

            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path)

            saved_images[img_hash] = os.path.join(folder, "images", img_filename)
            img_counter += 1

        rows.append({
            "image_path": saved_images[img_hash],
            "question":   question,
            "answer":     answer
        })

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "question", "answer"])
        writer.writeheader()
        writer.writerows(rows)

    unique_images = len(saved_images)
    print(f"  ✓ {unique_images} unique images saved")
    print(f"  ✓ {len(rows)} QA pairs saved to {csv_path}")

print("\nDone! SLAKE dataset ready.")