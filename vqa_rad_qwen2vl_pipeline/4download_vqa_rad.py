from datasets import load_dataset
from tqdm import tqdm
import os
from pathlib import Path
from PIL import Image
import pandas as pd

# Load dataset_slake
ds = load_dataset("flaviagiammarino/vqa-rad")

# Export folder
export_root = "dataset_vqa_rad"
os.makedirs(export_root, exist_ok=True)

def export_split(data, split_name):
    split_dir = os.path.join(export_root, split_name)
    img_dir = os.path.join(split_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    for i, sample in enumerate(tqdm(data, desc=f"Exporting {split_name}")):
        img = sample["image"].convert("RGB")
        q = sample["question"].strip()
        a = sample["answer"].strip()

        filename = f"{split_name}_{i:05d}.jpg"
        filepath = os.path.join(img_dir, filename)
        if not os.path.exists(filepath):
            img.save(filepath, quality=95, optimize=True)

        records.append({
            "image_path": Path(filepath).as_posix(),
            "question": q,
            "answer": a
        })

    pd.DataFrame(records).to_csv(
        os.path.join(split_dir, f"{split_name}.csv"),
        index=False, encoding="utf-8-sig"
    )
    return len(records)

# Split train into train/val (90/10)
train_data = ds["train"].train_test_split(test_size=0.1, seed=42)

train_count = export_split(train_data["train"], "train")
val_count = export_split(train_data["test"], "validation")
test_count = export_split(ds["test"], "test")

print(f"\nExport complete:")
print(f"  Train: {train_count}")
print(f"  Val: {val_count}")
print(f"  Test: {test_count}")