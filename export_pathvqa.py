from datasets import load_dataset
from tqdm import tqdm
import os
from pathlib import Path
from PIL import Image
import pandas as pd

# 1) Import first, before creating any folder
ds = load_dataset("flaviagiammarino/path-vqa")

# 2) Export folder name must NOT be "datasets"
export_root = "dataset_pathvqa"
os.makedirs(export_root, exist_ok=True)

def export_split(split_name):
    split_dir = os.path.join(export_root, split_name)
    img_dir = os.path.join(split_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    for i, sample in enumerate(tqdm(ds[split_name], desc=f"Exporting {split_name}")):
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

for split in ds.keys():  # train, validation, test
    export_split(split)
