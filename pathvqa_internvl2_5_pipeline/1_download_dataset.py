"""
Step 1: Download PathVQA dataset from HuggingFace
Uses native train / validation / test splits.
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = (PROJECT_ROOT / ".hf_cache").resolve()
os.environ["HF_HOME"] = str(CACHE_DIR)

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "flaviagiammarino/path-vqa"
OUTPUT_DIR = PROJECT_ROOT / "dataset"
SPLITS = ["train", "validation", "test"]

print("=" * 60)
print("PATH-VQA DATASET DOWNLOAD")
print("=" * 60)
print(f"Dataset:    {DATASET_ID}")
print(f"Output dir: {OUTPUT_DIR}")
print("=" * 60 + "\n")

all_csvs = [OUTPUT_DIR / s / f"{s}.csv" for s in SPLITS]
if all(p.exists() for p in all_csvs):
    print("Dataset already downloaded.\n")
    for csv_path in all_csvs:
        df = pd.read_csv(csv_path)
        yn = df["answer"].str.lower().str.strip().isin(["yes", "no"]).sum()
        split = csv_path.stem
        print(f"  {split:10s}: {len(df):6d} samples  (yes/no: {yn:5d} | open: {len(df)-yn:5d})")
    print(f"\nOutput: {OUTPUT_DIR}/")
    print("Next: run 2_preprocess.py")
    print("=" * 60)
    raise SystemExit(0)

print("Downloading dataset...")
dataset = load_dataset(DATASET_ID, cache_dir=str(CACHE_DIR))
print(f"Available splits: {list(dataset.keys())}")

for split_name in SPLITS:
    split_data = dataset[split_name]
    split_dir = OUTPUT_DIR / split_name
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, sample in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
        image_path = images_dir / f"{split_name}_{idx:05d}.jpg"
        sample["image"].convert("RGB").save(image_path, quality=95)
        rows.append({
            "question": sample["question"].strip(),
            "answer": sample["answer"].strip(),
            "image_path": str(image_path),
        })

    df = pd.DataFrame(rows)
    csv_path = split_dir / f"{split_name}.csv"
    df.to_csv(csv_path, index=False)
    yn = df["answer"].str.lower().str.strip().isin(["yes", "no"]).sum()
    print(f"  {split_name:10s}: {len(df):6d} samples  (yes/no: {yn:5d} | open: {len(df)-yn:5d})")

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE!")
print("=" * 60)
print(f"Output: {OUTPUT_DIR}/")
print("Next: run 2_preprocess.py")
print("=" * 60)
