"""
Step 1: Download VQA-RAD dataset from Hugging Face
Saves images + CSV files for train / val / test splits

Output structure:
    dataset/
        train/      train.csv + images/
        val/        val.csv   + images/
        test/       test.csv  + images/
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = (PROJECT_ROOT.parent / ".hf_cache").resolve()
os.environ["HF_HOME"] = str(CACHE_DIR)

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# ========== CONFIG ==========
DATASET_ID = "flaviagiammarino/vqa-rad"
OUTPUT_DIR = PROJECT_ROOT / "dataset"
VAL_RATIO = 0.1
SEED = 42

print("=" * 60)
print("VQA-RAD DATASET DOWNLOAD")
print("=" * 60)
print(f"Dataset:    {DATASET_ID}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Val split:  {VAL_RATIO * 100:.0f}% of train")
print("=" * 60 + "\n")

print("Downloading dataset...")
dataset = load_dataset(DATASET_ID, cache_dir=str(CACHE_DIR))
print(f"Available splits: {list(dataset.keys())}")

train_val = dataset["train"].train_test_split(test_size=VAL_RATIO, seed=SEED)
splits = {
    "train": train_val["train"],
    "val": train_val["test"],
    "test": dataset["test"],
}

for split_name, split_data in splits.items():
    split_dir = OUTPUT_DIR / split_name
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, sample in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
        image_path = images_dir / f"{idx:05d}.jpg"
        sample["image"].convert("RGB").save(image_path, quality=95)
        rows.append({
            "question": sample["question"],
            "answer": sample["answer"],
            "image_path": str(image_path),
        })

    df = pd.DataFrame(rows)
    csv_path = split_dir / f"{split_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  {split_name}: {len(df)} samples -> {csv_path}")

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE!")
print("=" * 60)
for split_name in splits:
    csv_path = OUTPUT_DIR / split_name / f"{split_name}.csv"
    df = pd.read_csv(csv_path)
    print(f"  {split_name:5s}: {len(df)} samples")
print(f"\nOutput: {OUTPUT_DIR}/")
print("Next: run 2_preprocess.py")
print("=" * 60)
