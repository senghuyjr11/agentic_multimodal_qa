"""
Step 1: Download PathVQA dataset from HuggingFace
=========================================================
Dataset : flaviagiammarino/path-vqa
Splits  : train (19,654) | validation (6,259) | test (6,719)
          — native HuggingFace splits, NO manual re-splitting —
Content : ~50% yes/no questions, ~50% open-ended
          Pathology microscopy images (H&E, IHC, etc.)

Output layout:
  dataset/
    train/
      train.csv          (image_path, question, answer)
      images/
    validation/
      validation.csv
      images/
    test/
      test.csv
      images/
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR    = (PROJECT_ROOT / ".hf_cache").resolve()
os.environ["HF_HOME"] = str(CACHE_DIR)

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ========== CONFIG ==========
DATASET_ID = "flaviagiammarino/path-vqa"
OUTPUT_DIR = PROJECT_ROOT / "dataset"

print("=" * 60)
print("PATH-VQA  —  STEP 1: DOWNLOAD")
print("=" * 60)
print(f"Dataset ID : {DATASET_ID}")
print(f"Output     : {OUTPUT_DIR}")
print(f"HF cache   : {CACHE_DIR}")
print("Splits     : train / validation / test  (native, no re-splitting)")
print("=" * 60 + "\n")

# Skip if already downloaded
all_csvs = [OUTPUT_DIR / s / f"{s}.csv" for s in ["train", "validation", "test"]]
if all(p.exists() for p in all_csvs):
    print("Dataset already downloaded. Summary:\n")
    for csv_path in all_csvs:
        df      = pd.read_csv(csv_path)
        yn      = df["answer"].str.lower().str.strip().isin(["yes", "no"]).sum()
        split   = csv_path.stem
        print(f"  {split:10s}: {len(df):6d} samples  "
              f"(yes/no: {yn:5d} | open: {len(df)-yn:5d})")
    print(f"\nOutput : {OUTPUT_DIR}/")
    print("Next   : run 2_preprocess.py")
    print("=" * 60)
    exit(0)

# ========== DOWNLOAD ==========
print("Downloading from HuggingFace ...")
dataset = load_dataset(DATASET_ID, cache_dir=str(CACHE_DIR))
print(f"Available splits: {list(dataset.keys())}\n")

# Native splits — no train/val manual split
SPLITS = ["train", "validation", "test"]

for split_name in SPLITS:
    split_data = dataset[split_name]
    split_dir  = OUTPUT_DIR / split_name
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, sample in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
        img_path = images_dir / f"{split_name}_{idx:05d}.jpg"
        if not img_path.exists():
            sample["image"].convert("RGB").save(img_path, quality=95, optimize=True)
        rows.append({
            "image_path": str(img_path),          # absolute path
            "question":   sample["question"].strip(),
            "answer":     sample["answer"].strip(),
        })

    df       = pd.DataFrame(rows)
    csv_path = split_dir / f"{split_name}.csv"
    df.to_csv(csv_path, index=False)
    yn = df["answer"].str.lower().str.strip().isin(["yes", "no"]).sum()
    print(f"  {split_name:10s}: {len(df):6d} samples saved  "
          f"(yes/no: {yn:5d} | open: {len(df)-yn:5d})")

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE!")
print("=" * 60)
for split_name in SPLITS:
    csv_path = OUTPUT_DIR / split_name / f"{split_name}.csv"
    df       = pd.read_csv(csv_path)
    yn       = df["answer"].str.lower().str.strip().isin(["yes", "no"]).sum()
    print(f"  {split_name:10s}: {len(df):6d} samples  "
          f"(yes/no: {yn:5d} | open: {len(df)-yn:5d})")
print(f"\nOutput : {OUTPUT_DIR}/")
print("Next   : run 2_preprocess.py")
print("=" * 60)
