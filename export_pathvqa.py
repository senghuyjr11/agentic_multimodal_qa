# export_pathvqa.py
from pathlib import Path
from datasets import load_dataset, DownloadConfig
from PIL import Image
import csv, io, os, sys

# ====== CONFIG (edit if you want) ======
PROJECT_ROOT = Path(".").resolve()
CACHE_DIR    = PROJECT_ROOT / ".hf_cache"            # keep HF cache inside your project
OUT_DIR      = PROJECT_ROOT / "data" / "pathvqa"     # where CSVs + images will go
IMG_DIR      = OUT_DIR / "images"
SPLITS       = ("train", "validation", "test")       # which splits to export
IMAGE_FORMAT = "JPEG"                                 # "JPEG" or "PNG" etc.
JPEG_QUALITY = 95                                     # ignored for PNG
# ======================================

def ensure_dirs():
    (CACHE_DIR).mkdir(parents=True, exist_ok=True)
    (IMG_DIR).mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_pil(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert to RGB for JPEG safety (many medical images are L or LA)
    if IMAGE_FORMAT.upper() == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    if IMAGE_FORMAT.upper() == "JPEG":
        img.save(path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    else:
        img.save(path, format=IMAGE_FORMAT)

def extract_image_from_example(ex):
    """
    Works for datasets where image is:
    - a PIL.Image.Image (Image feature), or
    - a dict with 'bytes' or 'path'
    Returns a PIL.Image.Image.
    """
    im = ex["image"]
    if isinstance(im, Image.Image):
        return im
    if isinstance(im, dict):
        if "bytes" in im and im["bytes"] is not None:
            return Image.open(io.BytesIO(im["bytes"]))
        if "path" in im and im["path"]:
            return Image.open(im["path"])
    raise ValueError("Unsupported image format in example")

def export_split(ds_split, split_name: str):
    rows = []
    # We’ll name files deterministically so reruns don’t duplicate
    for i, ex in enumerate(ds_split):
        img = extract_image_from_example(ex)

        img_name = f"{split_name}_{i:07d}.jpg" if IMAGE_FORMAT.upper()=="JPEG" else f"{split_name}_{i:07d}.{IMAGE_FORMAT.lower()}"
        img_path = IMG_DIR / img_name

        if not img_path.exists():
            try:
                save_pil(img, img_path)
            except Exception as e:
                print(f"[WARN] Skipping image {i} in {split_name}: {e}", file=sys.stderr)
                continue

        rows.append({
            "split": split_name,
            "image_path": str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "question": ex.get("question", ""),
            "answer": ex.get("answer", ""),
            "question_type": ex.get("question_type", ""),
        })

        # light progress print
        if (i+1) % 1000 == 0:
            print(f"{split_name}: {i+1} examples processed...")

    if rows:
        csv_path = OUT_DIR / f"{split_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ Wrote {csv_path} ({len(rows)} rows)")
    else:
        print(f"[WARN] No rows written for {split_name}")

def main():
    ensure_dirs()

    # keep cache inside your project
    os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR)

    print("• Loading PathVQA into project-local cache…")
    ds = load_dataset(
        "flaviagiammarino/path-vqa",
        cache_dir=str(CACHE_DIR),
        download_config=DownloadConfig(cache_dir=str(CACHE_DIR)),
    )

    print(ds)  # quick sanity print

    for split in SPLITS:
        if split in ds:
            print(f"\n=== Exporting split: {split} ===")
            export_split(ds[split], split)
        else:
            print(f"[INFO] Split not present: {split}")

    print(f"\nAll done.\nImages → {IMG_DIR}\nCSVs   → {OUT_DIR}")

if __name__ == "__main__":
    main()
