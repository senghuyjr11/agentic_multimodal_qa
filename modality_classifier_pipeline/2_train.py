"""
Step 2: Train ViT modality classifier
3 classes: Pathology (0) | Radiology (1) | Reject (2)

- Base model: google/vit-base-patch16-224
- Radiology class now uses SLAKE instead of VQA-RAD
- Label smoothing 0.1 to prevent overconfidence (especially on reject class)
- Saves best checkpoint by validation accuracy
- Output: model/ (inside this folder)
- GPU: RTX 5070
"""
from pathlib import Path
import csv
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.resolve()

# ========== CONFIG ==========
PATHVQA_TRAIN_CSV = PROJECT_ROOT / "dataset_pathvqa" / "train"      / "train.csv"
PATHVQA_VAL_CSV   = PROJECT_ROOT / "dataset_pathvqa" / "validation" / "validation.csv"
SLAKE_TRAIN_CSV   = PROJECT_ROOT / "dataset_slake"   / "train"      / "train.csv"
SLAKE_VAL_CSV     = PROJECT_ROOT / "dataset_slake"   / "val"        / "val.csv"
REJECT_DIR        = PROJECT_ROOT / "dataset_reject"   / "images"
OUTPUT_DIR        = str(PROJECT_ROOT / "model")

EPOCHS            = 5
BATCH_SIZE        = 32
LR                = 2e-5
REJECT_VAL_RATIO  = 0.1


# ========== DATASET ==========
class ModalityDataset(Dataset):
    def __init__(self, pathvqa_csv, slake_csv, reject_dir, processor, split=""):
        self.processor = processor
        self.samples   = []

        # Label 0: Pathology (PathVQA)
        seen = set()
        with open(pathvqa_csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                path = row["image_path"]
                if path not in seen and os.path.exists(path):
                    self.samples.append((path, 0))
                    seen.add(path)
        pathology_count = len(seen)

        # Label 1: Radiology (SLAKE)
        # Remap absolute paths from original machine to local images/ folder
        slake_images_dir = Path(slake_csv).parent / "images"
        seen = set()
        with open(slake_csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                filename = Path(row["image_path"]).name
                path = str(slake_images_dir / filename)
                if path not in seen and os.path.exists(path):
                    self.samples.append((path, 1))
                    seen.add(path)
        radiology_count = len(seen)

        # Label 2: Reject (ImageNet-Mini + non-medical)
        exts = {".jpg", ".jpeg", ".png"}
        all_reject = sorted([
            str(reject_dir / f.name)
            for f in Path(reject_dir).iterdir()
            if f.suffix.lower() in exts
        ])

        n_val = int(len(all_reject) * REJECT_VAL_RATIO)
        reject_images = all_reject[:n_val] if split == "VAL" else all_reject[n_val:]
        for path in reject_images:
            self.samples.append((path, 2))

        print(f"[{split}] Pathology: {pathology_count} | Radiology: {radiology_count} | Reject: {len(reject_images)} | Total: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), torch.tensor(label)


# ========== MAIN ==========
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("MODALITY CLASSIFIER — STEP 2: TRAIN")
    print("=" * 60)
    print(f"Device     : {device}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
        print(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Epochs     : {EPOCHS}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"LR         : {LR}")
    print(f"Output     : {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=3,
        id2label={0: "pathology", 1: "radiology", 2: "reject"},
        label2id={"pathology": 0, "radiology": 1, "reject": 2},
        ignore_mismatched_sizes=True,
    ).to(device)

    print("Building datasets...")
    train_set = ModalityDataset(PATHVQA_TRAIN_CSV, SLAKE_TRAIN_CSV, REJECT_DIR, processor, split="TRAIN")
    val_set   = ModalityDataset(PATHVQA_VAL_CSV,   SLAKE_VAL_CSV,   REJECT_DIR, processor, split="VAL")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_loss = 0.0
        for pixels, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            pixels, labels = pixels.to(device), labels.to(device)
            logits = model(pixel_values=pixels).logits
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        class_correct = {0: 0, 1: 0, 2: 0}
        class_total   = {0: 0, 1: 0, 2: 0}

        with torch.no_grad():
            for pixels, labels in val_loader:
                pixels, labels = pixels.to(device), labels.to(device)
                preds = model(pixel_values=pixels).logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                for cls in [0, 1, 2]:
                    mask = labels == cls
                    class_correct[cls] += (preds[mask] == labels[mask]).sum().item()
                    class_total[cls]   += mask.sum().item()

        acc      = correct / total
        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch + 1}: loss={avg_loss:.4f}  val_acc={acc:.2%}")
        for cls, name in [(0, "Pathology"), (1, "Radiology"), (2, "Reject")]:
            if class_total[cls] > 0:
                print(f"  {name}: {class_correct[cls] / class_total[cls]:.2%}  ({class_correct[cls]}/{class_total[cls]})")

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            print(f"  Saved to {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best val accuracy : {best_acc:.2%}")
    print(f"Model saved to    : {OUTPUT_DIR}")
    print("Next: run 3_evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
