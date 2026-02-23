"""
train_modality_classifier_v4.py
3-class ViT classifier: Pathology (0) | Radiology (1) | Reject (2)

Changes from v2:
- Reject dataset now includes ImageNet-Mini (5,123 total reject images)
- Uses proper validation CSVs for Pathology and Radiology
- Saves to modality_classifier_v4

Run: python train_modality_classifier_v4.py
"""

import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm import tqdm


class ModalityDataset(Dataset):
    def __init__(
        self,
        pathvqa_csv: str,
        vqa_rad_csv: str,
        reject_dir: str,
        processor,
        split: str = "",
        reject_val_ratio: float = 0.1
    ):
        self.processor = processor
        self.samples = []

        # Label 0: Pathology — unique images from CSV
        seen = set()
        with open(pathvqa_csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                path = row["image_path"]
                if path not in seen and os.path.exists(path):
                    self.samples.append((path, 0))
                    seen.add(path)
        pathology_count = len(seen)

        # Label 1: Radiology — unique images from CSV
        seen = set()
        with open(vqa_rad_csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                path = row["image_path"]
                if path not in seen and os.path.exists(path):
                    self.samples.append((path, 1))
                    seen.add(path)
        radiology_count = len(seen)

        # Label 2: Reject — split from reject folder
        all_reject = [
            os.path.join(reject_dir, f)
            for f in os.listdir(reject_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        all_reject.sort()  # deterministic split

        n_val = int(len(all_reject) * reject_val_ratio)
        if split == "VAL":
            reject_images = all_reject[:n_val]
        else:
            reject_images = all_reject[n_val:]

        for path in reject_images:
            self.samples.append((path, 2))
        reject_count = len(reject_images)

        print(f"[{split}] Unique images loaded:")
        print(f"  Pathology : {pathology_count}")
        print(f"  Radiology : {radiology_count}")
        print(f"  Reject    : {reject_count}")
        print(f"  Total     : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), torch.tensor(label)


def train():
    # === CONFIG ===
    PATHVQA_TRAIN_CSV = "dataset_pathvqa/train/train.csv"
    PATHVQA_VAL_CSV   = "dataset_pathvqa/validation/validation.csv"
    VQARAD_TRAIN_CSV  = "dataset_vqa_rad/train/train.csv"
    VQARAD_VAL_CSV    = "dataset_vqa_rad/validation/validation.csv"
    REJECT_IMAGES     = "dataset_reject/images"
    OUTPUT_DIR        = "modality_classifier_v4"
    EPOCHS            = 5
    BATCH_SIZE        = 32
    LR                = 2e-5
    # ==============

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=3,
        id2label={0: "pathology", 1: "radiology", 2: "reject"},
        label2id={"pathology": 0, "radiology": 1, "reject": 2},
        ignore_mismatched_sizes=True
    ).to(device)

    train_set = ModalityDataset(
        PATHVQA_TRAIN_CSV, VQARAD_TRAIN_CSV, REJECT_IMAGES,
        processor, split="TRAIN"
    )
    val_set = ModalityDataset(
        PATHVQA_VAL_CSV, VQARAD_VAL_CSV, REJECT_IMAGES,
        processor, split="VAL"
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_acc = 0
    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_loss = 0
        for pixels, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            pixels, labels = pixels.to(device), labels.to(device)
            outputs = model(pixel_values=pixels, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation — overall + per-class
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

        print(f"\nEpoch {epoch + 1}: Loss={avg_loss:.4f}, Val Acc={acc:.2%}")
        for cls, name in [(0, "Pathology"), (1, "Radiology"), (2, "Reject")]:
            if class_total[cls] > 0:
                cls_acc = class_correct[cls] / class_total[cls]
                print(f"  {name}: {cls_acc:.2%} ({class_correct[cls]}/{class_total[cls]})")

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            print(f"  ✓ Saved checkpoint to {OUTPUT_DIR}")

    print(f"\nTraining complete! Best accuracy: {best_acc:.2%}")
    print(f"Model saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    train()