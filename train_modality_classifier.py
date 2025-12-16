"""
train_modality_classifier.py
Run once: python train_modality_classifier.py
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm import tqdm


class ModalityDataset(Dataset):
    def __init__(self, pathvqa_dir: str, vqa_rad_dir: str, processor):
        self.processor = processor
        self.samples = []

        for img in os.listdir(pathvqa_dir):
            if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                self.samples.append((os.path.join(pathvqa_dir, img), 0))

        for img in os.listdir(vqa_rad_dir):
            if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                self.samples.append((os.path.join(vqa_rad_dir, img), 1))

        print(f"Loaded {len(self.samples)} images")
        print(f"  Pathology: {sum(1 for _, l in self.samples if l == 0)}")
        print(f"  Radiology: {sum(1 for _, l in self.samples if l == 1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), torch.tensor(label)


def train():
    # === CONFIG ===
    PATHVQA_IMAGES = "dataset_pathvqa/train/images"
    VQA_RAD_IMAGES = "dataset_vqa_rad/train/images"
    OUTPUT_DIR = "modality_classifier"
    EPOCHS = 5
    BATCH_SIZE = 32
    LR = 2e-5
    # ==============

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        id2label={0: "pathology", 1: "radiology"},
        label2id={"pathology": 0, "radiology": 1},
        ignore_mismatched_sizes=True
    ).to(device)

    dataset = ModalityDataset(PATHVQA_IMAGES, VQA_RAD_IMAGES, processor)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_acc = 0
    for epoch in range(EPOCHS):
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

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for pixels, labels in val_loader:
                pixels, labels = pixels.to(device), labels.to(device)
                preds = model(pixel_values=pixels).logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val Acc={acc:.2%}")

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            print(f"  ✓ Saved checkpoint")

    print(f"\nTraining complete! Best accuracy: {best_acc:.2%}")
    print(f"Model saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    train()