"""
Step 3: Evaluate trained ViT modality classifier

Outputs saved to results/<timestamp>/:
  - metrics.json              overall + per-class accuracy, avg confidence
  - confusion_matrix.png      heatmap with counts and percentages
  - confidence_distribution.png  histogram per class
  - per_class_accuracy.png    bar chart
  - calibration.png           reliability diagram (confidence vs actual accuracy)

Loads model from: model/ (inside this folder)
"""
from pathlib import Path
import csv
import json
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm import tqdm
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.resolve()

# ========== CONFIG ==========
PATHVQA_TEST_CSV = PROJECT_ROOT / "dataset_pathvqa" / "test" / "test.csv"
SLAKE_TEST_CSV   = PROJECT_ROOT / "dataset_slake"   / "test" / "test.csv"
REJECT_DIR       = PROJECT_ROOT / "dataset_reject"  / "images"
MODEL_DIR        = str(PROJECT_ROOT / "model")
RESULTS_DIR      = PROJECT_ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")

REJECT_VAL_RATIO = 0.1
BATCH_SIZE       = 32
CLASS_NAMES      = {0: "Pathology", 1: "Radiology", 2: "Reject"}


# ========== DATASET ==========
class EvalDataset(Dataset):
    def __init__(self, pathvqa_csv, slake_csv, reject_dir, processor):
        self.processor = processor
        self.samples   = []

        # Pathology
        seen = set()
        with open(pathvqa_csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                path = row["image_path"]
                if path not in seen and os.path.exists(path):
                    self.samples.append((path, 0))
                    seen.add(path)
        print(f"Pathology (test) : {len(seen)}")

        # Radiology (SLAKE — remap absolute paths)
        slake_images_dir = Path(slake_csv).parent / "images"
        seen = set()
        with open(slake_csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                filename = Path(row["image_path"]).name
                path = str(slake_images_dir / filename)
                if path not in seen and os.path.exists(path):
                    self.samples.append((path, 1))
                    seen.add(path)
        print(f"Radiology (test) : {len(seen)}")

        # Reject — val portion (same held-out split as training)
        exts = {".jpg", ".jpeg", ".png"}
        all_reject = sorted([
            str(reject_dir / f.name)
            for f in Path(reject_dir).iterdir()
            if f.suffix.lower() in exts
        ])
        n_val = int(len(all_reject) * REJECT_VAL_RATIO)
        for path in all_reject[:n_val]:
            self.samples.append((path, 2))
        print(f"Reject (val)     : {n_val}")
        print(f"Total            : {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), torch.tensor(label)


# ========== PLOTS ==========
def plot_confusion_matrix(confusion, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = np.array(confusion)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = [CLASS_NAMES[i] for i in range(3)]
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13)

    for r in range(3):
        for c in range(3):
            color = "white" if cm_norm[r, c] > 0.5 else "black"
            ax.text(c, r, f"{cm[r, c]}\n({cm_norm[r, c]:.1%})",
                    ha="center", va="center", fontsize=10, color=color)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_confidence_distribution(class_confs, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    for cls in range(3):
        ax = axes[cls]
        confs = class_confs[cls]
        if confs:
            ax.hist(confs, bins=20, range=(0, 1), color=colors[cls], edgecolor="white", linewidth=0.5)
            ax.axvline(np.mean(confs), color="black", linestyle="--", linewidth=1.2,
                       label=f"mean={np.mean(confs):.2f}")
            ax.legend(fontsize=9)
        ax.set_title(CLASS_NAMES[cls], fontsize=12)
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    fig.suptitle("Confidence Distribution per Class", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_per_class_accuracy(class_correct, class_total, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [CLASS_NAMES[i] for i in range(3)]
    accs   = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(3)]
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    bars = ax.bar(labels, accs, color=colors, edgecolor="white", linewidth=0.8)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=11)

    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Per-Class Accuracy", fontsize=13)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_calibration(all_confs, all_preds, all_labels, save_path, n_bins=10):
    """Reliability diagram: how well confidence matches actual accuracy."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs  = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = [(lo <= c < hi) for c in all_confs]
        if sum(mask) == 0:
            continue
        subset_acc  = np.mean([all_preds[j] == all_labels[j] for j in range(len(mask)) if mask[j]])
        subset_conf = np.mean([all_confs[j] for j in range(len(mask)) if mask[j]])
        bin_accs.append(subset_acc)
        bin_confs.append(subset_conf)
        bin_counts.append(sum(mask))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.scatter(bin_confs, bin_accs, s=[c / 5 for c in bin_counts],
               color="#4e79a7", edgecolor="white", linewidth=0.5, zorder=3)
    ax.plot(bin_confs, bin_accs, color="#4e79a7", linewidth=1.5, label="Model")
    ax.set_xlabel("Mean Confidence", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Calibration (Reliability Diagram)", fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ========== MAIN ==========
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("MODALITY CLASSIFIER — STEP 3: EVALUATE")
    print("=" * 60)
    print(f"Model  : {MODEL_DIR}")
    print(f"Device : {device}\n")

    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    model     = ViTForImageClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    print("Loading test data...")
    dataset = EvalDataset(PATHVQA_TEST_CSV, SLAKE_TEST_CSV, REJECT_DIR, processor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_preds  = []
    all_labels = []
    all_confs  = []

    with torch.no_grad():
        for pixels, labels in tqdm(loader, desc="Evaluating"):
            pixels = pixels.to(device)
            logits = model(pixel_values=pixels).logits
            probs  = torch.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_confs.extend(confs.cpu().tolist())

    # ========== METRICS ==========
    N = len(all_labels)
    correct     = sum(p == g for p, g in zip(all_preds, all_labels))
    overall_acc = correct / N

    class_correct = {0: 0, 1: 0, 2: 0}
    class_total   = {0: 0, 1: 0, 2: 0}
    class_confs   = {0: [], 1: [], 2: []}

    for pred, label, conf in zip(all_preds, all_labels, all_confs):
        class_total[label]  += 1
        class_confs[label].append(conf)
        if pred == label:
            class_correct[label] += 1

    confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for pred, label in zip(all_preds, all_labels):
        confusion[label][pred] += 1

    # ========== PRINT ==========
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Overall accuracy : {overall_acc:.2%}  ({correct}/{N})\n")

    print("Per-class accuracy:")
    for cls in range(3):
        if class_total[cls] > 0:
            acc      = class_correct[cls] / class_total[cls]
            avg_conf = sum(class_confs[cls]) / len(class_confs[cls])
            print(f"  {CLASS_NAMES[cls]:10s}: {acc:.2%}  ({class_correct[cls]}/{class_total[cls]})  avg_conf={avg_conf:.3f}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = f"{'':12s}" + "".join(f"{CLASS_NAMES[c]:>12s}" for c in range(3))
    print(header)
    for r in range(3):
        print(f"{CLASS_NAMES[r]:12s}" + "".join(f"{confusion[r][c]:>12d}" for c in range(3)))

    # ========== SAVE ==========
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving outputs to: {RESULTS_DIR}")

    metrics = {
        "overall_accuracy": overall_acc,
        "correct": correct,
        "total": N,
        "per_class": {
            CLASS_NAMES[cls]: {
                "accuracy":    class_correct[cls] / class_total[cls] if class_total[cls] else 0,
                "correct":     class_correct[cls],
                "total":       class_total[cls],
                "avg_confidence": sum(class_confs[cls]) / len(class_confs[cls]) if class_confs[cls] else 0,
            }
            for cls in range(3)
        },
        "confusion_matrix": confusion,
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: metrics.json")

    plot_confusion_matrix(confusion,                                   RESULTS_DIR / "confusion_matrix.png")
    plot_confidence_distribution(class_confs,                          RESULTS_DIR / "confidence_distribution.png")
    plot_per_class_accuracy(class_correct, class_total,                RESULTS_DIR / "per_class_accuracy.png")
    plot_calibration(all_confs, all_preds, all_labels,                 RESULTS_DIR / "calibration.png")

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
