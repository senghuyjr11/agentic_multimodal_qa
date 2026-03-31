"""
Step 2: Preprocess VQA-RAD for Qwen3-VL-8B-Instruct
- System prompt for medical VQA
- Answer-only label masking
- 384x384 image resize
- Stores question/answer for eval and oversampling
- Canonicalizes common short-form radiology answers to reduce exact-match fragmentation
"""
from pathlib import Path
import os
import re

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = (PROJECT_ROOT.parent / ".hf_cache").resolve()
os.environ["HF_HOME"] = str(CACHE_DIR)

import pickle
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLProcessor

# ========== CONFIG ==========
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
IMAGE_SIZE = (384, 384)
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "preprocessed"

SYSTEM_PROMPT = (
    "You are a medical image analysis assistant. "
    "For yes/no questions, answer with only 'yes' or 'no'. "
    "For other questions, give a brief and accurate answer."
)

print("=" * 60)
print(f"VQA-RAD PREPROCESSING — {MODEL_ID}")
print("=" * 60)
print(f"Image size:  {IMAGE_SIZE}")
print(f"Dataset dir: {DATASET_DIR}")
print(f"Output dir:  {OUTPUT_DIR}")
print("=" * 60 + "\n")

print(f"Loading processor: {MODEL_ID}")
processor = Qwen3VLProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
EOS_TOKEN_ID = processor.tokenizer.eos_token_id
print(f"EOS token ID: {EOS_TOKEN_ID} ({processor.tokenizer.eos_token})\n")


def canonicalize_answer(question: str, answer: str) -> str:
    """Collapse common VQA-RAD aliases into a smaller set of exact-match targets."""
    q = question.strip().lower()
    a = " ".join(str(answer).strip().lower().split())

    # exact alias table first
    alias_map = {
        "left side": "left",
        "right side": "right",
        "left lung": "left",
        "right lung": "right",
        "left hemithorax": "left",
        "right hemithorax": "right",
        "ap view": "ap",
        "pa view": "pa",
        "ap projection": "ap",
        "pa projection": "pa",
        "axial view": "axial",
        "coronal view": "coronal",
        "sagittal view": "sagittal",
        "not seen here": "not seen",
        "not visualized": "not seen",
        "not visible": "not seen",
        "none seen": "not seen",
    }
    if a in alias_map:
        return alias_map[a]

    # question-aware canonicalization
    if a in ("yes", "no"):
        return a

    if "which side" in q or "right or left" in q or "left or right" in q:
        if re.search(r"\bleft\b", a):
            return "left"
        if re.search(r"\bright\b", a):
            return "right"

    if "ap" in q or "pa" in q or "projection" in q or "view" in q:
        if re.search(r"\bap\b", a):
            return "ap"
        if re.search(r"\bpa\b", a):
            return "pa"
        if "axial" in a:
            return "axial"
        if "coronal" in a:
            return "coronal"
        if "sagittal" in a:
            return "sagittal"

    if (
        "visible" in q
        or "seen" in q
        or "present" in q
        or "identified" in q
    ) and ("not seen" in a or "not visible" in a or "not visualized" in a):
        return "not seen"

    return a


def preprocess_sample(row, processor):
    question = str(row["question"]).strip()
    answer = canonicalize_answer(question, str(row["answer"]).strip())
    image_path = row["image_path"]

    with Image.open(image_path) as img:
        image = img.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]

    input_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    input_ids_dict = processor(
        text=[input_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    answer_ids = processor.tokenizer(
        answer, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]

    input_ids = input_ids_dict["input_ids"][0]
    full_ids = torch.cat([input_ids, answer_ids, torch.tensor([EOS_TOKEN_ID])])

    attention_mask = torch.cat([
        input_ids_dict["attention_mask"][0],
        torch.ones(len(answer_ids) + 1, dtype=torch.long),
    ])

    labels = full_ids.clone()
    labels[:len(input_ids)] = -100

    image_grid_thw = input_ids_dict["image_grid_thw"][0].cpu()
    if image_grid_thw.dim() == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)

    processed = {
        "input_ids": full_ids.cpu(),
        "attention_mask": attention_mask.cpu(),
        "labels": labels.cpu(),
        "pixel_values": input_ids_dict["pixel_values"].squeeze(0).cpu(),
        "image_grid_thw": image_grid_thw,
        "question": question,
        "answer": answer,
    }

    if (labels != -100).sum().item() == 0:
        raise ValueError(f"No answer tokens! Q: {question[:50]}, A: {answer}")
    if labels[-1].item() == -100:
        raise ValueError("EOS token is masked!")

    return processed


def preprocess_split(csv_path, output_dir, processor):
    print(f"\n{'=' * 60}\nProcessing: {csv_path}\n{'=' * 60}")
    data = pd.read_csv(csv_path)
    print(f"Total samples: {len(data)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_samples = []
    failed_indices = []

    for idx in tqdm(range(len(data)), desc="Preprocessing"):
        try:
            row = data.iloc[idx]
            processed = preprocess_sample(row, processor)
            processed_samples.append(processed)

            if idx == 0:
                labels = processed["labels"]
                answer_ids = labels[labels != -100]
                decoded = processor.tokenizer.decode(answer_ids, skip_special_tokens=True)
                print(f"\nFirst sample check:")
                print(f"  Q: {row['question']}")
                print(f"  A(raw): {row['answer']}")
                print(f"  A(canon): {processed['answer']} | Decoded: '{decoded}'")
                print(
                    f"  Tokens: total={len(processed['input_ids'])}, "
                    f"answer={(labels != -100).sum()}"
                )

        except Exception as e:
            print(f"\nFailed on sample {idx}: {e}")
            failed_indices.append(idx)

    print(f"\nProcessed: {len(processed_samples)} | Failed: {len(failed_indices)}")

    output_file = output_path / "preprocessed_data.pt"
    torch.save(processed_samples, output_file)

    with open(output_path / "metadata.pkl", "wb") as f:
        pickle.dump({
            "num_samples": len(processed_samples),
            "failed_indices": failed_indices,
            "model_id": MODEL_ID,
            "image_size": IMAGE_SIZE,
            "system_prompt": SYSTEM_PROMPT,
            "answer_canonicalization": True,
        }, f)

    seq_lengths = [len(s["input_ids"]) for s in processed_samples]
    ans_lengths = [(s["labels"] != -100).sum().item() for s in processed_samples]
    print(
        f"Seq length: min={min(seq_lengths)}, max={max(seq_lengths)}, "
        f"mean={sum(seq_lengths) / len(seq_lengths):.1f}"
    )
    print(
        f"Answer len: min={min(ans_lengths)}, max={max(ans_lengths)}, "
        f"mean={sum(ans_lengths) / len(ans_lengths):.1f}"
    )
    print(f"Saved to: {output_file}")
    return len(processed_samples)


if __name__ == "__main__":
    splits = {
        DATASET_DIR / "train" / "train.csv": OUTPUT_DIR / "train",
        DATASET_DIR / "val" / "val.csv": OUTPUT_DIR / "val",
        DATASET_DIR / "test" / "test.csv": OUTPUT_DIR / "test",
    }

    for csv_file in splits:
        if not csv_file.exists():
            print(f"Error: {csv_file} not found! Run 1_download_dataset.py first.")
            exit(1)

    counts = {}
    for csv_path, out_dir in splits.items():
        split_name = csv_path.stem
        counts[split_name] = preprocess_split(csv_path, out_dir, processor)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    for name, count in counts.items():
        print(f"  {name}: {count} samples")
    print(f"\nOutput: {OUTPUT_DIR}/")
    print("Next: run 3_train.py")
    print("=" * 60)
