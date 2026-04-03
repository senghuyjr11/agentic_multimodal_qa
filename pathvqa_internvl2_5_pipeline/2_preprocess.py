"""
Step 2: Preprocess PathVQA for InternVL2.5-8B
- Fixed 1-tile preprocessing at 448x448
- Prompt uses <img><IMG_CONTEXT>*N</img> format
- Answer-only label masking
- Stores question/answer for evaluation
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = (PROJECT_ROOT / ".hf_cache").resolve()
os.environ["HF_HOME"] = str(CACHE_DIR)

import pickle
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
IMG_SIZE = 448
MAX_NUM = 1
SHARD_SIZE = 2000
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "preprocessed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
SYSTEM_PROMPT = (
    "You are an expert medical image analysis assistant. "
    "For yes/no questions, respond with only 'yes' or 'no'. "
    "For other questions, give a concise and accurate answer."
)

print("=" * 60)
print(f"PATH-VQA PREPROCESSING — {MODEL_ID}")
print("=" * 60)
print(f"Image size:  {IMG_SIZE}x{IMG_SIZE} (1 tile fixed)")
print(f"Dataset dir: {DATASET_DIR}")
print(f"Output dir:  {OUTPUT_DIR}")
print(f"Tensor device: {DEVICE}")
print("=" * 60 + "\n")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, use_fast=False, cache_dir=str(CACHE_DIR)
)
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=str(CACHE_DIR))
num_image_token = getattr(config, "num_image_token", 256)
img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
eos_token_id = tokenizer.eos_token_id

MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=DEVICE).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=DEVICE).view(3, 1, 1)


def dynamic_preprocess(image: Image.Image):
    image = image.convert("RGB")
    tensor = TF.pil_to_tensor(image).to(device=DEVICE, dtype=torch.float32) / 255.0
    tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0),
        size=(IMG_SIZE, IMG_SIZE),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0)


def preprocess_sample(row):
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    image_path = row["image_path"]

    with Image.open(image_path) as img:
        pixel_values = dynamic_preprocess(img).to(torch.float16).cpu()

    num_tiles = pixel_values.shape[0]
    image_token_str = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (num_tiles * num_image_token) + IMG_END_TOKEN
    question_with_img = image_token_str + "\n" + question

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_with_img},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    full_ids = torch.cat([prompt_ids, answer_ids, torch.tensor([eos_token_id])])
    attention_mask = torch.ones(len(full_ids), dtype=torch.long)
    labels = full_ids.clone()
    labels[:len(prompt_ids)] = -100

    if (labels != -100).sum().item() == 0:
        raise ValueError(f"No answer tokens found! Q: {question[:50]} A: {answer}")
    if labels[-1].item() == -100:
        raise ValueError("EOS token is masked!")

    num_img_ctx = (prompt_ids == img_context_token_id).sum().item()
    expected = num_tiles * num_image_token
    if num_img_ctx != expected:
        raise ValueError(f"Expected {expected} IMG_CONTEXT tokens, got {num_img_ctx}")

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_flags": torch.ones(num_tiles, 1, dtype=torch.long),
        "question": question,
        "answer": answer,
    }


def preprocess_split(csv_path, output_dir):
    print(f"\n{'='*60}\nProcessing: {csv_path}\n{'='*60}")
    data = pd.read_csv(csv_path)
    print(f"Total samples: {len(data)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_samples = []
    failed_indices = []

    for idx in tqdm(range(len(data)), desc="Preprocessing"):
        try:
            row = data.iloc[idx]
            processed = preprocess_sample(row)
            processed_samples.append(processed)

            if idx == 0:
                labels = processed["labels"]
                answer_ids = labels[labels != -100]
                decoded = tokenizer.decode(answer_ids, skip_special_tokens=True)
                print("\nFirst sample check:")
                print(f"  Q: {row['question']}")
                print(f"  A: {row['answer']} | Decoded: '{decoded}'")
                print(f"  Tokens: total={len(processed['input_ids'])}, answer={(labels != -100).sum()}")
                print(f"  Tiles: {processed['pixel_values'].shape[0]}")
        except Exception as e:
            print(f"\nFailed on sample {idx}: {e}")
            failed_indices.append(idx)

    shard_files = []
    for shard_idx, start in enumerate(range(0, len(processed_samples), SHARD_SIZE)):
        shard = processed_samples[start:start + SHARD_SIZE]
        shard_file = output_path / f"preprocessed_data_{shard_idx:03d}.pt"
        torch.save(shard, shard_file)
        shard_files.append(shard_file.name)

    with open(output_path / "metadata.pkl", "wb") as f:
        pickle.dump({
            "num_samples": len(processed_samples),
            "failed_indices": failed_indices,
            "model_id": MODEL_ID,
            "img_size": IMG_SIZE,
            "max_num": MAX_NUM,
            "num_image_token": num_image_token,
            "system_prompt": SYSTEM_PROMPT,
            "shard_size": SHARD_SIZE,
            "shard_files": shard_files,
        }, f)

    print(f"\nProcessed: {len(processed_samples)} | Failed: {len(failed_indices)}")
    print(f"Saved shards: {len(shard_files)} -> {output_path}")
    return len(processed_samples)


if __name__ == "__main__":
    splits = {
        DATASET_DIR / "train" / "train.csv": OUTPUT_DIR / "train",
        DATASET_DIR / "validation" / "validation.csv": OUTPUT_DIR / "validation",
        DATASET_DIR / "test" / "test.csv": OUTPUT_DIR / "test",
    }

    for csv_file in splits:
        if not csv_file.exists():
            print(f"Error: {csv_file} not found! Run 1_download_dataset.py first.")
            raise SystemExit(1)

    counts = {}
    for csv_path, out_dir in splits.items():
        counts[csv_path.stem] = preprocess_split(csv_path, out_dir)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    for name, count in counts.items():
        print(f"  {name}: {count} samples")
    print(f"\nOutput: {OUTPUT_DIR}/")
    print("Next: run 3_train.py")
    print("=" * 60)
