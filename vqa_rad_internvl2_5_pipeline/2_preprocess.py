"""
Step 2: Preprocess VQA-RAD for InternVL2.5-8B
- Dynamic tiling up to max_num=4 tiles (448x448 each)
- Thumbnail tile always included as first tile
- Build prompt with <img><IMG_CONTEXT>*(num_tiles*256)</img> token string
- Answer-only label masking
- Stores question/answer for eval
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = (PROJECT_ROOT / ".hf_cache").resolve()
os.environ["HF_HOME"] = str(CACHE_DIR)

import math
import pickle
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

# ========== CONFIG ==========
MODEL_ID   = "OpenGVLab/InternVL2_5-8B"
IMG_SIZE   = 448
MAX_NUM    = 1    # fixed 1 tile — simpler, more stable training

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN   = "<img>"
IMG_END_TOKEN     = "</img>"

SYSTEM_PROMPT = (
    "You are a medical image analysis assistant. "
    "For yes/no questions, answer with only 'yes' or 'no'. "
    "For other questions, give a brief and accurate answer."
)

DATASET_DIR = PROJECT_ROOT.parent / "dataset"
OUTPUT_DIR  = PROJECT_ROOT / "preprocessed"

print("=" * 60)
print(f"VQA-RAD PREPROCESSING — {MODEL_ID}")
print("=" * 60)
print(f"Image size:  {IMG_SIZE}x{IMG_SIZE}  (1 tile fixed)")
print(f"Dataset dir: {DATASET_DIR}")
print(f"Output dir:  {OUTPUT_DIR}")
print("=" * 60 + "\n")

# ========== LOAD TOKENIZER + CONFIG ==========
print(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, use_fast=False, cache_dir=str(CACHE_DIR)
)

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=str(CACHE_DIR))
num_image_token = getattr(config, "num_image_token", 256)
print(f"num_image_token: {num_image_token}")

img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
eos_token_id         = tokenizer.eos_token_id
print(f"IMG_CONTEXT token id: {img_context_token_id}")
print(f"EOS token id:         {eos_token_id}\n")

# ========== IMAGE TRANSFORM ==========
image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ========== DYNAMIC TILING ==========
def find_best_aspect_ratio(aspect_ratio, target_ratios, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = image_size * image_size
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = MAX_NUM,
                        image_size: int = IMG_SIZE, use_thumbnail: bool = True):
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    # Build candidate tile grids
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = find_best_aspect_ratio(aspect_ratio, target_ratios, image_size)
    target_w = image_size * best_ratio[0]
    target_h = image_size * best_ratio[1]
    num_tiles = best_ratio[0] * best_ratio[1]

    resized = image.resize((target_w, target_h), Image.BICUBIC)

    # Slice into tiles
    processed_images = []
    for i in range(best_ratio[1]):        # rows
        for j in range(best_ratio[0]):    # cols
            box = (
                j * image_size,
                i * image_size,
                (j + 1) * image_size,
                (i + 1) * image_size,
            )
            processed_images.append(resized.crop(box))

    if use_thumbnail and num_tiles > 1:
        thumbnail = image.resize((image_size, image_size), Image.BICUBIC)
        processed_images.append(thumbnail)
        num_tiles += 1

    pixel_values = torch.stack([image_transform(img.convert("RGB")) for img in processed_images])
    return pixel_values  # (num_tiles, 3, 448, 448)


# ========== PREPROCESSING ==========
def preprocess_sample(row):
    question   = str(row["question"]).strip()
    answer     = str(row["answer"]).strip()
    image_path = row["image_path"]

    with Image.open(image_path) as img:
        pixel_values = dynamic_preprocess(img.convert("RGB"))  # (num_tiles, 3, 448, 448)

    num_tiles = pixel_values.shape[0]

    # image_token_str: each tile gets num_image_token IMG_CONTEXT tokens
    image_token_str = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (num_tiles * num_image_token) + IMG_END_TOKEN

    # Build conversation
    question_with_img = image_token_str + "\n" + question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question_with_img},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]

    answer_ids = tokenizer(
        answer, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]

    full_ids = torch.cat([prompt_ids, answer_ids, torch.tensor([eos_token_id])])
    attention_mask = torch.ones(len(full_ids), dtype=torch.long)

    labels = full_ids.clone()
    labels[:len(prompt_ids)] = -100

    if (labels != -100).sum().item() == 0:
        raise ValueError(f"No answer tokens found! Q: {question[:50]}, A: {answer}")
    if labels[-1].item() == -100:
        raise ValueError("EOS token is masked!")

    # Verify IMG_CONTEXT count
    num_img_ctx = (prompt_ids == img_context_token_id).sum().item()
    expected    = num_tiles * num_image_token
    if num_img_ctx != expected:
        raise ValueError(f"Expected {expected} IMG_CONTEXT tokens, got {num_img_ctx}")

    return {
        "input_ids":      full_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "pixel_values":   pixel_values,                          # (num_tiles, 3, 448, 448)
        "image_flags":    torch.ones(num_tiles, 1, dtype=torch.long),  # (num_tiles, 1)
        "question":       question,
        "answer":         answer,
    }


def preprocess_split(csv_path, output_dir):
    print(f"\n{'='*60}\nProcessing: {csv_path}\n{'='*60}")
    data = pd.read_csv(csv_path)
    print(f"Total samples: {len(data)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_samples = []
    failed_indices    = []

    for idx in tqdm(range(len(data)), desc="Preprocessing"):
        try:
            row       = data.iloc[idx]
            processed = preprocess_sample(row)
            processed_samples.append(processed)

            if idx == 0:
                labels     = processed["labels"]
                answer_ids = labels[labels != -100]
                decoded    = tokenizer.decode(answer_ids, skip_special_tokens=True)
                num_tiles  = processed["pixel_values"].shape[0]
                print(f"\nFirst sample check:")
                print(f"  Q: {row['question']}")
                print(f"  A: {row['answer']} | Decoded: '{decoded}'")
                print(f"  Tokens: total={len(processed['input_ids'])}, "
                      f"answer={(labels!=-100).sum()}")
                print(f"  Tiles: {num_tiles}  |  IMG_CONTEXT count: "
                      f"{(processed['input_ids'] == img_context_token_id).sum()}")
                print(f"  pixel_values: {processed['pixel_values'].shape}")

        except Exception as e:
            print(f"\nFailed on sample {idx}: {e}")
            failed_indices.append(idx)

    print(f"\nProcessed: {len(processed_samples)} | Failed: {len(failed_indices)}")

    tile_counts = [s["pixel_values"].shape[0] for s in processed_samples]
    print(f"Tile distribution: min={min(tile_counts)}, max={max(tile_counts)}, "
          f"mean={sum(tile_counts)/len(tile_counts):.2f}")

    output_file = output_path / "preprocessed_data.pt"
    torch.save(processed_samples, output_file)

    with open(output_path / "metadata.pkl", "wb") as f:
        pickle.dump({
            "num_samples":      len(processed_samples),
            "failed_indices":   failed_indices,
            "model_id":         MODEL_ID,
            "img_size":         IMG_SIZE,
            "max_num":          MAX_NUM,
            "num_image_token":  num_image_token,
            "system_prompt":    SYSTEM_PROMPT,
        }, f)

    seq_lengths = [len(s["input_ids"]) for s in processed_samples]
    ans_lengths = [(s["labels"] != -100).sum().item() for s in processed_samples]
    print(f"Seq length: min={min(seq_lengths)}, max={max(seq_lengths)}, "
          f"mean={sum(seq_lengths)/len(seq_lengths):.1f}")
    print(f"Answer len: min={min(ans_lengths)}, max={max(ans_lengths)}, "
          f"mean={sum(ans_lengths)/len(ans_lengths):.1f}")
    print(f"Saved to: {output_file}")
    return len(processed_samples)


# ========== RUN ==========
if __name__ == "__main__":
    splits = {
        DATASET_DIR / "train" / "train.csv": OUTPUT_DIR / "train",
        DATASET_DIR / "val"   / "val.csv":   OUTPUT_DIR / "val",
        DATASET_DIR / "test"  / "test.csv":  OUTPUT_DIR / "test",
    }

    for csv_file in splits:
        if not csv_file.exists():
            print(f"Error: {csv_file} not found! Run 1_download_dataset.py first.")
            exit(1)

    counts = {}
    for csv_path, out_dir in splits.items():
        split_name = csv_path.stem
        counts[split_name] = preprocess_split(csv_path, out_dir)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    for name, count in counts.items():
        print(f"  {name}: {count} samples")
    print(f"\nOutput: {OUTPUT_DIR}/")
    print("Next: run 3_train.py")
    print("=" * 60)
