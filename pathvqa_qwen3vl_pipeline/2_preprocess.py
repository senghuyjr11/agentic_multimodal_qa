"""
Step 2: Preprocess PathVQA for Qwen3-VL-8B-Instruct
=========================================================
- Image size    : 448x448  (larger than 384 → better visual features)
- System prompt : instructs model to answer concisely (yes/no or brief)
- Label masking : prompt tokens masked (-100), answer + EOS kept
- EOS NOT masked — model learns to stop generating
- Saves question/answer strings for evaluation step
- Runs on CPU; no GPU needed

Output layout:
  preprocessed/
    train/       preprocessed_data.pt + metadata.pkl
    validation/  preprocessed_data.pt + metadata.pkl
    test/        preprocessed_data.pt + metadata.pkl
"""
from pathlib import Path
import os
import pickle

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR    = (PROJECT_ROOT.parent / ".hf_cache").resolve()
os.environ["HF_HOME"] = str(CACHE_DIR)

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLProcessor

# ========== CONFIG ==========
MODEL_ID   = "Qwen/Qwen3-VL-8B-Instruct"
IMAGE_SIZE = (448, 448)      # 448 → better features; consistent patch count for collator

DATASET_DIR = PROJECT_ROOT / "dataset_slake"
OUTPUT_DIR  = PROJECT_ROOT / "preprocessed"

# Medical QA system prompt — keeps yes/no answers clean and open answers brief
SYSTEM_PROMPT = (
    "You are an expert medical image analysis assistant. "
    "For yes/no questions, respond with only 'yes' or 'no'. "
    "For other questions, give a concise and accurate answer."
)

print("=" * 60)
print("PATH-VQA  —  STEP 2: PREPROCESS")
print("=" * 60)
print(f"Model      : {MODEL_ID}")
print(f"Image size : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
print(f"Dataset    : {DATASET_DIR}")
print(f"Output     : {OUTPUT_DIR}")
print("=" * 60 + "\n")

# ========== LOAD PROCESSOR ==========
print(f"Loading processor: {MODEL_ID}")
processor    = Qwen3VLProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
EOS_TOKEN_ID = processor.tokenizer.eos_token_id
print(f"EOS token id : {EOS_TOKEN_ID}  ({processor.tokenizer.eos_token})")

# Sanity-check that system prompt works in the chat template
_test = processor.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": [{"type": "image"}, {"type": "text", "text": "test"}]},
    ],
    tokenize=False, add_generation_prompt=True,
)
print(f"\nChat template sample:\n{_test}\n")


# ========== PREPROCESS ONE SAMPLE ==========
def preprocess_sample(row: pd.Series) -> dict:
    question   = str(row["question"]).strip()
    answer     = str(row["answer"]).strip()
    image_path = Path(str(row["image_path"]))

    # Load and resize to fixed size for uniform patch count
    with Image.open(image_path) as img:
        image = img.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BICUBIC)

    # Build conversation: system + user (image + question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role":    "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]

    input_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize prompt + image
    enc = processor(
        text=[input_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    # Tokenize answer (no special tokens — we append EOS manually)
    answer_ids = processor.tokenizer(
        answer, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]

    prompt_ids = enc["input_ids"][0]

    # Full sequence = prompt + answer + EOS
    full_ids = torch.cat([prompt_ids, answer_ids, torch.tensor([EOS_TOKEN_ID])])

    # Attention mask covers entire sequence
    attention_mask = torch.cat([
        enc["attention_mask"][0],
        torch.ones(len(answer_ids) + 1, dtype=torch.long),
    ])

    # Labels: mask prompt, keep answer + EOS  (EOS NOT masked)
    labels = full_ids.clone()
    labels[: len(prompt_ids)] = -100

    # image_grid_thw: ensure shape is (1, 3)
    image_grid_thw = enc["image_grid_thw"][0].cpu()
    if image_grid_thw.dim() == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)

    result = {
        "input_ids":      full_ids.cpu(),
        "attention_mask": attention_mask.cpu(),
        "labels":         labels.cpu(),
        "pixel_values":   enc["pixel_values"].squeeze(0).cpu(),
        "image_grid_thw": image_grid_thw,
        "question":       question,   # stored for evaluation
        "answer":         answer,     # stored for evaluation
    }

    # Strict checks
    n_answer_tokens = (labels != -100).sum().item()
    if n_answer_tokens == 0:
        raise ValueError(f"No answer tokens in labels! Q: {question[:60]}  A: {answer}")
    if labels[-1].item() == -100:
        raise ValueError("EOS token is masked — fix label masking!")
    if labels[-1].item() != EOS_TOKEN_ID:
        raise ValueError(f"Last label {labels[-1].item()} != EOS {EOS_TOKEN_ID}")

    return result


# ========== PREPROCESS ONE SPLIT ==========
def preprocess_split(csv_path: Path, output_dir: Path) -> int:
    print(f"\n{'='*60}")
    print(f"Split   : {csv_path}")
    print(f"Output  : {output_dir}")
    print(f"{'='*60}")

    data = pd.read_csv(csv_path)
    print(f"Samples : {len(data)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    failed    = []

    for idx in tqdm(range(len(data)), desc="Preprocessing"):
        try:
            row    = data.iloc[idx]
            sample = preprocess_sample(row)
            processed.append(sample)

            # Verbose check on first sample
            if idx == 0:
                lbl     = sample["labels"]
                ans_ids = lbl[lbl != -100]
                decoded = processor.tokenizer.decode(ans_ids, skip_special_tokens=True)
                print(f"\nFirst sample check:")
                print(f"  Question       : {row['question']}")
                print(f"  Answer (gt)    : {row['answer']}")
                print(f"  Answer (decoded): '{decoded}'")
                print(f"  Total tokens   : {len(sample['input_ids'])}")
                print(f"  Masked (prompt): {(lbl == -100).sum().item()}")
                print(f"  Trainable      : {(lbl != -100).sum().item()}")
                print(f"  EOS check      : last label={lbl[-1].item()} == EOS={EOS_TOKEN_ID} "
                      f"-> {'OK' if lbl[-1].item() == EOS_TOKEN_ID else 'FAIL'}")
                print(f"  pixel_values   : {sample['pixel_values'].shape}")
                print(f"  image_grid_thw : {sample['image_grid_thw'].shape}")

        except Exception as e:
            print(f"\n  [SKIP] sample {idx}: {e}")
            failed.append(idx)

    print(f"\nDone: processed={len(processed)} | failed={len(failed)}")

    # Save tensors
    out_file = output_dir / "preprocessed_data.pt"
    torch.save(processed, out_file)

    # Save metadata
    seq_lens = [len(s["input_ids"])         for s in processed]
    ans_lens = [(s["labels"] != -100).sum().item() for s in processed]
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump({
            "num_samples":    len(processed),
            "failed_indices": failed,
            "model_id":       MODEL_ID,
            "image_size":     IMAGE_SIZE,
            "system_prompt":  SYSTEM_PROMPT,
            "eos_masked":     False,
            "seq_len_min":    min(seq_lens),
            "seq_len_max":    max(seq_lens),
            "seq_len_mean":   sum(seq_lens) / len(seq_lens),
            "ans_len_min":    min(ans_lens),
            "ans_len_max":    max(ans_lens),
            "ans_len_mean":   sum(ans_lens) / len(ans_lens),
        }, f)

    print(f"Seq len  : min={min(seq_lens)}  max={max(seq_lens)}  "
          f"mean={sum(seq_lens)/len(seq_lens):.1f}")
    print(f"Ans len  : min={min(ans_lens)}  max={max(ans_lens)}  "
          f"mean={sum(ans_lens)/len(ans_lens):.1f}")
    print(f"Saved -> {out_file}")
    return len(processed)


# ========== RUN ALL SPLITS ==========
if __name__ == "__main__":
    # Native PathVQA splits: train / validation / test
    splits = {
        "train":      (DATASET_DIR / "train"      / "train.csv",      OUTPUT_DIR / "train"),
        "validation": (DATASET_DIR / "validation" / "validation.csv", OUTPUT_DIR / "validation"),
        "test":       (DATASET_DIR / "test"        / "test.csv",       OUTPUT_DIR / "test"),
    }

    for name, (csv_path, _) in splits.items():
        if not csv_path.exists():
            print(f"\nError: {csv_path} not found!")
            print("Run 1_download_dataset.py first.")
            exit(1)

    counts = {}
    for name, (csv_path, out_dir) in splits.items():
        counts[name] = preprocess_split(csv_path, out_dir)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    for name, n in counts.items():
        print(f"  {name:10s}: {n} samples")
    print(f"\nOutput : {OUTPUT_DIR}/")
    print("Next   : run 3_train.py")
    print("=" * 60)
