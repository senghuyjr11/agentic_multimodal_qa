"""
Preprocessing script for Qwen3-VL (compatible with training script)
Optimized for low RAM usage by saving each sample individually.
OPTIMIZED: Uses min/max pixels for consistent image dimensions
"""
from pathlib import Path
import os
import torch
import pandas as pd
from PIL import Image
from transformers import Qwen3VLProcessor
from tqdm import tqdm
import pickle
import json

print("Loading processor...")
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
processor = Qwen3VLProcessor.from_pretrained(MODEL_ID)


def preprocess_sample(row, processor):
    """
    Preprocess a single sample with consistent image sizing
    """
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    image_path = row["image_path"]

    # Load image - processor will handle sizing
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    # Create messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        },
        {
            "role": "assistant",
            "content": answer
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Process with image - use min/max pixels for consistent sizing
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        min_pixels=256*28*28,   # Minimum resolution (ensures quality)
        max_pixels=1024*28*28,  # Maximum resolution (prevents OOM)
    )

    # Remove batch dimension and move to CPU
    processed = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            processed[k] = v.squeeze(0).cpu()

    # Handle missing image_grid_thw
    if "image_grid_thw" not in processed:
        processed["image_grid_thw"] = torch.tensor([[1, 16, 16]], dtype=torch.long)

    # Validate image_grid_thw shape
    if processed["image_grid_thw"].dim() == 1:
        processed["image_grid_thw"] = processed["image_grid_thw"].unsqueeze(0)

    # Create labels
    processed["labels"] = processed["input_ids"].clone()

    # --- ROBUST MASKING LOGIC ---
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>assistant\n")
    mask_until = -1

    if assistant_token_id is not None and assistant_token_id >= 0:
        try:
            assistant_indices = (processed["labels"] == assistant_token_id).nonzero(as_tuple=True)[0]
            if assistant_indices.numel() > 0:
                mask_until = assistant_indices[0].item() + 1
        except Exception as e:
            print(f"\nToken lookup failed: {e}. Using fallback.")
            mask_until = -1

    if mask_until == -1:
        seq_len = len(processed["labels"])
        mask_until = int(seq_len * 0.8)

    processed["labels"][:mask_until] = -100

    return processed


def preprocess_dataset(csv_path, output_dir, processor):
    """
    Preprocess dataset and save each sample to a separate file
    """
    print(f"\n{'=' * 60}")
    print(f"Processing: {csv_path}")
    print('=' * 60)

    data = pd.read_csv(csv_path)
    print(f"Total samples in CSV: {len(data)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    failed_indices = []
    success_file_paths = []

    for idx in tqdm(range(len(data)), desc="Preprocessing"):
        row = data.iloc[idx]
        output_file = output_path / f"sample_{idx:08d}.pt"

        # Skip if file already exists
        if output_file.exists():
            success_file_paths.append(str(output_file))
            continue

        # Check if image exists
        if not os.path.exists(row["image_path"]):
            failed_indices.append(idx)
            continue

        try:
            processed = preprocess_sample(row, processor)
            torch.save(processed, output_file)
            success_file_paths.append(str(output_file))

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"ERROR on sample {idx}:")
            print(f"  Question: {row.get('question', 'N/A')[:50]}...")
            print(f"  Image: {row.get('image_path', 'N/A')}")
            print(f"  Error: {str(e)}")
            print(f"{'='*60}")
            failed_indices.append(idx)

    num_success = len(success_file_paths)
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Successfully processed: {num_success}")
    print(f"  Failed: {len(failed_indices)}")
    print('=' * 60)

    if num_success == 0:
        raise ValueError("No samples were successfully processed")

    # Save file list (CRITICAL for training script!)
    file_list_path = output_path / "file_list.json"
    with open(file_list_path, "w") as f:
        json.dump(success_file_paths, f)
    print(f"\nSaved list of processed files to: {file_list_path}")

    # Save metadata
    metadata = {
        "num_samples": num_success,
        "failed_indices": failed_indices,
        "processor_name": MODEL_ID,
        "first_sample_path": success_file_paths[0] if success_file_paths else None
    }
    metadata_file = output_path / "metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved metadata to: {metadata_file}")

    return num_success


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("QWEN3-VL PREPROCESSING SCRIPT (OPTIMIZED)")
    print("=" * 60)

    TRAIN_CSV = "dataset_pathvqa/train/train.csv"
    VAL_CSV = "dataset_pathvqa/validation/validation.csv"
    TRAIN_OUTPUT = "preprocessed_data/train_individual"
    VAL_OUTPUT = "preprocessed_data/val_individual"

    # Preprocess training set
    print("\n" + "=" * 60)
    print("STEP 1: PREPROCESSING TRAINING SET")
    print("=" * 60)
    train_samples = preprocess_dataset(TRAIN_CSV, TRAIN_OUTPUT, processor)

    # Preprocess validation set
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING VALIDATION SET")
    print("=" * 60)
    val_samples = preprocess_dataset(VAL_CSV, VAL_OUTPUT, processor)

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    print(f"\nPreprocessed data saved to:")
    print(f"  - {TRAIN_OUTPUT}/ (Individual .pt files)")
    print(f"  - {VAL_OUTPUT}/ (Individual .pt files)")
    print("\n" + "=" * 60)
    print("Next step: Run 2train_pathvqa.py")
    print("=" * 60)