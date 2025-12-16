"""
VQA-RAD Preprocessing for Qwen3-VL-2B-Instruct
- EOS token NOT masked
- Image size 384x384
"""
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

print("Loading Qwen3-VL processor...")
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID)

EOS_TOKEN_ID = processor.tokenizer.eos_token_id
print(f"EOS token ID: {EOS_TOKEN_ID}")
print(f"EOS token: {processor.tokenizer.eos_token}")

# Test chat template
test_messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "test"}]}]
test_template = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
print(f"Chat template sample:\n{test_template}\n")


def preprocess_sample(row, processor, target_size=(384, 384)):
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    image_path = row["image_path"]

    with Image.open(image_path) as img:
        image = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)

    # Create input message
    user_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]

    input_text = processor.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize input
    input_ids_dict = processor(
        text=[input_text],
        images=[image],
        return_tensors="pt",
        padding=True
    )

    # Tokenize answer
    answer_ids = processor.tokenizer(
        answer,
        add_special_tokens=False,
        return_tensors="pt"
    )["input_ids"][0]

    # Build full sequence
    input_ids = input_ids_dict["input_ids"][0]
    eos_token = processor.tokenizer.eos_token_id

    full_ids = torch.cat([
        input_ids,
        answer_ids,
        torch.tensor([eos_token])
    ])

    # Attention mask
    attention_mask_input = input_ids_dict["attention_mask"][0]
    attention_mask_answer = torch.ones(len(answer_ids) + 1, dtype=attention_mask_input.dtype)
    attention_mask = torch.cat([attention_mask_input, attention_mask_answer])

    # Labels: mask input, keep answer + EOS
    labels = full_ids.clone()
    labels[:len(input_ids)] = -100

    # Handle vision tensors
    image_grid_thw_tensor = input_ids_dict["image_grid_thw"][0].cpu()
    if image_grid_thw_tensor.dim() == 1:
        image_grid_thw_tensor = image_grid_thw_tensor.unsqueeze(0)

    pixel_values_tensor = input_ids_dict["pixel_values"].squeeze(0).cpu()

    processed = {
        "input_ids": full_ids.cpu(),
        "attention_mask": attention_mask.cpu(),
        "labels": labels.cpu(),
        "pixel_values": pixel_values_tensor,
        "image_grid_thw": image_grid_thw_tensor,
    }

    # Verify
    answer_tokens = (labels != -100).sum().item()
    if answer_tokens == 0:
        raise ValueError(f"No answer tokens! Q: {question[:50]}, A: {answer}")
    if labels[-1].item() == -100:
        raise ValueError("EOS token is masked!")

    return processed


def preprocess_dataset(csv_path, output_dir, processor):
    print(f"\n{'='*60}")
    print(f"Processing: {csv_path}")
    print(f"{'='*60}")

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
                print(f"\n{'='*60}")
                print("FIRST SAMPLE VERIFICATION:")
                print(f"{'='*60}")
                print(f"Question: {row['question']}")
                print(f"Answer: {row['answer']}")

                labels = processed['labels']
                answer_ids = labels[labels != -100]
                decoded_answer = processor.tokenizer.decode(answer_ids, skip_special_tokens=True)

                print(f"\nLabel statistics:")
                print(f"  Total tokens: {len(processed['input_ids'])}")
                print(f"  Input tokens (masked): {(labels == -100).sum().item()}")
                print(f"  Answer tokens (trainable): {len(answer_ids)}")
                print(f"  Last label value: {labels[-1].item()} (should be EOS={EOS_TOKEN_ID})")
                print(f"\nDecoded answer: '{decoded_answer}'")
                print(f"Expected answer: '{row['answer']}'")

                if labels[-1].item() == EOS_TOKEN_ID:
                    print("\n✅ EOS token correctly included!")
                else:
                    print("\n❌ ERROR: EOS mismatch!")

                if decoded_answer.strip() == row['answer'].strip():
                    print("✅ Answer matches!")
                print(f"{'='*60}\n")

        except Exception as e:
            print(f"\n❌ Failed on sample {idx}: {e}")
            failed_indices.append(idx)
            continue

    print(f"\n✓ Processed: {len(processed_samples)}")
    print(f"✗ Failed: {len(failed_indices)}")

    output_file = output_path / "preprocessed_data.pt"
    torch.save(processed_samples, output_file)

    metadata = {
        "num_samples": len(processed_samples),
        "failed_indices": failed_indices,
        "processor_name": MODEL_ID,
        "image_size": 384,
        "eos_masked": False
    }
    with open(output_path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"✓ Saved to: {output_file}")

    seq_lengths = [len(s["input_ids"]) for s in processed_samples]
    answer_lengths = [(s["labels"] != -100).sum().item() for s in processed_samples]
    print(f"Seq length: min={min(seq_lengths)}, max={max(seq_lengths)}, mean={sum(seq_lengths)/len(seq_lengths):.1f}")
    print(f"Answer len: min={min(answer_lengths)}, max={max(answer_lengths)}, mean={sum(answer_lengths)/len(answer_lengths):.1f}")

    return len(processed_samples)


if __name__ == "__main__":
    print("="*60)
    print("VQA-RAD PREPROCESSING FOR Qwen3-VL-2B-Instruct")
    print("="*60 + "\n")

    TRAIN_CSV = "dataset_vqa_rad/train/train.csv"
    VAL_CSV = "dataset_vqa_rad/validation/validation.csv"
    TEST_CSV = "dataset_vqa_rad/test/test.csv"

    TRAIN_OUTPUT = "preprocessed_vqa_rad_qwen3/train"
    VAL_OUTPUT = "preprocessed_vqa_rad_qwen3/val"
    TEST_OUTPUT = "preprocessed_vqa_rad_qwen3/test"

    for csv_file in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not os.path.exists(csv_file):
            print(f"❌ Error: {csv_file} not found!")
            exit(1)

    print("✓ All CSV files found\n")

    train_samples = preprocess_dataset(TRAIN_CSV, TRAIN_OUTPUT, processor)
    val_samples = preprocess_dataset(VAL_CSV, VAL_OUTPUT, processor)
    test_samples = preprocess_dataset(TEST_CSV, TEST_OUTPUT, processor)

    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"Train: {train_samples}")
    print(f"Val:   {val_samples}")
    print(f"Test:  {test_samples}")
    print(f"\nOutput: preprocessed_vqa_rad_qwen3/")
    print("="*60)