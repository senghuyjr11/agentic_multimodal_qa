"""
CORRECTED PathVQA Preprocessing v2
Key fixes:
1. EOS token NOT masked - model learns to stop
2. Larger image size (384x384) for better visual features
"""
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLProcessor

print("Loading processor...")
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
processor = Qwen2VLProcessor.from_pretrained(MODEL_ID)

# Get EOS token for verification
EOS_TOKEN_ID = processor.tokenizer.eos_token_id
print(f"EOS token ID: {EOS_TOKEN_ID}")


def preprocess_sample(row, processor, target_size=(384, 384)):
    """
    CORRECTED preprocessing:
    - INPUT: question with add_generation_prompt=True
    - LABELS: answer tokens + EOS (NOT masked!)
    - Larger image size for better features
    """
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    image_path = row["image_path"]
    
    # Load and resize image - LARGER SIZE
    with Image.open(image_path) as img:
        image = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    
    # Step 1: Create INPUT (question + image + "assistant" prompt)
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
    
    # Step 2: Tokenize the INPUT
    input_ids_dict = processor(
        text=[input_text],
        images=[image],
        return_tensors="pt",
        padding=True
    )
    
    # Step 3: Tokenize the ANSWER
    answer_ids = processor.tokenizer(
        answer,
        add_special_tokens=False,
        return_tensors="pt"
    )["input_ids"][0]
    
    # Step 4: Create full sequence
    input_ids = input_ids_dict["input_ids"][0]
    eos_token = processor.tokenizer.eos_token_id
    
    # Concatenate: input_ids + answer_ids + EOS
    full_ids = torch.cat([
        input_ids,
        answer_ids,
        torch.tensor([eos_token])
    ])
    
    # Attention mask
    attention_mask_input = input_ids_dict["attention_mask"][0]
    attention_mask_answer = torch.ones(len(answer_ids) + 1, dtype=attention_mask_input.dtype)
    attention_mask = torch.cat([attention_mask_input, attention_mask_answer])
    
    # Create labels: mask input, KEEP answer + EOS
    labels = full_ids.clone()
    labels[:len(input_ids)] = -100  # Mask input only
    # *** KEY FIX: DO NOT mask EOS token! ***
    # labels[-1] stays as EOS so model learns to stop
    
    # Package tensors
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
    
    # Verify EOS is NOT masked
    if labels[-1].item() == -100:
        raise ValueError("EOS token is masked! This is wrong!")
    
    return processed


def preprocess_dataset(csv_path, output_dir, processor):
    """Preprocess entire dataset"""
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
            
            # Debug first sample
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
                
                # Critical checks
                if labels[-1].item() == -100:
                    print("\n❌ ERROR: EOS is masked!")
                elif labels[-1].item() == EOS_TOKEN_ID:
                    print("\n✅ EOS token correctly included in labels!")
                
                if "assistant" in decoded_answer.lower():
                    print("❌ ERROR: 'assistant' in labels!")
                elif decoded_answer.strip() == row['answer'].strip():
                    print("✅ Answer matches exactly!")
                
                print(f"{'='*60}\n")
                
        except Exception as e:
            print(f"\n❌ Failed on sample {idx}: {e}")
            failed_indices.append(idx)
            continue
    
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successfully processed: {len(processed_samples)}")
    print(f"✗ Failed: {len(failed_indices)}")
    
    # Save
    output_file = output_path / "preprocessed_data.pt"
    print(f"\nSaving to: {output_file}")
    torch.save(processed_samples, output_file)
    
    # Metadata
    metadata = {
        "num_samples": len(processed_samples),
        "failed_indices": failed_indices,
        "processor_name": MODEL_ID,
        "image_size": 384,
        "eos_masked": False  # Important flag
    }
    metadata_file = output_path / "metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"✓ Saved successfully")
    
    # Statistics
    seq_lengths = [len(s["input_ids"]) for s in processed_samples]
    answer_lengths = [(s["labels"] != -100).sum().item() for s in processed_samples]
    
    print(f"\nSequence length statistics:")
    print(f"  Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, Mean: {sum(seq_lengths)/len(seq_lengths):.1f}")
    
    print(f"\nAnswer token statistics:")
    print(f"  Min: {min(answer_lengths)}, Max: {max(answer_lengths)}, Mean: {sum(answer_lengths)/len(answer_lengths):.1f}")
    
    return len(processed_samples)


if __name__ == "__main__":
    print("="*60)
    print("CORRECTED PATHVQA PREPROCESSING v2")
    print("="*60)
    print("Fixes:")
    print("  1. EOS token NOT masked (model learns to stop)")
    print("  2. Image size 384x384 (better features)")
    print("="*60 + "\n")

    TRAIN_CSV = "dataset_vqa_rad/train/train.csv"
    VAL_CSV = "dataset_vqa_rad/validation/validation.csv"
    TEST_CSV = "dataset_vqa_rad/test/test.csv"

    TRAIN_OUTPUT = "preprocessed_vqa_rad/train"
    VAL_OUTPUT = "preprocessed_vqa_rad/val"
    TEST_OUTPUT = "preprocessed_vqa_rad/test"
    
    # Verify files exist
    for csv_file in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not os.path.exists(csv_file):
            print(f"❌ Error: {csv_file} not found!")
            exit(1)
    
    print("✓ All CSV files found\n")
    
    # Process all
    train_samples = preprocess_dataset(TRAIN_CSV, TRAIN_OUTPUT, processor)
    val_samples = preprocess_dataset(VAL_CSV, VAL_OUTPUT, processor)
    test_samples = preprocess_dataset(TEST_CSV, TEST_OUTPUT, processor)
    
    # Summary
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"Training samples:   {train_samples}")
    print(f"Validation samples: {val_samples}")
    print(f"Test samples:       {test_samples}")
    print(f"\nData saved to: dataset_vqa_rad/")
    print("\n⚠️  UPDATE YOUR TRAINING SCRIPT:")
    print("   Change paths to: dataset_vqa_rad/")
    print("="*60)