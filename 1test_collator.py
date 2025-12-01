"""
TEST COLLATOR - Verify collator works with preprocessed data
"""
import torch
import json
from pathlib import Path

print("="*60)
print("TESTING COLLATOR")
print("="*60)

# Load preprocessed samples
preprocessed_path = "preprocessed_data/train_individual"
file_list_path = Path(preprocessed_path) / "file_list.json"

if not file_list_path.exists():
    print(f"ERROR: Run 1preprocess_pathvqa.py first!")
    exit(1)

with open(file_list_path, 'r') as f:
    file_paths = json.load(f)

print(f"\n✓ Found {len(file_paths)} preprocessed samples")

# Load 3 samples to create a batch
print("\nLoading 3 samples for testing...")
batch = []
for i in range(3):
    sample = torch.load(file_paths[i], weights_only=False)
    batch.append(sample)
    print(f"Sample {i}:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}")

# Test the collator
print("\n" + "="*60)
print("TESTING COLLATOR FUNCTION")
print("="*60)

def fast_collator(batch):
    """
    Collator for preprocessed data with padding (handles variable image sizes)
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    pixel_values = [item["pixel_values"] for item in batch]

    has_grid_thw = "image_grid_thw" in batch[0]
    if has_grid_thw:
        image_grid_thw = [item["image_grid_thw"] for item in batch]

    max_len = max(len(ids) for ids in input_ids)
    batch_size = len(batch)
    pad_token_id = 151643

    input_ids_padded = torch.full((batch_size, max_len), pad_token_id, dtype=input_ids[0].dtype)
    attention_mask_padded = torch.zeros((batch_size, max_len), dtype=attention_mask[0].dtype)
    labels_padded = torch.full((batch_size, max_len), -100, dtype=labels[0].dtype)

    for i, (ids, mask, labs) in enumerate(zip(input_ids, attention_mask, labels)):
        seq_len = len(ids)
        input_ids_padded[i, :seq_len] = ids
        attention_mask_padded[i, :seq_len] = mask
        labels_padded[i, :seq_len] = labs

    # For Qwen3-VL 2D pixel_values: concatenate all patches together
    print(f"\nPixel values dimensionality: {pixel_values[0].dim()}D")

    if pixel_values[0].dim() == 2:
        # Concatenate along patch dimension: [total_patches, hidden_dim]
        print("Using 2D format - concatenating patches")
        pixel_values_concat = torch.cat(pixel_values, dim=0)
        print(f"  Concatenated shape: {pixel_values_concat.shape}")
        for i, pv in enumerate(pixel_values):
            print(f"  Sample {i}: {pv.shape} patches")
    else:
        # 3D format fallback
        print("Using 3D format (C, H, W)")
        max_h = max(pv.shape[1] for pv in pixel_values)
        max_w = max(pv.shape[2] for pv in pixel_values)
        num_channels = pixel_values[0].shape[0]

        print(f"  Channels: {num_channels}, Max H: {max_h}, Max W: {max_w}")

        pixel_values_concat = torch.zeros(
            (batch_size, num_channels, max_h, max_w),
            dtype=pixel_values[0].dtype
        )

        for i, pv in enumerate(pixel_values):
            c, h, w = pv.shape
            pixel_values_concat[i, :c, :h, :w] = pv
            print(f"  Sample {i}: {pv.shape} -> padded to {pixel_values_concat[i].shape}")

    result = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
        "pixel_values": pixel_values_concat,
    }

    if has_grid_thw:
        try:
            image_grid_thw_stacked = torch.cat(image_grid_thw, dim=0)
            result["image_grid_thw"] = image_grid_thw_stacked
            print(f"\n✓ image_grid_thw: {image_grid_thw_stacked.shape}")
        except Exception as e:
            print(f"\nWarning: Could not process image_grid_thw: {e}")
            result["image_grid_thw"] = torch.tensor([[1, 28, 28]] * batch_size, dtype=torch.long)

    return result


# Run the collator
try:
    print("\nRunning collator...")
    collated = fast_collator(batch)

    print("\n" + "="*60)
    print("SUCCESS! Collator works correctly")
    print("="*60)
    print("\nCollated batch shapes:")
    for k, v in collated.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}, dtype={v.dtype}")

except Exception as e:
    print("\n" + "="*60)
    print("ERROR IN COLLATOR")
    print("="*60)
    import traceback
    traceback.print_exc()
    exit(1)