"""
Step 2: Training with preprocessed cached data
FULLY OPTIMIZED for RTX 5070 (16GB VRAM) - Maximum GPU Utilization
Matches 1preprocess_pathvqa.py preprocessing output
"""
import gc
import os

import torch
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.cuda.empty_cache()
gc.collect()

import json
from pathlib import Path


class CachedDataset(Dataset):
    """
    Loads preprocessed tensors from individual files on disk on demand.
    Zero RAM overhead for the whole dataset.
    """

    def __init__(self, preprocessed_path):
        file_list_path = Path(preprocessed_path) / "file_list.json"

        if not file_list_path.exists():
            print(
                f"ERROR: File list not found at {file_list_path}. Please run 1preprocess_pathvqa.py first.")
            raise FileNotFoundError(f"File list not found: {file_list_path}")

        print(f"Loading file list from: {file_list_path}")
        with open(file_list_path, 'r') as f:
            self.file_paths = json.load(f)

        print(f"Loaded {len(self.file_paths)} preprocessed samples")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        return torch.load(file_path, weights_only=False)


def fast_collator(batch):
    """
    Collator for preprocessed data with padding
    For Qwen3-VL: pixel_values are 2D [num_patches, hidden_dim] and must preserve original grid_thw
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

    # Pad text sequences
    input_ids_padded = torch.full((batch_size, max_len), pad_token_id, dtype=input_ids[0].dtype)
    attention_mask_padded = torch.zeros((batch_size, max_len), dtype=attention_mask[0].dtype)
    labels_padded = torch.full((batch_size, max_len), -100, dtype=labels[0].dtype)

    for i, (ids, mask, labs) in enumerate(zip(input_ids, attention_mask, labels)):
        seq_len = len(ids)
        input_ids_padded[i, :seq_len] = ids
        attention_mask_padded[i, :seq_len] = mask
        labels_padded[i, :seq_len] = labs

    # For Qwen3-VL 2D pixel_values: concatenate all patches together
    # Model uses image_grid_thw to know where each image starts/ends
    if pixel_values[0].dim() == 2:
        # Concatenate along patch dimension: [total_patches, hidden_dim]
        pixel_values_concat = torch.cat(pixel_values, dim=0)
    else:
        # 3D format fallback (shouldn't happen with Qwen3-VL but keep for compatibility)
        max_h = max(pv.shape[1] for pv in pixel_values)
        max_w = max(pv.shape[2] for pv in pixel_values)
        num_channels = pixel_values[0].shape[0]

        pixel_values_concat = torch.zeros(
            (batch_size, num_channels, max_h, max_w),
            dtype=pixel_values[0].dtype
        )

        for i, pv in enumerate(pixel_values):
            c, h, w = pv.shape
            pixel_values_concat[i, :c, :h, :w] = pv

    result = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
        "pixel_values": pixel_values_concat,
    }

    # Stack image_grid_thw to indicate dimensions of each image in the batch
    if has_grid_thw:
        try:
            # Stack all grid_thw tensors
            image_grid_thw_stacked = torch.cat(image_grid_thw, dim=0)
            result["image_grid_thw"] = image_grid_thw_stacked
        except Exception as e:
            print(f"Warning: Could not process image_grid_thw: {e}")
            import traceback
            traceback.print_exc()
            # Fallback
            result["image_grid_thw"] = torch.tensor([[1, 28, 28]] * batch_size, dtype=torch.long)

    return result


def compute_metrics(eval_preds):
    """
    ULTRA-FAST evaluation metrics using simple token accuracy
    No external models, pure PyTorch operations
    """
    logits, labels = eval_preds
    predictions = logits.argmax(-1)

    # Limit samples for faster evaluation
    max_samples = min(50, len(predictions))
    predictions = predictions[:max_samples]
    labels = labels[:max_samples]

    # Token-level accuracy (super fast)
    mask = labels != -100
    correct = (predictions == labels) & mask
    token_accuracy = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0

    # Sequence-level accuracy (how many complete sequences match)
    seq_correct = 0
    for pred, label in zip(predictions, labels):
        label_mask = label != -100
        if label_mask.sum() > 0:
            if torch.all(pred[label_mask] == label[label_mask]):
                seq_correct += 1
    sequence_accuracy = seq_correct / len(predictions)

    return {
        "token_accuracy": token_accuracy,
        "sequence_accuracy": sequence_accuracy,
    }


class FastTrainer(Trainer):
    """Trainer with periodic GPU memory cleanup"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_counter = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)

        self.cleanup_counter += 1
        if self.cleanup_counter % 200 == 0:
            torch.cuda.empty_cache()

        return loss


if __name__ == '__main__':

    # Model setup
    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        max_memory={0: "14GiB", "cpu": "0GiB"},
        low_cpu_mem_usage=True,
    )

    model = prepare_model_for_kbit_training(model)
    processor = Qwen3VLProcessor.from_pretrained(MODEL_ID)

    lora_config = LoraConfig(
        r=32,  # Increased from 16 for more capacity
        lora_alpha=64,  # Increased proportionally
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # Load cached datasets
    print("\n" + "=" * 60)
    print("LOADING CACHED DATASETS")
    print("=" * 60)

    train_dataset = CachedDataset("preprocessed_data/train_individual")
    val_dataset = CachedDataset("preprocessed_data/val_individual")

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    print("\nVerifying sample structure:")
    sample = train_dataset[0]
    print(f"Keys in sample: {sample.keys()}")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    training_args = TrainingArguments(
        output_dir="./qwen3vl_qlora_pathvqa_cached",

        # Batch settings optimized for RTX 5070 GPU utilization
        per_device_train_batch_size=4,  # Increased from 2 for better GPU usage
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Maintains effective batch size of 32

        # Learning parameters
        learning_rate=2e-4,
        num_train_epochs=5,
        lr_scheduler_type="cosine",
        warmup_steps=300,
        max_grad_norm=1.0,

        # Logging and saving
        logging_steps=25,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=False,

        # Performance optimization
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",  # Essential for VRAM efficiency
        gradient_checkpointing=True,
        tf32=True,  # Speeds up computation on RTX GPUs

        # DataLoader settings for maximum GPU utilization
        remove_unused_columns=False,
        dataloader_num_workers=10,  # Increased from 6
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4,  # Increased from 2

        report_to="none",
        eval_accumulation_steps=2,
        group_by_length=False,  # Don't group by length for vision models
    )

    trainer = FastTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=fast_collator,
        compute_metrics=compute_metrics
    )

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION (RTX 5070 FULLY OPTIMIZED)")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"LoRA rank: 32 (increased capacity)")
    print(f"GPU Memory Limit: 14GiB")
    print(f"Precision: BF16 (better for vision)")
    print(f"Per-device batch: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"DataLoader workers: {training_args.dataloader_num_workers}")
    print(f"Prefetch factor: {training_args.dataloader_prefetch_factor}")
    print(f"Steps per epoch: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
    print(f"Metrics: Token Accuracy & Sequence Accuracy (FAST!)")
    print("=" * 60 + "\n")

    # Check for existing checkpoint
    checkpoint = None
    if os.path.exists("./qwen3vl_qlora_pathvqa_cached"):
        checkpoints = [d for d in os.listdir("./qwen3vl_qlora_pathvqa_cached") if d.startswith("checkpoint")]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            checkpoint = os.path.join("./qwen3vl_qlora_pathvqa_cached", latest_checkpoint)
            print(f"Resuming from: {checkpoint}\n")

    # Train
    torch.cuda.empty_cache()
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save final model
    model.save_pretrained("./qwen3vl_pathvqa_adapters_cached")
    print("\nTraining complete!")

    torch.cuda.empty_cache()