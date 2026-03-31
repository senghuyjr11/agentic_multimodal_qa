"""
Step 3: Fine-tune Qwen3-VL-8B-Instruct on VQA-RAD
- Full bf16 (no quantization) — intended for H100 / large-memory GPUs
- DoRA r=32, alpha=64
- Yes/no oversampled 3x
- Extra oversampling for short categorical radiology answers
- 600 steps, lr=5e-5, early stopping patience=10
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = (PROJECT_ROOT.parent / ".hf_cache").resolve()
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import EarlyStoppingCallback


def main():
    TEST_MODE = False

    import gc
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset, Subset
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import Qwen3VLProcessor as AutoProcessor
    from transformers import Trainer, TrainingArguments

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    gc.collect()

    # ========== CONFIG ==========
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
    OUTPUT_DIR = str(PROJECT_ROOT / "checkpoints")
    ADAPTER_DIR = str(PROJECT_ROOT / "adapters")

    BATCH_SIZE = 1
    GRAD_ACCUM = 8

    SHORT_CATEGORICAL_ANSWERS = {
        "left", "right", "ap", "pa", "axial", "coronal", "sagittal",
        "frontal", "lateral", "supine", "upright", "not seen",
    }

    # ========== DATASET ==========
    class CachedDataset(Dataset):
        def __init__(self, path):
            print(f"Loading: {path}")
            self.data = torch.load(path, weights_only=False)
            print(f"  Loaded {len(self.data)} samples")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    class ListDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # ========== MODEL ==========
    print("\n" + "=" * 60)
    print("LOADING MODEL  (full bf16 — Qwen3-VL-8B-Instruct)")
    print("=" * 60)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        cache_dir=str(CACHE_DIR),
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    processor.tokenizer.padding_side = "left"
    PAD_TOKEN_ID = processor.tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        use_dora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.print_trainable_parameters()

    # ========== COLLATOR ==========
    def fast_collator(batch):
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]
        pixel_values = [b["pixel_values"] for b in batch]
        image_grid_thw = [b["image_grid_thw"] for b in batch]

        max_len = max(len(ids) for ids in input_ids)
        bs = len(batch)

        input_ids_padded = torch.full((bs, max_len), PAD_TOKEN_ID, dtype=input_ids[0].dtype)
        attention_mask_padded = torch.zeros((bs, max_len), dtype=attention_mask[0].dtype)
        labels_padded = torch.full((bs, max_len), -100, dtype=labels[0].dtype)

        for i, (ids, mask, labs) in enumerate(zip(input_ids, attention_mask, labels)):
            length = len(ids)
            input_ids_padded[i, -length:] = ids
            attention_mask_padded[i, -length:] = mask
            labels_padded[i, -length:] = labs

        pixel_values_stacked = torch.stack(pixel_values, dim=0)

        if image_grid_thw[0].dim() == 2:
            image_grid_thw_stacked = torch.cat(image_grid_thw, dim=0)
        else:
            image_grid_thw_stacked = torch.stack(image_grid_thw, dim=0)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "pixel_values": pixel_values_stacked,
            "image_grid_thw": image_grid_thw_stacked,
        }

    # ========== LOAD DATASETS ==========
    print("\n" + "=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    train_dataset = CachedDataset(PREPROCESSED_DIR / "train" / "preprocessed_data.pt")
    val_dataset = CachedDataset(PREPROCESSED_DIR / "val" / "preprocessed_data.pt")

    if TEST_MODE:
        train_dataset = Subset(train_dataset, list(range(min(100, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(25, len(val_dataset)))))
        print(f"TEST MODE: train={len(train_dataset)}, val={len(val_dataset)}")

    all_train = list(train_dataset if TEST_MODE else train_dataset.data)
    yn_samples = [s for s in all_train if s.get("answer", "").strip().lower() in ("yes", "no")]
    short_cat_samples = [
        s for s in all_train
        if s.get("answer", "").strip().lower() in SHORT_CATEGORICAL_ANSWERS
    ]

    oversampled_train = ListDataset(all_train + yn_samples + yn_samples + short_cat_samples)

    print(f"\nTraining dataset:")
    print(f"  Original train:       {len(all_train)}")
    print(f"  Yes/No extra (2x):    {len(yn_samples) * 2}")
    print(f"  Short categorical +1: {len(short_cat_samples)}")
    print(f"  Total oversampled:    {len(oversampled_train)}")

    # ========== TRAINING ==========
    print("\n" + "=" * 60)
    print("TRAINING: Full bf16, DoRA r=32, lr=5e-5, 600 steps, targeted oversampling")
    print("=" * 60)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        max_steps=600,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=25,
        save_steps=25,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        remove_unused_columns=False,
        prediction_loss_only=True,
        report_to="none",
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
        tf32=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=oversampled_train,
        eval_dataset=val_dataset if not TEST_MODE else Subset(val_dataset, list(range(len(val_dataset)))),
        data_collator=fast_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    try:
        torch.cuda.empty_cache()
        trainer.train()
        print("\nTraining complete!")
    except KeyboardInterrupt:
        print("\nInterrupted — saving current adapters...")
        model.save_pretrained(ADAPTER_DIR)
        processor.save_pretrained(ADAPTER_DIR)
        raise
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        raise

    print("\nSaving final adapters...")
    model.save_pretrained(ADAPTER_DIR)
    processor.save_pretrained(ADAPTER_DIR)
    print(f"Saved to: {ADAPTER_DIR}")
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("Next: run 4_evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
