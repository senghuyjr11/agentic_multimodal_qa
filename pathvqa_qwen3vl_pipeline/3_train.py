"""
Step 3: Fine-tune Qwen3-VL-8B-Instruct on PathVQA
=========================================================
Strategy for high accuracy:
  - Full bf16  (no quantization — A5000 handles 8B easily)
  - DoRA r=64  (higher rank → more capacity than r=16/32)
    Targets: attention (q/k/v/o) + MLP (gate/up/down)
  - Yes/No 3x oversampling to strengthen binary classification
  - Effective batch = 4 × 8 = 32  (A5000 throughput)
  - Cosine LR 2e-5, warmup 5%, weight decay 0.05
  - Early stopping patience = 10
  - Targets: Yes/No ≥ 90%  |  Overall Exact Match ≥ 65%
  - GPU: A5000

Usage:
  python 3_train.py
  python 3_train.py --test   # quick 5-min sanity check
"""
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR    = (PROJECT_ROOT.parent / ".hf_cache").resolve()
os.environ["HF_HOME"]                = str(CACHE_DIR)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

TEST_MODE = "--test" in sys.argv


def main():
    import torch
    import gc
    from peft import get_peft_model, LoraConfig
    from transformers import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from torch.utils.data import Dataset, Subset

    # ========== GPU INFO ==========
    print("=" * 60)
    print("PATH-VQA  —  STEP 3: TRAIN")
    print("=" * 60)
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
        print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"TEST MODE      : {TEST_MODE}")
    print("=" * 60 + "\n")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.cuda.empty_cache()
    gc.collect()

    # ========== CONFIG ==========
    MODEL_ID         = "Qwen/Qwen3-VL-8B-Instruct"
    PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
    CHECKPOINT_DIR   = str(PROJECT_ROOT / "checkpoints")
    ADAPTER_DIR      = str(PROJECT_ROOT / "adapters")

    # A5000: batch=4, grad_accum=8 → effective batch = 32
    BATCH_SIZE = 4 if not TEST_MODE else 2
    GRAD_ACCUM = 8

    # ========== DATASET CLASSES ==========
    class CachedDataset(Dataset):
        def __init__(self, path: Path):
            print(f"Loading : {path}")
            self.data = torch.load(path, weights_only=False)
            print(f"  {len(self.data)} samples")

        def __len__(self):  return len(self.data)
        def __getitem__(self, idx): return self.data[idx]

    class ListDataset(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self):  return len(self.data)
        def __getitem__(self, idx): return self.data[idx]

    # ========== LOAD MODEL (full bf16) ==========
    print("=" * 60)
    print(f"LOADING MODEL: {MODEL_ID}  [full bf16 — no quantization]")
    print("=" * 60)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        cache_dir=str(CACHE_DIR),
    )
    model = model.cuda()

    processor = Qwen3VLProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    processor.tokenizer.padding_side = "left"
    PAD_TOKEN_ID = processor.tokenizer.pad_token_id

    # ========== DoRA — attention + MLP ==========
    # r=64 gives more capacity than r=16/32; use_dora=True for weight decomposition
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        use_dora=True,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",        # MLP
        ],
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    # ========== COLLATOR ==========
    def fast_collator(batch):
        input_ids      = [b["input_ids"]      for b in batch]
        attention_mask = [b["attention_mask"]  for b in batch]
        labels         = [b["labels"]          for b in batch]
        pixel_values   = [b["pixel_values"]    for b in batch]
        image_grid_thw = [b["image_grid_thw"]  for b in batch]

        max_len = max(len(ids) for ids in input_ids)
        bs      = len(batch)

        input_ids_padded      = torch.full((bs, max_len), PAD_TOKEN_ID, dtype=input_ids[0].dtype)
        attention_mask_padded = torch.zeros((bs, max_len), dtype=attention_mask[0].dtype)
        labels_padded         = torch.full((bs, max_len), -100,         dtype=labels[0].dtype)

        for i, (ids, mask, labs) in enumerate(zip(input_ids, attention_mask, labels)):
            L = len(ids)
            input_ids_padded[i, -L:]      = ids
            attention_mask_padded[i, -L:] = mask
            labels_padded[i, -L:]         = labs

        # pixel_values: list of (num_patches, C) → (B, num_patches, C)
        pixel_values_stacked = torch.stack(pixel_values, dim=0)

        # image_grid_thw: each (1, 3) → (B, 3)
        if image_grid_thw[0].dim() == 2:
            image_grid_thw_stacked = torch.cat(image_grid_thw, dim=0)
        else:
            image_grid_thw_stacked = torch.stack(image_grid_thw, dim=0)

        return {
            "input_ids":      input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels":         labels_padded,
            "pixel_values":   pixel_values_stacked,
            "image_grid_thw": image_grid_thw_stacked,
        }

    # ========== LOAD DATASETS ==========
    print("\n" + "=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    train_full  = CachedDataset(PREPROCESSED_DIR / "train"      / "preprocessed_data.pt")
    val_dataset = CachedDataset(PREPROCESSED_DIR / "validation" / "preprocessed_data.pt")

    if TEST_MODE:
        train_full  = Subset(train_full,  list(range(min(300, len(train_full)))))
        val_dataset = Subset(val_dataset, list(range(min(80,  len(val_dataset)))))
        print(f"TEST MODE: train={len(train_full)}, val={len(val_dataset)}")

    # ----- Yes/No 3x oversampling -----
    # PathVQA is ~50/50 yes-no vs open. Oversampling yes/no makes the model
    # more robust on binary questions without hurting open-ended performance.
    all_train  = list(train_full if TEST_MODE else train_full.data)
    yn_samples = [s for s in all_train
                  if s.get("answer", "").strip().lower() in ("yes", "no")]
    train_dataset = ListDataset(all_train + yn_samples + yn_samples)

    print(f"\nOriginal train     : {len(all_train)}")
    print(f"Yes/No extra (×2)  : {len(yn_samples) * 2}")
    print(f"Total (oversampled): {len(train_dataset)}")
    print(f"Validation         : {len(val_dataset)}")

    # ========== TRAINING ARGS ==========
    print("\n" + "=" * 60)
    print("TRAINING CONFIG")
    print("=" * 60)

    steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)
    eval_interval   = max(100, steps_per_epoch // 4)   # ~4 evals per epoch

    args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=GRAD_ACCUM,

        # LR: 2e-5 with cosine decay, 5% warmup, light weight decay
        learning_rate=2e-5,
        num_train_epochs=3 if not TEST_MODE else 1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.05,
        max_grad_norm=1.0,

        eval_strategy="steps",
        eval_steps=eval_interval if not TEST_MODE else 20,
        save_steps=eval_interval if not TEST_MODE else 20,
        logging_steps=10,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,

        remove_unused_columns=False,
        prediction_loss_only=True,
        report_to="none",

        # A5000: full bf16, TF32, no quantization
        bf16=True,
        fp16=False,
        tf32=True,
        optim="adamw_torch",

        # gradient_checkpointing enabled via enable() above
        gradient_checkpointing=False,

        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=fast_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    print(f"Model            : {MODEL_ID}")
    print(f"DoRA             : r=64, alpha=128, dropout=0.05")
    print(f"Target modules   : q/k/v/o + gate/up/down")
    print(f"Train samples    : {len(train_dataset)}")
    print(f"Val samples      : {len(val_dataset)}")
    print(f"Eff batch size   : {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"Learning rate    : 2e-5 (cosine, warmup 5%)")
    print(f"Epochs           : {args.num_train_epochs}")
    print(f"Eval every       : {args.eval_steps} steps")
    print(f"GPU              : A5000 — full bf16")
    print("=" * 60 + "\n")

    # ========== TRAIN ==========
    try:
        torch.cuda.empty_cache()
        trainer.train()
        print("\nTraining complete!")
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint ...")
        model.save_pretrained(ADAPTER_DIR + "_interrupted")
        processor.save_pretrained(ADAPTER_DIR + "_interrupted")
        raise
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback; traceback.print_exc()
        torch.cuda.empty_cache()
        try:
            model.save_pretrained(ADAPTER_DIR + "_emergency")
            processor.save_pretrained(ADAPTER_DIR + "_emergency")
            print(f"Emergency save → {ADAPTER_DIR}_emergency")
        except Exception:
            pass
        raise

    # ========== SAVE ==========
    print(f"\nSaving final adapters to: {ADAPTER_DIR}")
    model.save_pretrained(ADAPTER_DIR)
    processor.save_pretrained(ADAPTER_DIR)
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print(f"Adapters  : {ADAPTER_DIR}")
    print("Next      : run 4_evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
