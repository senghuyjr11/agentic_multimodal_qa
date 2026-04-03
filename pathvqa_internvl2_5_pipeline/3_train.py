"""
Step 3: Fine-tune InternVL2.5-8B on PathVQA
- Full bf16 on H100
- DoRA r=32 on language-model attention layers
- Yes/No oversampled 3x
- Native PathVQA train / validation splits
- 3 epochs with cosine LR
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = (PROJECT_ROOT / ".hf_cache").resolve()
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import EarlyStoppingCallback


def main():
    TEST_MODE = False

    import gc
    import torch
    from peft import get_peft_model, LoraConfig
    from torch.utils.data import Dataset, Subset
    from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    gc.collect()

    MODEL_ID = "OpenGVLab/InternVL2_5-8B"
    PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
    OUTPUT_DIR = str(PROJECT_ROOT / "checkpoints")
    ADAPTER_DIR = str(PROJECT_ROOT / "adapters")
    BATCH_SIZE = 1
    GRAD_ACCUM = 8

    class CachedDataset(Dataset):
        def __init__(self, path):
            print(f"Loading: {path}")
            shard_files = sorted(path.parent.glob("preprocessed_data*.pt"))
            if not shard_files:
                raise FileNotFoundError(f"No preprocessed shards found in {path.parent}")
            self.data = []
            for shard_file in shard_files:
                shard = torch.load(shard_file, weights_only=False)
                self.data.extend(shard)
                print(f"  Loaded shard: {shard_file.name} ({len(shard)} samples)")
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

    print("\n" + "=" * 60)
    print("LOADING MODEL  (full bf16 — InternVL2.5-8B)")
    print("=" * 60)

    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
        use_flash_attn=True,
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=False, cache_dir=str(CACHE_DIR)
    )
    tokenizer.padding_side = "left"
    PAD_TOKEN_ID = tokenizer.pad_token_id or tokenizer.eos_token_id
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    for name, param in model.named_parameters():
        if "language_model" not in name:
            param.requires_grad = False

    attn_names = [
        name for name, _ in model.language_model.named_modules()
        if any(k in name for k in ["wqkv", "wo", "q_proj", "k_proj"])
    ]
    target_modules = ["wqkv", "wo"] if any("wqkv" in n for n in attn_names) else ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        use_dora=True,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model.language_model = get_peft_model(model.language_model, lora_config)
    model.language_model.enable_input_require_grads()
    model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.language_model.print_trainable_parameters()

    def fast_collator(batch):
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]
        pixel_values = [b["pixel_values"] for b in batch]
        image_flags = [b["image_flags"] for b in batch]

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

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_flags": torch.cat(image_flags, dim=0),
        }

    print("\n" + "=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    train_dataset = CachedDataset(PREPROCESSED_DIR / "train" / "preprocessed_data.pt")
    val_dataset = CachedDataset(PREPROCESSED_DIR / "validation" / "preprocessed_data.pt")

    if TEST_MODE:
        train_dataset = Subset(train_dataset, list(range(min(100, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(25, len(val_dataset)))))

    all_train = list(train_dataset if TEST_MODE else train_dataset.data)
    yn_samples = [s for s in all_train if s.get("answer", "").strip().lower() in ("yes", "no")]
    train_dataset_final = ListDataset(all_train + yn_samples + yn_samples)

    print(f"Original train:    {len(all_train)}")
    print(f"Yes/No extra (2x): {len(yn_samples) * 2}")
    print(f"Total oversampled: {len(train_dataset_final)}")
    print(f"Validation:        {len(val_dataset)}")

    steps_per_epoch = max(1, len(train_dataset_final) // (BATCH_SIZE * GRAD_ACCUM))
    eval_interval = max(100, steps_per_epoch // 4)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-5,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.05,
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=eval_interval,
        save_steps=eval_interval,
        logging_steps=10,
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
        gradient_checkpointing=False,
        tf32=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset_final,
        eval_dataset=val_dataset,
        data_collator=fast_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    try:
        torch.cuda.empty_cache()
        trainer.train()
        print("\nTraining complete!")
    except KeyboardInterrupt:
        print("\nInterrupted — saving current adapters...")
        model.language_model.save_pretrained(ADAPTER_DIR)
        tokenizer.save_pretrained(ADAPTER_DIR)
        raise
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("\nSaving final adapters...")
    model.language_model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Saved to: {ADAPTER_DIR}")
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("Next: run 4_evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
