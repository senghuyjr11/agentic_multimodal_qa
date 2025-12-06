"""
Complete Training Script for PathVQA with Qwen2-VL
Optimized settings to prevent overfitting

DEBUG MODE: Set TEST_MODE=True to run quick sanity check on small data
"""
from pathlib import Path
import os
PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = (PROJECT_ROOT / ".hf_cache").resolve()
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR))

# ========== TEST MODE CONFIGURATION ==========
# Set this to True for quick sanity check (5-10 minutes)
# Set this to False for full training (12+ hours)
TEST_MODE = False

if TEST_MODE:
    print("="*70)
    print("🧪 TEST MODE ENABLED")
    print("="*70)
    print("Running quick sanity check on small dataset")
    print("This will take ~5-10 minutes")
    print("If everything works, set TEST_MODE=False and rerun")
    print("="*70 + "\n")
else:
    print("="*70)
    print("🚀 FULL TRAINING MODE")
    print("="*70)
    print("Running complete training on full dataset")
    print("This will take ~12 hours")
    print("="*70 + "\n")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Change to your GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import (
    Qwen2VLForConditionalGeneration, 
    Qwen2VLProcessor, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer
)
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import gc

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.cuda.empty_cache()
gc.collect()


# ========== DATASET ==========
class CachedDataset(Dataset):
    """Loads preprocessed data from disk"""
    def __init__(self, preprocessed_path):
        print(f"Loading preprocessed data from: {preprocessed_path}")
        self.data = torch.load(preprocessed_path, weights_only=False)
        print(f"✓ Loaded {len(self.data)} preprocessed samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ========== MODEL SETUP ==========
print("\n" + "="*60)
print("LOADING MODEL")
print("="*60)

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "18GiB", "cpu": "0GiB"},
    low_cpu_mem_usage=True,
)

model = prepare_model_for_kbit_training(model)
processor = Qwen2VLProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "left"
PAD_TOKEN_ID = processor.tokenizer.pad_token_id

def fast_collator(batch):
    """
    Collator for preprocessed PathVQA samples:
    - input_ids: [seq_len]
    - attention_mask: [seq_len]
    - labels: [seq_len]
    - pixel_values: [num_patches, hidden_dim]
    - image_grid_thw: [1, 3]
    """
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    pixel_values = [b["pixel_values"] for b in batch]
    image_grid_thw = [b["image_grid_thw"] for b in batch]

    # Text padding (LEFT)
    max_len = max(len(ids) for ids in input_ids)
    bs = len(batch)

    input_ids_padded = torch.full(
        (bs, max_len), PAD_TOKEN_ID, dtype=input_ids[0].dtype
    )
    attention_mask_padded = torch.zeros(
        (bs, max_len), dtype=attention_mask[0].dtype
    )
    labels_padded = torch.full(
        (bs, max_len), -100, dtype=labels[0].dtype
    )

    for i, (ids, mask, labs) in enumerate(zip(input_ids, attention_mask, labels)):
        L = len(ids)
        input_ids_padded[i, -L:] = ids
        attention_mask_padded[i, -L:] = mask
        labels_padded[i, -L:] = labs

    # Vision tensors
    # pixel_values: list of [num_patches, hidden_dim] -> [B, num_patches, hidden_dim]
    pixel_values_stacked = torch.stack(pixel_values, dim=0)

    # image_grid_thw: each [1, 3] -> [B, 3] (Qwen2-VL is okay with [B, 3])
    if image_grid_thw[0].dim() == 2:
        image_grid_thw_stacked = torch.cat(image_grid_thw, dim=0)  # [B, 3]
    else:
        image_grid_thw_stacked = torch.stack(image_grid_thw, dim=0)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
        "pixel_values": pixel_values_stacked,
        "image_grid_thw": image_grid_thw_stacked,
    }

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,  # Higher dropout to prevent overfitting
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.print_trainable_parameters()


# ========== LOAD DATASETS ==========
print("\n" + "="*60)
print("LOADING DATASETS")
print("="*60)

train_dataset = CachedDataset("preprocessed_data_v2/train/preprocessed_data.pt")
val_dataset = CachedDataset("preprocessed_data_v2/val/preprocessed_data.pt")

# ========== TEST MODE: Use tiny subset ==========
if TEST_MODE:
    print("\n🧪 TEST MODE: Using small subset of data")
    from torch.utils.data import Subset
    
    # Use 500 training samples (was 100 - too small!)
    train_indices = list(range(min(500, len(train_dataset))))
    train_dataset = Subset(train_dataset, train_indices)
    
    # Use 100 validation samples (was 50)
    val_indices = list(range(min(100, len(val_dataset))))
    val_dataset = Subset(val_dataset, val_indices)
    
    print(f"  Train: {len(train_dataset)} samples (subset)")
    print(f"  Val: {len(val_dataset)} samples (subset)")
else:
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

eval_dataset_final = val_dataset


# ========== METRICS ==========
def compute_metrics(eval_preds):
    """
    Fast metrics for training monitoring
    Only computes token accuracy (no text generation during training)
    Real generation evaluation happens after training completes
    """
    logits, labels = eval_preds
    predictions = logits.argmax(-1)

    # Simple token-level accuracy
    valid_mask = labels != -100
    correct = (predictions == labels) & valid_mask
    accuracy = correct.sum() / valid_mask.sum() if valid_mask.sum() > 0 else 0

    return {
        "token_accuracy": float(accuracy),
    }
    
    # Note: For real evaluation metrics (exact match, F1), 
    # run the separate evaluation script after training completes


# ========== TRAINING ARGUMENTS ==========
training_args = TrainingArguments(
    output_dir="./qwen2vl_pathvqa_test" if TEST_MODE else "./qwen2vl_pathvqa_final",
    
    # Batch settings
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    
    # Learning rate and epochs
    learning_rate=5e-5,
    num_train_epochs=2 if TEST_MODE else 3,  # 2 epochs for test, 3 for real
    lr_scheduler_type="cosine",
    warmup_steps=20 if TEST_MODE else 300,  # Quick warmup for test
    max_grad_norm=1.0,
    weight_decay=0.01,
    
    # Logging/saving - more frequent in test mode
    logging_steps=5 if TEST_MODE else 50,
    save_steps=25 if TEST_MODE else 300,
    eval_steps=25 if TEST_MODE else 300,
    eval_strategy="steps",
    
    # Monitor loss only during training
    prediction_loss_only=True,
    
    save_total_limit=2 if TEST_MODE else 3,
    
    # Early stopping
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Performance
    fp16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    
    # Data loading
    remove_unused_columns=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=4,
    dataloader_persistent_workers=True,
    
    ddp_find_unused_parameters=False,
    report_to="none",
    tf32=True,
    eval_accumulation_steps=4,
)


# ========== CUSTOM TRAINER ==========
class MemoryEfficientTrainer(Trainer):
    """Trainer with memory management and overfitting detection"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_counter = 0
        self.best_eval_loss = float('inf')
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)
        
        # Periodic memory cleanup
        self.cleanup_counter += 1
        if self.cleanup_counter % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        return loss
    
    def log(self, logs):
        """Enhanced logging with overfitting detection"""
        super().log(logs)
        
        # Check for overfitting
        if "eval_loss" in logs and "loss" in logs:
            gap = logs["eval_loss"] - logs["loss"]
            logs["overfitting_gap"] = gap
            
            if gap > 0.5:
                print(f"\n⚠️  Overfitting detected! Gap: {gap:.3f}")
                print(f"   Train loss: {logs['loss']:.4f}")
                print(f"   Eval loss: {logs['eval_loss']:.4f}")
            
            # Track best eval loss
            if logs["eval_loss"] < self.best_eval_loss:
                self.best_eval_loss = logs["eval_loss"]
                print(f"   🎯 New best eval loss: {self.best_eval_loss:.4f}")
    
    def evaluation_loop(self, *args, **kwargs):
        """Clear cache before/after evaluation"""
        torch.cuda.empty_cache()
        result = super().evaluation_loop(*args, **kwargs)
        torch.cuda.empty_cache()
        return result


# ========== VALIDATION DATASET SELECTION ==========
# For proper training: use full validation set, monitor loss only
# Real metrics (exact match, F1) should be computed AFTER training

# Use full validation set for accurate loss monitoring
eval_dataset_final = val_dataset

# If full set causes OOM during eval, use 2000-3000 samples:
# MAX_EVAL_SAMPLES = 2000
# eval_indices = list(range(min(MAX_EVAL_SAMPLES, len(val_dataset))))
# eval_dataset_final = Subset(val_dataset, eval_indices)

print(f"Using validation samples: {len(eval_dataset_final)}")


# ========== CREATE TRAINER ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset_final,
    data_collator=fast_collator,
)



# ========== PRINT CONFIGURATION ==========
print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)
print(f"Model: {MODEL_ID}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset_final)}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Learning rate: {training_args.learning_rate}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Steps per epoch: ~{len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
print(f"Evaluation every: {training_args.eval_steps} steps")
print(f"LoRA dropout: {lora_config.lora_dropout}")
print(f"Weight decay: {training_args.weight_decay}")
print("="*60 + "\n")


# ========== CHECK FOR RESUME ==========
checkpoint = None
if os.path.exists(training_args.output_dir):
    checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        checkpoint = os.path.join(training_args.output_dir, latest_checkpoint)
        print(f"📂 Found checkpoint: {checkpoint}")
        print(f"   Resume training? (yes/no): ", end="")
        response = input().strip().lower()
        if response != "yes":
            checkpoint = None
            print("   Starting fresh training...\n")
        else:
            print(f"   Resuming from: {checkpoint}\n")


# ========== TRAINING ==========
print("="*60)
print("🚀 STARTING TRAINING")
print("="*60 + "\n")

try:
    torch.cuda.empty_cache()
    trainer.train(resume_from_checkpoint=checkpoint)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    trainer.save_model("./qwen2vl_pathvqa_interrupted")
    print("✓ Saved interrupted checkpoint to ./qwen2vl_pathvqa_interrupted")
    
except Exception as e:
    print(f"\n❌ Training error: {e}")
    import traceback
    traceback.print_exc()
    torch.cuda.empty_cache()
    try:
        trainer.save_model("./qwen2vl_pathvqa_emergency")
        print("✓ Emergency save to ./qwen2vl_pathvqa_emergency")
    except:
        print("✗ Emergency save failed")
    raise


# ========== SAVE FINAL MODEL ==========
print("\nSaving final model...")
if TEST_MODE:
    model.save_pretrained("./qwen2vl_pathvqa_test_adapters")
    processor.save_pretrained("./qwen2vl_pathvqa_test_adapters")
    print("✓ Test model saved to ./qwen2vl_pathvqa_test_adapters")
else:
    model.save_pretrained("./qwen2vl_pathvqa_adapters")
    processor.save_pretrained("./qwen2vl_pathvqa_adapters")
    print("✓ Model saved to ./qwen2vl_pathvqa_adapters")

torch.cuda.empty_cache()

# ========== TEST MODE: Quick Evaluation ==========
if TEST_MODE:
    print("\n" + "="*60)
    print("🧪 TEST MODE: Running quick evaluation on 50 test samples")
    print("="*60 + "\n")
    
    # Load test data
    test_dataset = CachedDataset("preprocessed_data_v2/test/preprocessed_data.pt")
    test_subset = Subset(test_dataset, list(range(min(50, len(test_dataset)))))
    
    print(f"Evaluating on {len(test_subset)} test samples...")
    
    predictions = []
    ground_truths = []
    
    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(test_subset)), desc="Generating"):
            try:
                sample = test_subset[idx]

                full_ids = sample["input_ids"]
                full_attn = sample["attention_mask"]
                labels = sample["labels"]

                # Find where answer starts (first non -100 in labels)
                answer_mask = labels != -100
                answer_positions = answer_mask.nonzero(as_tuple=False)
                if len(answer_positions) == 0:
                    answer_start = full_ids.shape[0]  # fallback
                else:
                    answer_start = answer_positions[0].item()

                # Use ONLY prompt (image + question + assistant) as input
                prompt_input_ids = full_ids[:answer_start]
                prompt_attention_mask = full_attn[:answer_start]

                input_ids = prompt_input_ids.unsqueeze(0).to(model.device)
                attention_mask = prompt_attention_mask.unsqueeze(0).to(model.device)

                pixel_values = sample["pixel_values"].unsqueeze(0).to(model.device)

                image_grid_thw = sample["image_grid_thw"].to(model.device)  # [1, 3]
                if image_grid_thw.dim() == 1:
                    image_grid_thw = image_grid_thw.unsqueeze(0)  # [1, 3]

                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

                # Only the newly generated tokens
                gen_only = generated_ids[0, input_ids.shape[1]:]
                pred_text = processor.tokenizer.decode(gen_only, skip_special_tokens=True).strip()

                labels_clean = labels[labels != -100]
                gt_text = processor.tokenizer.decode(labels_clean, skip_special_tokens=True).strip()

                predictions.append(pred_text)
                ground_truths.append(gt_text)

                if idx < 5:
                    print(f"\nSample {idx+1}:")
                    print(f"  Pred: {pred_text}")
                    print(f"  True: {gt_text}")
                    print(f"  Match: {'✓' if pred_text.lower().strip() == gt_text.lower().strip() else '✗'}")

            except Exception as e:
                print(f"Error on sample {idx}: {e}")
                predictions.append("")
                ground_truths.append("")

    
    # Calculate simple metrics
    exact_matches = sum(
        p.lower().strip() == g.lower().strip()
        for p, g in zip(predictions, ground_truths)
    )
    exact_match_rate = exact_matches / len(predictions)
    
    print("\n" + "="*60)
    print("TEST MODE EVALUATION RESULTS")
    print("="*60)
    print(f"Test samples: {len(predictions)}")
    print(f"Exact Match: {exact_match_rate:.4f} ({exact_match_rate*100:.1f}%)")
    print("="*60)
    
    print("\n✅ TEST MODE VERIFICATION:")
    if exact_match_rate > 0.30:
        print("🎉 GREAT! Model is learning (>30% on small test)")
        print("✅ Everything looks good!")
        print("✅ Set TEST_MODE=False and run full training!")
    elif exact_match_rate > 0.15:
        print("⚠️  OKAY: Model is learning but could be better (15-30%)")
        print("   Check:")
        print("   - Are predictions clean? (no 'assistant' tokens)")
        print("   - Are they reasonable answers?")
        print("   If yes, proceed with full training")
    elif exact_match_rate > 0.05:
        print("⚠️  WARNING: Low performance (5-15%)")
        print("   This might be okay for 1 epoch on tiny data")
        print("   But verify:")
        print("   - Predictions aren't empty")
        print("   - No system tokens in output")
    else:
        print("❌ PROBLEM: Very low performance (<5%)")
        print("   Something is wrong!")
        print("   - Check preprocessing verification")
        print("   - Look at sample predictions above")
        print("   - Fix issues before full training")
    
    print("\n" + "="*60)

print("\n" + "="*60)
print("🎉 ALL DONE!")
print("="*60)
if TEST_MODE:
    print("Next: If test passed, set TEST_MODE=False and rerun")
else:
    print("Next: Run evaluation script on full test set")
print("="*60)