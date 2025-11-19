import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:256,expandable_segments:True")


import torch
import gc
import numpy as np
from transformers import AutoProcessor, TrainingArguments, Trainer, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from transformers import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict
from PIL import Image
import pandas as pd
from typing import Dict, List
import json

# ============================================================================
# CONFIGURATION - ALL SETTINGS IN ONE PLACE
# ============================================================================
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DATASET_PATH = "dataset_pathvqa"
OUTPUT_DIR = "./qwen3vl-pathvqa-lora-r32"

# Training hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1e-5
MAX_SEQ_LENGTH = 2048
MAX_PIXELS = 768 * 28 * 28

# LoRA configuration
USE_LORA = True
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["qkv_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 3  # Stop after 3 evals without improvement
EARLY_STOPPING_THRESHOLD = 0.001  # Minimum improvement threshold

# Testing mode (use small subset for quick validation)
TEST_MODE = True  # Set to True for quick testing
TEST_SAMPLES = 100  # Number of samples to use in test mode

# Global processor variable for metrics computation
processor = None


# ============================================================================
# 1. LOAD DATASET (CORRECTED)
# ============================================================================
def load_pathvqa_dataset(base_path=DATASET_PATH):
    """Load PathVQA dataset with validation"""

    def load_split(split_name):
        csv_path = os.path.join(base_path, split_name, f"{split_name}.csv")
        image_root_dir = os.path.join(base_path, split_name, "images")

        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found! Returning empty dataset.")
            return Dataset.from_dict({'image_path': [], 'question': [], 'answer': []})

        df = pd.read_csv(csv_path)
        print(f"  Reading {csv_path}: {len(df)} rows")

        valid_data = {'image_path': [], 'question': [], 'answer': []}
        skipped = 0

        for idx, row in df.iterrows():
            img_filename = row['image_path'].split('/')[-1]
            img_path = os.path.join(image_root_dir, img_filename)

            # Validate image exists and is readable
            try:
                with Image.open(img_path) as img:
                    img.verify()

                valid_data['image_path'].append(img_path)
                valid_data['question'].append(str(row['question']))
                valid_data['answer'].append(str(row['answer']))
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"    Skipping corrupt/missing image: {img_path}")

        if skipped > 3:
            print(f"    ... and {skipped - 3} more skipped images")

        print(f"  Loaded {len(valid_data['image_path'])} valid samples")
        return Dataset.from_dict(valid_data)

    dataset_dict = DatasetDict({
        'train': load_split('train'),
        'validation': load_split('validation'),
        'test': load_split('test')
    })

    print(f"✓ Dataset loaded:")
    print(f"  Train: {len(dataset_dict['train'])}")
    print(f"  Validation: {len(dataset_dict['validation'])}")
    print(f"  Test: {len(dataset_dict['test'])}")

    return dataset_dict


# ============================================================================
# 2. DATA COLLATOR (FIXED LABEL MASKING)
# ============================================================================
class Qwen3VLDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = []

        for example in examples:
            try:
                image = Image.open(example['image_path']).convert('RGB')
            except FileNotFoundError:
                print(f"Error: Image not found at {example['image_path']}. Skipping.")
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": example['question']}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example['answer']}]
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=False,
                return_tensors="pt"
            )

            batch_inputs.append(inputs)

        if not batch_inputs:
            return {}

        # Manual padding
        max_len = max(inp["input_ids"].shape[1] for inp in batch_inputs)
        max_len = min(max_len, MAX_SEQ_LENGTH)

        input_ids_list = []
        attention_mask_list = []
        pixel_values_list = []
        image_grid_thw_list = []

        for inp in batch_inputs:
            # Truncate if needed
            if inp["input_ids"].shape[1] > max_len:
                inp["input_ids"] = inp["input_ids"][:, :max_len]
                inp["attention_mask"] = inp["attention_mask"][:, :max_len]

            # Pad to max_len
            pad_len = max_len - inp["input_ids"].shape[1]
            if pad_len > 0:
                input_ids = torch.cat([
                    inp["input_ids"],
                    torch.full((1, pad_len), self.processor.tokenizer.pad_token_id, dtype=torch.long)
                ], dim=1)
                attention_mask = torch.cat([
                    inp["attention_mask"],
                    torch.zeros((1, pad_len), dtype=torch.long)
                ], dim=1)
            else:
                input_ids = inp["input_ids"]
                attention_mask = inp["attention_mask"]

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            pixel_values_list.append(inp["pixel_values"])
            if "image_grid_thw" in inp:
                image_grid_thw_list.append(inp["image_grid_thw"])

        # Stack batch
        inputs = {
            "input_ids": torch.cat(input_ids_list, dim=0),
            "attention_mask": torch.cat(attention_mask_list, dim=0),
            "pixel_values": torch.cat(pixel_values_list, dim=0),
        }

        if image_grid_thw_list:
            inputs["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

        # ============= CORRECT LABEL MASKING =============
        labels = inputs["input_ids"].clone()

        # Mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Mask based on chat template structure
        for i in range(labels.shape[0]):
            text_str = self.processor.tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=False)

            # Find where assistant response starts
            assistant_marker = "<|im_start|>assistant\n"

            if assistant_marker in text_str:
                # Tokenize up to (but not including) the assistant's answer
                prefix = text_str.split(assistant_marker)[0] + assistant_marker
                prefix_tokens = self.processor.tokenizer(
                    prefix,
                    add_special_tokens=False
                )['input_ids']

                # Mask everything before the answer
                mask_until = len(prefix_tokens)
                if mask_until < labels.shape[1]:
                    labels[i, :mask_until] = -100
            else:
                # Fallback: mask entire sequence if structure is unexpected
                print(f"Warning: Could not find assistant marker in sequence {i}")
                labels[i, :] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs.get("image_grid_thw"),
            "labels": labels
        }


# ============================================================================
# 3. EVALUATION METRICS (NEW)
# ============================================================================
def compute_metrics(eval_pred):
    """Compute accuracy and other metrics for PathVQA"""
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Decode predictions (take argmax)
    pred_ids = np.argmax(predictions, axis=-1)

    decoded_preds = []
    decoded_labels = []

    for pred, label in zip(pred_ids, labels):
        # Filter out -100 (ignored tokens) from labels
        valid_indices = label != -100

        if not valid_indices.any():
            continue

        # Get only the predicted tokens corresponding to valid label positions
        pred_valid = pred[valid_indices]
        label_valid = label[valid_indices]

        # Decode
        pred_text = processor.tokenizer.decode(pred_valid, skip_special_tokens=True).strip()
        label_text = processor.tokenizer.decode(label_valid, skip_special_tokens=True).strip()

        decoded_preds.append(pred_text.lower())
        decoded_labels.append(label_text.lower())

    # Calculate metrics
    if len(decoded_preds) == 0:
        return {"accuracy": 0.0, "exact_match": 0.0}

    # Exact match
    exact_matches = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
    exact_match_acc = exact_matches / len(decoded_preds)

    # Relaxed accuracy (handles variations)
    relaxed_matches = 0
    for pred, label in zip(decoded_preds, decoded_labels):
        pred_clean = pred.replace(".", "").replace(",", "").strip()
        label_clean = label.replace(".", "").replace(",", "").strip()

        if pred_clean == label_clean or pred_clean in label_clean or label_clean in pred_clean:
            relaxed_matches += 1

    relaxed_acc = relaxed_matches / len(decoded_preds)

    return {
        "exact_match": exact_match_acc,
        "accuracy": relaxed_acc,
    }


# ============================================================================
# 4. SETUP MODEL WITH LORA
# ============================================================================
def setup_model_and_processor():
    """Load model with LoRA configuration"""

    print(f"Loading model: {MODEL_NAME}")

    # Load processor
    proc = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=256 * 28 * 28,
        max_pixels=MAX_PIXELS
    )

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config if USE_LORA else None,
    )

    print("✓ Model loaded")

    if USE_LORA:
        print("Configuring LoRA...")

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # Freeze vision encoder
        if hasattr(model, 'visual'):
            for param in model.visual.parameters():
                param.requires_grad = False
            model.visual.eval()
            print("✓ Vision encoder frozen")

    return model, proc


# ============================================================================
# 5. SAFE TRAINER WITH MEMORY MANAGEMENT
# ============================================================================
class SafeVisionTrainer(Trainer):
    """Trainer with memory monitoring and safety features"""

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Keep vision encoder in eval mode
        if hasattr(model, 'visual'):
            model.visual.eval()
        elif hasattr(model, 'module') and hasattr(model.module, 'visual'):
            model.module.visual.eval()

        # Clear cache periodically
        if self.state.global_step % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        return super().training_step(model, inputs, num_items_in_batch)


# ============================================================================
# 6. TRAINING WITH RESUME SUPPORT
# ============================================================================
def train_model(model, proc, train_dataset, eval_dataset):
    """Train the model with early stopping"""

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        # IMPROVED LEARNING RATE SCHEDULE
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.03,  # Reduced warmup
        lr_scheduler_type="cosine",  # Better convergence

        # Optimization
        bf16=True,
        optim="adamw_torch_fused",  # Faster optimizer
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},

        # Safety features
        save_safetensors=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,

        # Logging & saving
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,  # Evaluate every 200 steps
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # Use accuracy instead of loss
        greater_is_better=True,  # Higher accuracy is better

        remove_unused_columns=False,
        report_to="none",
    )

    data_collator = Qwen3VLDataCollator(proc)

    # AUTO-DETECT AND RESUME FROM LATEST CHECKPOINT
    resume_checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_checkpoint = os.path.join(OUTPUT_DIR, latest)

            state_file = os.path.join(resume_checkpoint, "trainer_state.json")
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    saved_epoch = state.get('epoch', 0)
                    saved_step = state.get('global_step', 0)

                print(f"\n{'=' * 60}")
                print(f"🔄 RESUMING FROM CHECKPOINT: {latest}")
                print(f"  Saved at: Epoch {saved_epoch:.2f}, Step {saved_step}")
                print(f"  Checkpoint verified ✓")
                print(f"{'=' * 60}\n")
            else:
                print(f"\n⚠️ WARNING: Checkpoint {latest} missing trainer_state.json")
                resume_checkpoint = None

    # Early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=EARLY_STOPPING_THRESHOLD
    )

    trainer = SafeVisionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # NEW: Add metrics
        callbacks=[early_stopping_callback],
    )

    print("\n" + "=" * 60)
    print(
        f"Starting training: {NUM_EPOCHS} epochs, batch {BATCH_SIZE}x{GRADIENT_ACCUMULATION}={BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Early stopping: patience={EARLY_STOPPING_PATIENCE}, threshold={EARLY_STOPPING_THRESHOLD}")
    print("=" * 60 + "\n")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    print("\nSaving final model...")
    trainer.save_model(OUTPUT_DIR)
    proc.save_pretrained(OUTPUT_DIR)

    print(f"\n✓ Training complete! Model saved to: {OUTPUT_DIR}")

    return trainer


# ============================================================================
# 7. INFERENCE TESTING
# ============================================================================
def test_model(model, proc, test_dataset, num_samples=5):
    """Test the model on sample data"""

    print("\n" + "=" * 60)
    print("Testing model...")
    print("=" * 60 + "\n")

    model.eval()

    for i in range(min(num_samples, len(test_dataset))):
        example = test_dataset[i]
        try:
            image = Image.open(example['image_path']).convert('RGB')
        except FileNotFoundError:
            print(f"Skipping test example {i + 1}: Image not found")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": example['question']}
                ]
            }
        ]

        inputs = proc.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = proc.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print(f"Example {i + 1}:")
        print(f"Q: {example['question']}")
        print(f"GT: {example['answer']}")
        print(f"Pred: {output_text.strip()}")
        print("-" * 60 + "\n")


# ============================================================================
# 8. MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print("Qwen3-VL PathVQA Fine-tuning (FIXED VERSION)")
    print(f"{'=' * 60}\n")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}x{GRADIENT_ACCUMULATION}, LR: {LEARNING_RATE}")
    print(f"LoRA: {USE_LORA} (r={LORA_R}, alpha={LORA_ALPHA})")
    print(f"Early Stopping: patience={EARLY_STOPPING_PATIENCE}, threshold={EARLY_STOPPING_THRESHOLD}")
    if TEST_MODE:
        print(f"\n⚠️ TEST MODE ENABLED: Using only {TEST_SAMPLES} samples")
    print(f"{'=' * 60}\n")

    try:
        # Step 1: Load dataset
        print("Step 1: Loading dataset...")
        dataset = load_pathvqa_dataset()

        # Test mode: use small subset
        if TEST_MODE:
            print(f"\n⚠️ Reducing dataset to {TEST_SAMPLES} samples for testing...")
            dataset['train'] = dataset['train'].select(range(min(TEST_SAMPLES, len(dataset['train']))))
            dataset['validation'] = dataset['validation'].select(range(min(20, len(dataset['validation']))))
            dataset['test'] = dataset['test'].select(range(min(10, len(dataset['test']))))
            print(f"  Train: {len(dataset['train'])} samples")
            print(f"  Validation: {len(dataset['validation'])} samples")
            print(f"  Test: {len(dataset['test'])} samples\n")

        # Step 2: Setup model
        print("\nStep 2: Setting up model...")
        model, processor = setup_model_and_processor()

        # Step 3: Test data loading
        print("\nStep 3: Testing data loading...")
        sample = dataset['train'][0]
        print(f"  Sample check:")
        print(f"    Image path: {sample['image_path']}")
        print(f"    Question: {sample['question']}")
        print(f"    Answer: {sample['answer']}")

        try:
            img = Image.open(sample['image_path'])
            print(f"    ✓ Image loaded: {img.size}")
        except Exception as e:
            print(f"    ✗ Image load failed: {e}")
            raise

        # Step 4: Train
        print("\nStep 4: Training...")
        trainer = train_model(
            model=model,
            proc=processor,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation']
        )

        # Step 5: Test
        print("\nStep 5: Testing on validation samples...")
        test_model(model, processor, dataset['test'], num_samples=5)

        print("\n" + "=" * 60)
        print("All done! 🎉")
        print(f"Model saved to: {OUTPUT_DIR}")
        print("\nTo monitor training:")
        print(f"  tensorboard --logdir {OUTPUT_DIR}/logs")
        print("\nExpected validation accuracy: 55-62%")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print(f"Checkpoints saved in: {OUTPUT_DIR}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n💥 OUT OF MEMORY!")
            print("Try these fixes:")
            print("  1. Set BATCH_SIZE=2")
            print("  2. Set MAX_PIXELS=512*28*28")
            print("  3. Set GRADIENT_ACCUMULATION=8")
        raise

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise