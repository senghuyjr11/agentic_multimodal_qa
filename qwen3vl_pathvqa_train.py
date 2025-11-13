import glob
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:256,expandable_segments:True")

import torch
from transformers import AutoProcessor, TrainingArguments, Trainer, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict
from PIL import Image
import pandas as pd
from typing import Dict, List

# ============================================================================
# CONFIGURATION - ALL SETTINGS IN ONE PLACE
# ============================================================================
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DATASET_PATH = "dataset_pathvqa"
OUTPUT_DIR = "./qwen3vl-pathvqa-lora"

MAX_SAMPLES_PER_SPLIT = None

# Training hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 6
GRADIENT_ACCUMULATION = 3
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 2048
MAX_PIXELS = 768 * 28 * 28

# LoRA configuration
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["qkv_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Add at the beginning of your training function:
def get_last_checkpoint(output_dir):
    """Find the most recent checkpoint"""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if checkpoints:
        # Sort by step number
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        last_checkpoint = checkpoints[-1]
        print(f"📁 Found checkpoint: {last_checkpoint}")
        return last_checkpoint
    return None

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
def load_pathvqa_dataset(base_path=DATASET_PATH):
    """Load PathVQA dataset with validation and optional sample limit"""

    def load_split(split_name):
        csv_path = os.path.join(base_path, split_name, f"{split_name}.csv")

        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found!")
            return Dataset.from_dict({'image_path': [], 'question': [], 'answer': []})

        df = pd.read_csv(csv_path)
        print(f"  Reading {csv_path}: {len(df)} rows")

        valid_data = {'image_path': [], 'question': [], 'answer': []}
        skipped = 0

        for idx, row in df.iterrows():
            img_path = row['image_path']

            # Validate image exists and is readable
            try:
                img = Image.open(img_path)
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
# 2. DATA COLLATOR WITH PROPER LABEL MASKING
# ============================================================================
class Qwen3VLDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        # Process each example individually to avoid token mismatch
        batch_inputs = []

        for example in examples:
            try:
                image = Image.open(example['image_path']).convert('RGB')
            except Exception as e:
                print(f"Warning: Failed to load image {example['image_path']} in collator. Skipping. Error: {e}")
                continue  # Skip this example if image fails to load

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
                    "content": [
                        {"type": "text", "text": example['answer']}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Process single example
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=False,
                return_tensors="pt"
            )

            batch_inputs.append(inputs)

        # If all examples in the batch failed, return an empty dict
        if not batch_inputs:
            return {}

        # Manual padding to max length in batch
        max_len = max(inp["input_ids"].shape[1] for inp in batch_inputs)
        max_len = min(max_len, MAX_SEQ_LENGTH)  # Cap at max length

        # Pad all inputs
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

        # Handle empty batch again after processing
        if not input_ids_list:
            return {}

        # Stack into batch
        inputs = {
            "input_ids": torch.cat(input_ids_list, dim=0),
            "attention_mask": torch.cat(attention_mask_list, dim=0),
            "pixel_values": torch.cat(pixel_values_list, dim=0),
        }

        if image_grid_thw_list:
            inputs["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

        # Create labels with proper masking
        labels = inputs["input_ids"].clone()

        # Mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Simple masking: only train on assistant response
        # The assistant token is usually after the image tokens
        for i in range(labels.shape[0]):
            input_ids = inputs["input_ids"][i]

            # Find assistant start token (im_start)
            assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

            # Find where assistant turn starts
            assistant_positions = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0]

            if len(assistant_positions) >= 2:  # Should have user and assistant im_start
                # Mask everything before the last im_start (which is assistant's turn)
                assistant_start = assistant_positions[-1].item()
                labels[i, :assistant_start] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs.get("image_grid_thw"),
            "labels": labels
        }


# ============================================================================
# 3. SETUP MODEL WITH LORA
# ============================================================================
def setup_model_and_processor():
    """Load model with LoRA configuration"""

    print(f"Loading model: {MODEL_NAME}")

    # Load processor
    processor = AutoProcessor.from_pretrained(
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
        dtype="auto",
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

    return model, processor


# ============================================================================
# 4. SAFE TRAINER WITH MEMORY MANAGEMENT
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
# 5. TRAINING
# ============================================================================
def train_model(model, processor, train_dataset, eval_dataset):
    """Train the model with proper crash recovery"""

    # Check for existing checkpoints
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,

        # Optimization
        bf16=True,
        optim="adamw_torch",
        warmup_steps=0.1,
        weight_decay=0.01,
        max_grad_norm=5.0,

        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},

        # Safety features
        save_safetensors=True,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=True,

        # Logging & saving
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        remove_unused_columns=False,
        report_to="none",

        # ✅ FIXED: Proper checkpoint resumption
        ignore_data_skip=False,  # Resume from exact data position
    )

    data_collator = Qwen3VLDataCollator(processor)

    trainer = SafeVisionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("\n" + "=" * 60)
    if last_checkpoint:
        print(f"🔄 RESUMING from: {last_checkpoint}")
    else:
        print(f"🆕 Starting NEW training")
    print(f"Training: {NUM_EPOCHS} epochs, batch {BATCH_SIZE}x{GRADIENT_ACCUMULATION}")
    print("=" * 60 + "\n")

    # Train with checkpoint resumption
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print("\nSaving final model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print(f"\n✓ Training complete! Model saved to: {OUTPUT_DIR}")
    return trainer

# ============================================================================
# 6. INFERENCE
# ============================================================================
def test_model(model, processor, test_dataset, num_samples=5):
    """Test the model"""

    print("\n" + "=" * 60)
    print("Testing model...")
    print("=" * 60 + "\n")

    model.eval()

    if len(test_dataset) == 0:
        print("No test samples to evaluate.")
        return

    for i in range(min(num_samples, len(test_dataset))):
        example = test_dataset[i]
        image = Image.open(example['image_path']).convert('RGB')

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": example['question']}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
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

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print(f"Example {i + 1}:")
        print(f"Q: {example['question']}")
        print(f"GT: {example['answer']}")
        print(f"Pred: {output_text}")
        print("-" * 60 + "\n")


# ============================================================================
# 7. MAIN
# ============================================================================
if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print("Qwen3-VL PathVQA Fine-tuning")
    print(f"{'=' * 60}\n")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}x{GRADIENT_ACCUMULATION}, LR: {LEARNING_RATE}")
    print(f"LoRA: {USE_LORA} (r={LORA_R}, alpha={LORA_ALPHA})")
    print(f"{'=' * 60}\n")

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Load dataset
            print("Step 1: Loading dataset...")
            dataset = load_pathvqa_dataset()

            # Setup model
            print("\nStep 2: Setting up model...")
            model, processor = setup_model_and_processor()

            # Train
            print("\nStep 3: Training...")
            trainer = train_model(
                model=model,
                processor=processor,
                train_dataset=dataset['train'],
                eval_dataset=dataset['validation']
            )

            # Test
            print("\nStep 4: Testing...")
            test_model(model, processor, dataset['test'])

            print("\n" + "=" * 60)
            print("All done! 🎉")
            print(f"Model saved to: {OUTPUT_DIR}")
            print("=" * 60)

            break  # Success - exit retry loop

        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            print(f"Checkpoints saved in: {OUTPUT_DIR}")
            print("Restart script to resume from last checkpoint")
            break

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n💥 OUT OF MEMORY!")
                print(f"Attempt {retry_count + 1}/{max_retries}")
                print("Clearing cache and retrying...")

                # Aggressive memory cleanup
                import gc
                import torch

                del model, processor
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                retry_count += 1
                if retry_count < max_retries:
                    print("Retrying in 10 seconds...")
                    import time

                    time.sleep(10)
                else:
                    print("Max retries reached. Suggestions:")
                    print("- Reduce BATCH_SIZE")
                    print("- Reduce MAX_PIXELS")
                    print("- Reduce MAX_SEQ_LENGTH")
                    raise
            else:
                print(f"\n❌ Runtime Error: {e}")
                print(f"Checkpoints saved in: {OUTPUT_DIR}")
                print("Restart script to resume from last checkpoint")
                raise

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print(f"Checkpoints saved in: {OUTPUT_DIR}")
            print("Restart script to resume from last checkpoint")
            raise