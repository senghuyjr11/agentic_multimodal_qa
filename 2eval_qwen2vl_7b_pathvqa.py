"""
Evaluation Script for PathVQA - v2 Preprocessed Data
"""
from pathlib import Path
import os
PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = (PROJECT_ROOT / ".hf_cache").resolve()
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR))
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Change to your GPU

import time
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig

print("="*70)
print("PATHVQA EVALUATION - v2 DATA")
print("="*70 + "\n")

# ========== CONFIG ==========
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_PATH = "./qwen2vl_pathvqa_adapters"
TEST_DATA_PATH = "preprocessed_data_v2/test/preprocessed_data.pt"

# ========== LOAD MODEL ==========
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

processor = Qwen2VLProcessor.from_pretrained(MODEL_ID)
print("✓ Model loaded\n")

# ========== LOAD DATA ==========
print("Loading test data...")
data = torch.load(TEST_DATA_PATH, weights_only=False)
print(f"✓ Loaded {len(data)} samples\n")

print("="*70)
print("STARTING EVALUATION")
print("="*70 + "\n")

# ========== EVALUATE ==========
predictions = []
ground_truths = []

start_time = time.time()

for idx in tqdm(range(len(data)), desc="Evaluating"):
    try:
        sample = data[idx]

        # Get where answer starts
        labels = sample["labels"]
        answer_mask = (labels != -100)
        answer_indices = answer_mask.nonzero(as_tuple=False)

        if len(answer_indices) == 0:
            predictions.append("")
            ground_truths.append("")
            continue

        answer_start = answer_indices[0].item()

        # Prepare inputs (prompt only, not answer)
        input_ids = sample["input_ids"][:answer_start].unsqueeze(0).to(model.device)
        attention_mask = sample["attention_mask"][:answer_start].unsqueeze(0).to(model.device)
        pixel_values = sample["pixel_values"].unsqueeze(0).to(model.device)

        image_grid_thw = sample["image_grid_thw"]
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)
        image_grid_thw = image_grid_thw.to(model.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=50,
                min_new_tokens=1,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Decode prediction
        new_tokens = generated_ids[0, input_ids.shape[1]:]
        pred_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Decode ground truth
        answer_tokens = labels[answer_mask]
        gt_text = processor.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

        predictions.append(pred_text)
        ground_truths.append(gt_text)

        # Show first 5 examples
        if idx < 5:
            match = "✓" if pred_text.lower().strip() == gt_text.lower().strip() else "✗"
            print(f"\nExample {idx+1}:")
            print(f"  Pred: {pred_text}")
            print(f"  True: {gt_text}")
            print(f"  Match: {match}")

        # Memory cleanup
        if idx % 500 == 0 and idx > 0:
            torch.cuda.empty_cache()
            elapsed = time.time() - start_time
            speed = idx / elapsed
            eta = (len(data) - idx) / speed / 60
            print(f"\n⏱️  Progress: {idx}/{len(data)}, Speed: {speed:.2f} samples/sec, ETA: {eta:.1f} min")

    except Exception as e:
        print(f"\n❌ Error on sample {idx}: {e}")
        predictions.append("")
        ground_truths.append("")

# ========== METRICS ==========
total_time = time.time() - start_time

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

# Exact match
exact_matches = sum(
    1 for p, g in zip(predictions, ground_truths)
    if p.lower().strip() == g.lower().strip()
)

# Partial match (word overlap)
partial_matches = sum(
    1 for p, g in zip(predictions, ground_truths)
    if len(set(p.lower().split()) & set(g.lower().split())) > 0 and p and g
)

# Yes/No accuracy (subset)
yes_no_preds = [(p, g) for p, g in zip(predictions, ground_truths) 
                if g.lower().strip() in ["yes", "no"]]
yes_no_correct = sum(1 for p, g in yes_no_preds if p.lower().strip() == g.lower().strip())

exact_match_rate = exact_matches / len(predictions)
partial_match_rate = partial_matches / len(predictions)
yes_no_acc = yes_no_correct / len(yes_no_preds) if yes_no_preds else 0

print(f"Total samples:     {len(predictions)}")
print(f"Exact Match:       {exact_match_rate:.4f} ({exact_match_rate*100:.2f}%)")
print(f"Partial Match:     {partial_match_rate:.4f} ({partial_match_rate*100:.2f}%)")
print(f"Yes/No Accuracy:   {yes_no_acc:.4f} ({yes_no_acc*100:.2f}%) [{len(yes_no_preds)} samples]")
print(f"\nTotal time:        {total_time/60:.1f} minutes")
print(f"Speed:             {len(predictions)/total_time:.2f} samples/sec")
print("="*70)

# Show correct examples
print("\n✅ CORRECT (first 15):")
correct = [(p, g) for p, g in zip(predictions, ground_truths)
           if p.lower().strip() == g.lower().strip()]
for i, (pred, gt) in enumerate(correct[:15], 1):
    print(f"  {i}. '{pred}'")
print(f"\n  (Total correct: {len(correct)})")

# Show incorrect examples
print("\n❌ INCORRECT (first 15):")
incorrect = [(p, g) for p, g in zip(predictions, ground_truths)
             if p.lower().strip() != g.lower().strip() and p and g]
for i, (pred, gt) in enumerate(incorrect[:15], 1):
    print(f"  {i}. Pred: '{pred}'")
    print(f"     True: '{gt}'")

print("\n" + "="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)

torch.cuda.empty_cache()