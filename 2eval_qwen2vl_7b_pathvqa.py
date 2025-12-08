"""
Evaluation Script for PathVQA - Saves Results to File
"""
import os

import json
import time
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from datetime import datetime

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

# ========== EVALUATE ==========
predictions = []
ground_truths = []
start_time = time.time()

for idx in tqdm(range(len(data)), desc="Evaluating"):
    try:
        sample = data[idx]
        labels = sample["labels"]
        answer_mask = (labels != -100)
        answer_indices = answer_mask.nonzero(as_tuple=False)

        if len(answer_indices) == 0:
            predictions.append("")
            ground_truths.append("")
            continue

        answer_start = answer_indices[0].item()

        input_ids = sample["input_ids"][:answer_start].unsqueeze(0).to(model.device)
        attention_mask = sample["attention_mask"][:answer_start].unsqueeze(0).to(model.device)
        pixel_values = sample["pixel_values"].unsqueeze(0).to(model.device)

        image_grid_thw = sample["image_grid_thw"]
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)
        image_grid_thw = image_grid_thw.to(model.device)

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

        new_tokens = generated_ids[0, input_ids.shape[1]:]
        pred_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        answer_tokens = labels[answer_mask]
        gt_text = processor.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

        predictions.append(pred_text)
        ground_truths.append(gt_text)

        if idx % 500 == 0 and idx > 0:
            torch.cuda.empty_cache()

    except Exception as e:
        predictions.append("")
        ground_truths.append("")

# ========== CALCULATE METRICS ==========
total_time = time.time() - start_time

exact_matches = sum(
    1 for p, g in zip(predictions, ground_truths)
    if p.lower().strip() == g.lower().strip()
)

partial_matches = sum(
    1 for p, g in zip(predictions, ground_truths)
    if len(set(p.lower().split()) & set(g.lower().split())) > 0 and p and g
)

yes_no_preds = [(p, g) for p, g in zip(predictions, ground_truths)
                if g.lower().strip() in ["yes", "no"]]
yes_no_correct = sum(1 for p, g in yes_no_preds if p.lower().strip() == g.lower().strip())

exact_match_rate = exact_matches / len(predictions)
partial_match_rate = partial_matches / len(predictions)
yes_no_acc = yes_no_correct / len(yes_no_preds) if yes_no_preds else 0

# ========== SAVE RESULTS ==========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save metrics
results = {
    "timestamp": timestamp,
    "model": MODEL_ID,
    "adapter": ADAPTER_PATH,
    "test_data": TEST_DATA_PATH,
    "total_samples": len(predictions),
    "exact_match": exact_match_rate,
    "exact_match_count": exact_matches,
    "partial_match": partial_match_rate,
    "partial_match_count": partial_matches,
    "yes_no_accuracy": yes_no_acc,
    "yes_no_samples": len(yes_no_preds),
    "yes_no_correct": yes_no_correct,
    "total_time_minutes": total_time / 60,
    "samples_per_second": len(predictions) / total_time,
}

results_file = f"eval_results_{timestamp}.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Metrics saved to: {results_file}")

# Save all predictions
predictions_data = [
    {"idx": i, "prediction": p, "ground_truth": g, "correct": p.lower().strip() == g.lower().strip()}
    for i, (p, g) in enumerate(zip(predictions, ground_truths))
]

predictions_file = f"eval_predictions_{timestamp}.json"
with open(predictions_file, "w") as f:
    json.dump(predictions_data, f, indent=2)
print(f"✓ Predictions saved to: {predictions_file}")

# ========== PRINT SUMMARY ==========
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Total samples:     {len(predictions)}")
print(f"Exact Match:       {exact_match_rate:.4f} ({exact_match_rate*100:.2f}%)")
print(f"Partial Match:     {partial_match_rate:.4f} ({partial_match_rate*100:.2f}%)")
print(f"Yes/No Accuracy:   {yes_no_acc:.4f} ({yes_no_acc*100:.2f}%) [{len(yes_no_preds)} samples]")
print(f"\nTotal time:        {total_time/60:.1f} minutes")
print(f"Speed:             {len(predictions)/total_time:.2f} samples/sec")
print("="*70)
print(f"\n📁 Results saved to:")
print(f"   - {results_file}")
print(f"   - {predictions_file}")
print("="*70)

torch.cuda.empty_cache()