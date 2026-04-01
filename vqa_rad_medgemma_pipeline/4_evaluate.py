"""
Step 4: Evaluate fine-tuned MedGemma 4B IT on VQA-RAD test set
- Yes/no questions use constrained decoding to force yes/no
- Open questions use normalized exact match and token F1
- Adds overall_accuracy alongside strict exact match

Notes:
- This evaluation is intentionally aligned with the cleaner VQA-RAD pipelines.
- If MedGemma tokenization differs slightly in your environment, verify the
  yes/no token IDs on the server before large-scale evaluation.
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = (PROJECT_ROOT.parent / ".hf_cache").resolve()
os.environ["HF_HOME"] = str(CACHE_DIR)

import json
import re
import string
import time
from datetime import datetime

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

print("=" * 70)
print("VQA-RAD EVALUATION — MedGemma 4B IT")
print("Metrics: Strict Exact Match | overall_accuracy | Yes/No Accuracy")
print("=" * 70 + "\n")

# ========== CONFIG ==========
MODEL_ID = "google/medgemma-4b-it"
ADAPTER_PATH = str(PROJECT_ROOT / "adapters")
TEST_DATA_PATH = PROJECT_ROOT / "preprocessed" / "test" / "preprocessed_data.pt"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ========== LOAD MODEL ==========
print("Loading model (full bf16)...")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    cache_dir=str(CACHE_DIR),
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
EOS_ID = processor.tokenizer.eos_token_id
PAD_ID = processor.tokenizer.pad_token_id

YES_ID = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
NO_ID = processor.tokenizer.encode("no", add_special_tokens=False)[0]
print(f"yes token id: {YES_ID} | no token id: {NO_ID} | eos: {EOS_ID}")
print("Model loaded\n")

# ========== LOAD DATA ==========
print("Loading test data...")
data = torch.load(TEST_DATA_PATH, weights_only=False)
print(f"Loaded {len(data)} samples\n")


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = _ARTICLES.sub(" ", text)
    text = text.translate(_PUNCT)
    text = " ".join(text.split())
    return text


def canonicalize_answer(question: str, answer: str) -> str:
    q = question.strip().lower()
    a = normalize(answer)

    alias_map = {
        "left side": "left",
        "right side": "right",
        "left lung": "left",
        "right lung": "right",
        "left hemithorax": "left",
        "right hemithorax": "right",
        "ap view": "ap",
        "pa view": "pa",
        "ap projection": "ap",
        "pa projection": "pa",
        "axial view": "axial",
        "coronal view": "coronal",
        "sagittal view": "sagittal",
        "not seen here": "not seen",
        "not visualized": "not seen",
        "not visible": "not seen",
        "none seen": "not seen",
    }
    if a in alias_map:
        return alias_map[a]

    if a in ("yes", "no"):
        return a

    if "which side" in q or "right or left" in q or "left or right" in q:
        if re.search(r"\bleft\b", a):
            return "left"
        if re.search(r"\bright\b", a):
            return "right"

    if "ap" in q or "pa" in q or "projection" in q or "view" in q:
        if re.search(r"\bap\b", a):
            return "ap"
        if re.search(r"\bpa\b", a):
            return "pa"
        if "axial" in a:
            return "axial"
        if "coronal" in a:
            return "coronal"
        if "sagittal" in a:
            return "sagittal"

    if (
        "visible" in q
        or "seen" in q
        or "present" in q
        or "identified" in q
    ) and ("not seen" in a or "not visible" in a or "not visualized" in a):
        return "not seen"

    return a


def make_yn_prefix_fn(prompt_length):
    def prefix_fn(batch_id, input_ids):
        generated = len(input_ids) - prompt_length
        if generated == 0:
            return [YES_ID, NO_ID]
        return [EOS_ID]

    return prefix_fn


def token_f1(pred: str, gt: str) -> float:
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def overall_vqa_score(question: str, pred: str, gt: str, q_type: str) -> float:
    pred = canonicalize_answer(question, pred)
    gt = canonicalize_answer(question, gt)

    if not pred or not gt:
        return 0.0

    if pred == gt:
        return 1.0

    if q_type == "yes_no":
        return 0.0

    pred_set = set(pred.split())
    gt_set = set(gt.split())
    overlap = pred_set & gt_set
    f1 = token_f1(pred, gt)

    if pred_set == gt_set:
        return 1.0

    if pred in gt or gt in pred:
        if len(overlap) >= max(1, min(len(pred_set), len(gt_set))):
            return 1.0
        return 0.5

    if f1 >= 0.8:
        return 1.0

    directional_terms = {
        "left", "right", "ap", "pa", "axial", "coronal", "sagittal",
        "frontal", "lateral", "supine", "upright", "not", "seen",
    }
    if overlap and (pred_set | gt_set) <= directional_terms:
        return 1.0

    if f1 >= 0.5:
        return 0.5

    if len(overlap) >= 1 and (len(pred_set) > 1 or len(gt_set) > 1):
        return 0.5

    return 0.0


predictions = []
ground_truths = []
question_types = []
start_time = time.time()

for idx in tqdm(range(len(data)), desc="Evaluating"):
    pred_text = ""
    gt_text = ""
    q_type = "unknown"

    try:
        sample = data[idx]
        labels = sample["labels"]
        answer_mask = labels != -100
        answer_indices = answer_mask.nonzero(as_tuple=False)

        if len(answer_indices) == 0:
            predictions.append(pred_text)
            ground_truths.append(gt_text)
            question_types.append(q_type)
            continue

        answer_start = answer_indices[0].item()

        model_inputs = {
            "input_ids": sample["input_ids"][:answer_start].unsqueeze(0).to(model.device),
            "attention_mask": sample["attention_mask"][:answer_start].unsqueeze(0).to(model.device),
            "pixel_values": sample["pixel_values"].unsqueeze(0).to(model.device, dtype=torch.bfloat16),
        }

        if "token_type_ids" in sample:
            model_inputs["token_type_ids"] = sample["token_type_ids"][:answer_start].unsqueeze(0).to(model.device)

        if "image_grid_thw" in sample:
            image_grid_thw = sample["image_grid_thw"]
            if image_grid_thw.dim() == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)
            model_inputs["image_grid_thw"] = image_grid_thw.to(model.device)

        question = sample.get("question", "")
        gt_tokens = labels[answer_mask]
        gt_text = canonicalize_answer(
            question,
            processor.tokenizer.decode(gt_tokens, skip_special_tokens=True),
        )

        closed = gt_text in ("yes", "no")
        q_type = "yes_no" if closed else "open"
        prompt_length = model_inputs["input_ids"].shape[1]

        with torch.no_grad():
            if closed:
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=2,
                    min_new_tokens=1,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    eos_token_id=EOS_ID,
                    pad_token_id=PAD_ID,
                    prefix_allowed_tokens_fn=make_yn_prefix_fn(prompt_length),
                )
                first_token = generated_ids[0, prompt_length].item()
                pred_text = "yes" if first_token == YES_ID else "no"
            else:
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=50,
                    min_new_tokens=1,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    eos_token_id=EOS_ID,
                    pad_token_id=PAD_ID,
                )
                new_tokens = generated_ids[0, prompt_length:]
                pred_text = canonicalize_answer(
                    question,
                    processor.tokenizer.decode(new_tokens, skip_special_tokens=True),
                )

        if idx % 200 == 0 and idx > 0:
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"\nError on sample {idx}: {e}")

    predictions.append(pred_text)
    ground_truths.append(gt_text)
    question_types.append(q_type)

# ========== METRICS ==========
total_time = time.time() - start_time

exact_matches = sum(p == g for p, g in zip(predictions, ground_truths))
exact_match_rate = exact_matches / len(predictions)
overall_scores = [
    overall_vqa_score(data[i].get("question", ""), p, g, t)
    for i, (p, g, t) in enumerate(zip(predictions, ground_truths, question_types))
]
overall_score_sum = sum(overall_scores)
overall_accuracy = overall_score_sum / len(predictions)
overall_full_credit_count = sum(score == 1.0 for score in overall_scores)
overall_half_credit_count = sum(score == 0.5 for score in overall_scores)

yes_no_pairs = [(p, g) for p, g, t in zip(predictions, ground_truths, question_types) if t == "yes_no"]
yes_no_correct = sum(p == g for p, g in yes_no_pairs)
yes_no_acc = yes_no_correct / len(yes_no_pairs) if yes_no_pairs else 0

open_pairs = [(p, g) for p, g, t in zip(predictions, ground_truths, question_types) if t == "open"]
open_correct = sum(p == g for p, g in open_pairs)
open_acc = open_correct / len(open_pairs) if open_pairs else 0
open_f1_scores = [token_f1(p, g) for p, g in open_pairs]
open_f1 = sum(open_f1_scores) / len(open_f1_scores) if open_f1_scores else 0
open_overall_scores = [
    overall_vqa_score(data[i].get("question", ""), p, g, t)
    for i, (p, g, t) in enumerate(zip(predictions, ground_truths, question_types))
    if t == "open"
]
open_overall_accuracy = sum(open_overall_scores) / len(open_overall_scores) if open_overall_scores else 0

# ========== SAVE ==========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

results = {
    "timestamp": timestamp,
    "model": MODEL_ID,
    "adapter": ADAPTER_PATH,
    "test_data": str(TEST_DATA_PATH),
    "total_samples": len(predictions),
    "strict_exact_match": exact_match_rate,
    "exact_match_count": exact_matches,
    "overall_accuracy": overall_accuracy,
    "overall_score_sum": overall_score_sum,
    "overall_full_credit_count": overall_full_credit_count,
    "overall_half_credit_count": overall_half_credit_count,
    "yes_no_accuracy": yes_no_acc,
    "yes_no_samples": len(yes_no_pairs),
    "yes_no_correct": yes_no_correct,
    "open_strict_exact_match": open_acc,
    "open_overall_accuracy": open_overall_accuracy,
    "open_samples": len(open_pairs),
    "open_correct": open_correct,
    "open_token_f1": open_f1,
    "total_time_minutes": total_time / 60,
    "samples_per_second": len(predictions) / total_time,
}

results_file = RESULTS_DIR / f"eval_results_{timestamp}.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

predictions_data = [
    {
        "idx": i,
        "question": data[i].get("question", ""),
        "prediction": p,
        "ground_truth": g,
        "question_type": t,
        "correct": p == g,
        "overall_score": overall_vqa_score(data[i].get("question", ""), p, g, t),
        "token_f1": token_f1(p, g) if t == "open" else None,
    }
    for i, (p, g, t) in enumerate(zip(predictions, ground_truths, question_types))
]

predictions_file = RESULTS_DIR / f"eval_predictions_{timestamp}.json"
with open(predictions_file, "w") as f:
    json.dump(predictions_data, f, indent=2)

# ========== PRINT ==========
print("\n" + "=" * 70)
print("FINAL RESULTS — MedGemma 4B IT on VQA-RAD")
print("=" * 70)
print(f"Total samples:     {len(predictions)}")
print(f"  Yes/No:          {len(yes_no_pairs)}")
print(f"  Open:            {len(open_pairs)}")
print(f"\nStrict Exact Match:{exact_match_rate * 100:.2f}%  ({exact_matches}/{len(predictions)})")
print(
    f"overall_accuracy:  {overall_accuracy * 100:.2f}%  "
    f"(score={overall_score_sum:.1f}; full={overall_full_credit_count}, half={overall_half_credit_count})"
)
print(f"Yes/No Accuracy:   {yes_no_acc * 100:.2f}%  ({yes_no_correct}/{len(yes_no_pairs)})")
print(f"Open Strict Exact: {open_acc * 100:.2f}%  ({open_correct}/{len(open_pairs)})")
print(f"Open Overall Acc:  {open_overall_accuracy * 100:.2f}%")
print(f"Open Token F1:     {open_f1 * 100:.2f}%")
print(f"\nTarget check:")
print(f"  Strict >= 60%:   {'PASS' if exact_match_rate >= 0.60 else 'FAIL'} ({exact_match_rate * 100:.2f}%)")
print(f"\nTime: {total_time / 60:.1f} min  ({len(predictions) / total_time:.2f} samples/sec)")
print("=" * 70)
print(f"\nSaved:")
print(f"  {results_file}")
print(f"  {predictions_file}")
print("=" * 70)

torch.cuda.empty_cache()
