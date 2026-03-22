"""
Step 4: Evaluate fine-tuned Qwen3-VL-8B-Instruct on SLAKE test set
- Constrained decoding for yes/no: forces only "yes"/"no" token output
- Open questions: free generation with normalization (article removal)
- Full bf16
- GPU: H100
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
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from peft import PeftModel
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration
from transformers import Qwen3VLProcessor as AutoProcessor
from datetime import datetime

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

print("=" * 70)
print("SLAKE EVALUATION — Qwen3-VL-8B-Instruct  [H100]")
print("=" * 70 + "\n")

# ========== CONFIG ==========
MODEL_ID       = "Qwen/Qwen3-VL-8B-Instruct"
ADAPTER_PATH   = str(PROJECT_ROOT / "adapters")
TEST_DATA_PATH = PROJECT_ROOT / "preprocessed" / "test" / "preprocessed_data.pt"
RESULTS_DIR    = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ========== LOAD MODEL ==========
print("Loading model  (full bf16)...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    cache_dir=str(CACHE_DIR),
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
EOS_ID = processor.tokenizer.eos_token_id
PAD_ID = processor.tokenizer.pad_token_id

# Pre-compute yes/no token IDs for constrained decoding
YES_ID = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
NO_ID  = processor.tokenizer.encode("no",  add_special_tokens=False)[0]
print(f"yes token id: {YES_ID} | no token id: {NO_ID} | eos: {EOS_ID}")
print("Model loaded\n")

# ========== LOAD DATA ==========
print("Loading test data...")
data = torch.load(TEST_DATA_PATH, weights_only=False)
print(f"Loaded {len(data)} samples\n")


# ========== HELPERS ==========
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT    = str.maketrans("", "", string.punctuation)

def normalize(text: str) -> str:
    """VQA standard normalization: lower, remove articles + punctuation, collapse spaces."""
    text = text.strip().lower()
    text = _ARTICLES.sub(" ", text)
    text = text.translate(_PUNCT)
    text = " ".join(text.split())
    return text


def make_yn_prefix_fn(prompt_length):
    """Constrained decoding: force first generated token to yes/no, then EOS."""
    def prefix_fn(batch_id, input_ids):
        generated = len(input_ids) - prompt_length
        if generated == 0:
            return [YES_ID, NO_ID]
        else:
            return [EOS_ID]
    return prefix_fn


def token_f1(pred: str, gt: str) -> float:
    pred_tokens = pred.lower().split()
    gt_tokens   = gt.lower().split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common    = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_bleu(pred: str, gt: str):
    smoother = SmoothingFunction().method1
    hyp = pred.lower().split()
    ref = [gt.lower().split()]
    if not hyp:
        return 0.0, 0.0
    b1 = sentence_bleu(ref, hyp, weights=(1, 0, 0, 0),             smoothing_function=smoother)
    b4 = sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
    return b1, b4


def compute_meteor(pred: str, gt: str) -> float:
    try:
        return meteor_score([gt.lower().split()], pred.lower().split())
    except Exception:
        return 0.0


# ========== EVALUATE ==========
predictions    = []
ground_truths  = []
question_types = []
start_time     = time.time()

for idx in tqdm(range(len(data)), desc="Evaluating"):
    pred_text = ""
    gt_text   = ""
    q_type    = "unknown"

    try:
        sample         = data[idx]
        labels         = sample["labels"]
        answer_mask    = labels != -100
        answer_indices = answer_mask.nonzero(as_tuple=False)

        if len(answer_indices) == 0:
            predictions.append(pred_text)
            ground_truths.append(gt_text)
            question_types.append(q_type)
            continue

        answer_start = answer_indices[0].item()

        input_ids      = sample["input_ids"][:answer_start].unsqueeze(0).to(model.device)
        attention_mask = sample["attention_mask"][:answer_start].unsqueeze(0).to(model.device)
        pixel_values   = sample["pixel_values"].unsqueeze(0).to(model.device)

        image_grid_thw = sample["image_grid_thw"]
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)
        image_grid_thw = image_grid_thw.to(model.device)

        gt_tokens = labels[answer_mask]
        gt_text   = normalize(processor.tokenizer.decode(gt_tokens, skip_special_tokens=True))

        closed = gt_text in ("yes", "no")
        q_type = "yes_no" if closed else "open"

        prompt_length = input_ids.shape[1]

        with torch.no_grad():
            if closed:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
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
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=50,
                    min_new_tokens=1,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    eos_token_id=EOS_ID,
                    pad_token_id=PAD_ID,
                )
                new_tokens = generated_ids[0, prompt_length:]
                pred_text  = normalize(
                    processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
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

exact_matches    = sum(p == g for p, g in zip(predictions, ground_truths))
exact_match_rate = exact_matches / len(predictions)

yes_no_pairs   = [(p, g) for p, g, t in zip(predictions, ground_truths, question_types) if t == "yes_no"]
yes_no_correct = sum(p == g for p, g in yes_no_pairs)
yes_no_acc     = yes_no_correct / len(yes_no_pairs) if yes_no_pairs else 0

open_pairs     = [(p, g) for p, g, t in zip(predictions, ground_truths, question_types) if t == "open"]
open_correct   = sum(p == g for p, g in open_pairs)
open_acc       = open_correct / len(open_pairs) if open_pairs else 0
open_f1_scores    = [token_f1(p, g) for p, g in open_pairs]
open_f1           = sum(open_f1_scores) / len(open_f1_scores) if open_f1_scores else 0

bleu1_list, bleu4_list, meteor_list = [], [], []
for p, g in open_pairs:
    b1, b4 = compute_bleu(p, g)
    bleu1_list.append(b1)
    bleu4_list.append(b4)
    meteor_list.append(compute_meteor(p, g))

avg_bleu1  = sum(bleu1_list)  / len(bleu1_list)  if bleu1_list  else 0.0
avg_bleu4  = sum(bleu4_list)  / len(bleu4_list)  if bleu4_list  else 0.0
avg_meteor = sum(meteor_list) / len(meteor_list) if meteor_list else 0.0

# ========== SAVE ==========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

results = {
    "timestamp":          timestamp,
    "model":              MODEL_ID,
    "adapter":            ADAPTER_PATH,
    "test_data":          str(TEST_DATA_PATH),
    "total_samples":      len(predictions),
    "exact_match":        exact_match_rate,
    "exact_match_count":  exact_matches,
    "yes_no_accuracy":    yes_no_acc,
    "yes_no_samples":     len(yes_no_pairs),
    "yes_no_correct":     yes_no_correct,
    "open_accuracy":      open_acc,
    "open_samples":       len(open_pairs),
    "open_correct":       open_correct,
    "open_token_f1":      open_f1,
    "open_bleu1":         avg_bleu1,
    "open_bleu4":         avg_bleu4,
    "open_meteor":        avg_meteor,
    "total_time_minutes": total_time / 60,
    "samples_per_second": len(predictions) / total_time,
}

results_file = RESULTS_DIR / f"eval_results_{timestamp}.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

predictions_data = [
    {
        "idx":           i,
        "question":      data[i].get("question", ""),
        "prediction":    p,
        "ground_truth":  g,
        "question_type": t,
        "correct":       p == g,
        "token_f1":      token_f1(p, g) if t == "open" else None,
    }
    for i, (p, g, t) in enumerate(zip(predictions, ground_truths, question_types))
]
predictions_file = RESULTS_DIR / f"eval_predictions_{timestamp}.json"
with open(predictions_file, "w") as f:
    json.dump(predictions_data, f, indent=2)

# ========== PRINT ==========
print("\n" + "=" * 70)
print("FINAL RESULTS — Qwen3-VL-8B on SLAKE")
print("=" * 70)
print(f"Total samples:     {len(predictions)}")
print(f"  Yes/No:          {len(yes_no_pairs)}")
print(f"  Open:            {len(open_pairs)}")
print(f"\nOverall Exact:     {exact_match_rate*100:.2f}%  ({exact_matches}/{len(predictions)})")
print(f"Yes/No Accuracy:   {yes_no_acc*100:.2f}%  ({yes_no_correct}/{len(yes_no_pairs)})")
print(f"Open Exact Match:  {open_acc*100:.2f}%  ({open_correct}/{len(open_pairs)})")
print(f"Open Token F1:     {open_f1*100:.2f}%")
print(f"BLEU-1:            {avg_bleu1*100:.2f}%")
print(f"BLEU-4:            {avg_bleu4*100:.2f}%")
print(f"METEOR:            {avg_meteor*100:.2f}%")
print(f"\nTime: {total_time/60:.1f} min  ({len(predictions)/total_time:.2f} samples/sec)")
print("=" * 70)
print(f"\nSaved:")
print(f"  {results_file}")
print(f"  {predictions_file}")
print("=" * 70)

torch.cuda.empty_cache()
