"""
Step 4: Evaluate fine-tuned Qwen3-VL-8B-Instruct on PathVQA test set
=========================================================
- Yes/No   : constrained decoding — guaranteed yes/no output, then F1/accuracy
- Open-ended: free greedy decoding → Exact Match, BLEU-1/4, METEOR, Token-F1
- Full bf16, GPU: H100

Metrics saved to:
  results/eval_results_<timestamp>.json
  results/eval_predictions_<timestamp>.json

Usage:
  python 4_evaluate.py
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR    = (PROJECT_ROOT.parent / ".hf_cache").resolve()
os.environ["HF_HOME"]                = str(CACHE_DIR)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from peft import PeftModel
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from datetime import datetime

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

print("=" * 70)
print("PATH-VQA  —  STEP 4: EVALUATE")
print("Model  : Qwen3-VL-8B-Instruct  [H100, full bf16]")
print("Metrics: Exact Match | Yes/No Acc | BLEU-1/4 | METEOR | Token-F1")
print("=" * 70 + "\n")

# ========== CONFIG ==========
MODEL_ID       = "Qwen/Qwen3-VL-8B-Instruct"
ADAPTER_PATH   = str(PROJECT_ROOT / "adapters")
TEST_DATA_PATH = PROJECT_ROOT / "preprocessed" / "test" / "preprocessed_data.pt"
RESULTS_DIR    = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

# ========== LOAD MODEL (full bf16) ==========
print("Loading base model ...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    cache_dir=str(CACHE_DIR),
)
model = model.cuda()

print("Loading LoRA/DoRA adapters ...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

processor = Qwen3VLProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
EOS_ID = processor.tokenizer.eos_token_id
PAD_ID = processor.tokenizer.pad_token_id or EOS_ID

# Pre-compute yes / no token IDs for constrained decoding
YES_ID = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
NO_ID  = processor.tokenizer.encode("no",  add_special_tokens=False)[0]
print(f"yes token: {YES_ID}  |  no token: {NO_ID}  |  eos: {EOS_ID}")
print(f"PAD: {PAD_ID}")
print("Model loaded.\n")

# ========== LOAD TEST DATA ==========
print("Loading test data ...")
data = torch.load(TEST_DATA_PATH, weights_only=False)
print(f"Loaded {len(data)} samples.\n")


# ========== HELPER FUNCTIONS ==========
def normalize(text: str) -> str:
    return text.strip().lower()


def is_yes_no(gt: str) -> bool:
    return gt in ("yes", "no")


def make_yn_prefix_fn(prompt_length: int):
    """Constrained decoding: first generated token must be yes or no, then EOS."""
    def prefix_fn(batch_id, input_ids):
        n_generated = len(input_ids) - prompt_length
        if n_generated == 0:
            return [YES_ID, NO_ID]    # force yes or no
        return [EOS_ID]               # immediately stop
    return prefix_fn


def token_f1(pred: str, gt: str) -> float:
    p_tok = pred.lower().split()
    g_tok = gt.lower().split()
    if not p_tok or not g_tok:
        return float(p_tok == g_tok)
    common = set(p_tok) & set(g_tok)
    if not common:
        return 0.0
    prec = len(common) / len(p_tok)
    rec  = len(common) / len(g_tok)
    return 2 * prec * rec / (prec + rec)


def compute_bleu(pred: str, gt: str):
    smoother = SmoothingFunction().method1
    hyp  = pred.lower().split()
    ref  = [gt.lower().split()]
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
        sample      = data[idx]
        labels      = sample["labels"]
        answer_mask = labels != -100
        ans_indices = answer_mask.nonzero(as_tuple=False)

        if len(ans_indices) == 0:
            predictions.append(pred_text)
            ground_truths.append(gt_text)
            question_types.append(q_type)
            continue

        answer_start = ans_indices[0].item()

        # Build prompt inputs (no answer tokens)
        input_ids      = sample["input_ids"][:answer_start].unsqueeze(0).to(model.device)
        attention_mask = sample["attention_mask"][:answer_start].unsqueeze(0).to(model.device)
        pixel_values   = sample["pixel_values"].unsqueeze(0).to(model.device)

        image_grid_thw = sample["image_grid_thw"]
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)
        image_grid_thw = image_grid_thw.to(model.device)

        # Ground-truth answer
        gt_tokens = labels[answer_mask]
        gt_text   = normalize(processor.tokenizer.decode(gt_tokens, skip_special_tokens=True))

        closed = is_yes_no(gt_text)
        q_type = "yes_no" if closed else "open"

        prompt_length = input_ids.shape[1]

        with torch.no_grad():
            if closed:
                # Constrained: guarantee yes or no as first token, then stop
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
                first_new = generated_ids[0, prompt_length].item()
                pred_text = "yes" if first_new == YES_ID else "no"

            else:
                # Free greedy decoding for open-ended
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

        # Periodic VRAM cleanup
        if idx % 500 == 0 and idx > 0:
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n[ERROR] sample {idx}: {e}")

    predictions.append(pred_text)
    ground_truths.append(gt_text)
    question_types.append(q_type)

# ========== COMPUTE METRICS ==========
total_time = time.time() - start_time
N          = len(predictions)

# --- Overall Exact Match ---
exact_correct    = sum(p == g for p, g in zip(predictions, ground_truths))
exact_match_rate = exact_correct / N

# --- Yes/No ---
yn_pairs   = [(p, g) for p, g, t in zip(predictions, ground_truths, question_types) if t == "yes_no"]
yn_correct = sum(p == g for p, g in yn_pairs)
yn_acc     = yn_correct / len(yn_pairs) if yn_pairs else 0.0

# --- Open-ended Exact Match ---
open_pairs   = [(p, g) for p, g, t in zip(predictions, ground_truths, question_types) if t == "open"]
open_correct = sum(p == g for p, g in open_pairs)
open_acc     = open_correct / len(open_pairs) if open_pairs else 0.0

# --- Generative metrics on open-ended ---
bleu1_list, bleu4_list, meteor_list, f1_list = [], [], [], []
for p, g in open_pairs:
    b1, b4 = compute_bleu(p, g)
    bleu1_list.append(b1)
    bleu4_list.append(b4)
    meteor_list.append(compute_meteor(p, g))
    f1_list.append(token_f1(p, g))

avg_bleu1  = sum(bleu1_list)  / len(bleu1_list)  if bleu1_list  else 0.0
avg_bleu4  = sum(bleu4_list)  / len(bleu4_list)  if bleu4_list  else 0.0
avg_meteor = sum(meteor_list) / len(meteor_list) if meteor_list else 0.0
avg_f1     = sum(f1_list)     / len(f1_list)     if f1_list     else 0.0

# ========== SAVE RESULTS ==========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

results = {
    "timestamp":           timestamp,
    "model":               MODEL_ID,
    "adapter":             ADAPTER_PATH,
    "test_data":           str(TEST_DATA_PATH),
    "total_samples":       N,
    # Primary
    "overall_exact_match": exact_match_rate,
    "exact_correct":       exact_correct,
    "yes_no_accuracy":     yn_acc,
    "yes_no_samples":      len(yn_pairs),
    "yes_no_correct":      yn_correct,
    "open_exact_match":    open_acc,
    "open_samples":        len(open_pairs),
    "open_correct":        open_correct,
    # Generative (open-ended)
    "open_bleu1":          avg_bleu1,
    "open_bleu4":          avg_bleu4,
    "open_meteor":         avg_meteor,
    "open_token_f1":       avg_f1,
    # Runtime
    "total_time_minutes":  total_time / 60,
    "samples_per_second":  N / total_time,
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

# ========== PRINT RESULTS ==========
print("\n" + "=" * 70)
print("FINAL RESULTS — Qwen3-VL-8B-Instruct on PathVQA")
print("=" * 70)
print(f"Total samples        : {N}")
print(f"\n{'─'*40}")
print(f"Overall Exact Match  : {exact_match_rate*100:.2f}%  ({exact_correct}/{N})")
print(f"Yes/No Accuracy      : {yn_acc*100:.2f}%  ({yn_correct}/{len(yn_pairs)})")
print(f"Open Exact Match     : {open_acc*100:.2f}%  ({open_correct}/{len(open_pairs)})")
print(f"\n{'─'*40}  Open-ended (n={len(open_pairs)})")
print(f"BLEU-1               : {avg_bleu1*100:.2f}%")
print(f"BLEU-4               : {avg_bleu4*100:.2f}%")
print(f"METEOR               : {avg_meteor*100:.2f}%")
print(f"Token F1             : {avg_f1*100:.2f}%")
print(f"\n{'─'*40}  Target Check")
print(f"Yes/No  ≥ 90%        : {'PASS' if yn_acc  >= 0.90 else 'FAIL'}  ({yn_acc*100:.2f}%)")
print(f"Overall ≥ 65%        : {'PASS' if exact_match_rate >= 0.65 else 'FAIL'}  ({exact_match_rate*100:.2f}%)")
print(f"\nTime : {total_time/60:.1f} min  |  Speed : {N/total_time:.2f} samples/sec")
print("=" * 70)
print(f"\nSaved:")
print(f"  {results_file}")
print(f"  {predictions_file}")
print("=" * 70)

torch.cuda.empty_cache()
