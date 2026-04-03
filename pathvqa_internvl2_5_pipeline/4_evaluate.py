"""
Step 4: Evaluate fine-tuned InternVL2.5-8B on PathVQA test set
- Yes/No: free generation, then extract yes/no
- Open-ended: free generation with normalized matching
- Reports strict_exact_match and overall_accuracy
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = (PROJECT_ROOT / ".hf_cache").resolve()
os.environ["HF_HOME"] = str(CACHE_DIR)

import json
import re
import string
import time
from datetime import datetime

import nltk
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

print("=" * 70)
print("PATH-VQA EVALUATION — InternVL2.5-8B")
print("Metrics: Strict Exact Match | overall_accuracy | Yes/No Acc | BLEU-1/4 | METEOR | Token-F1")
print("=" * 70 + "\n")

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
ADAPTER_PATH = str(PROJECT_ROOT / "adapters")
TEST_DATA_PATH = PROJECT_ROOT / "preprocessed" / "test" / "preprocessed_data.pt"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("Loading model (full bf16)...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cuda",
    use_flash_attn=True,
    trust_remote_code=True,
    cache_dir=str(CACHE_DIR),
)
model.language_model = PeftModel.from_pretrained(model.language_model, ADAPTER_PATH)
model = model.to(torch.bfloat16)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, use_fast=False, cache_dir=str(CACHE_DIR)
)
EOS_ID = tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id or tokenizer.eos_token_id
model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

print("Loading test data...")
shard_files = sorted(TEST_DATA_PATH.parent.glob("preprocessed_data*.pt"))
if not shard_files:
    raise FileNotFoundError(f"No preprocessed shards found in {TEST_DATA_PATH.parent}")
data = []
for shard_file in shard_files:
    shard = torch.load(shard_file, weights_only=False)
    data.extend(shard)
    print(f"  Loaded shard: {shard_file.name} ({len(shard)} samples)")
print(f"Loaded {len(data)} samples\n")

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = _ARTICLES.sub(" ", text)
    text = text.translate(_PUNCT)
    return " ".join(text.split())


def is_yes_no(gt: str) -> bool:
    return gt in ("yes", "no")


def extract_yn(raw: str) -> str:
    m = re.search(r"\b(yes|no)\b", raw.strip().lower())
    return m.group(1) if m else "no"


def token_f1(pred: str, gt: str) -> float:
    p_tok = pred.lower().split()
    g_tok = gt.lower().split()
    if not p_tok or not g_tok:
        return float(p_tok == g_tok)
    common = set(p_tok) & set(g_tok)
    if not common:
        return 0.0
    prec = len(common) / len(p_tok)
    rec = len(common) / len(g_tok)
    return 2 * prec * rec / (prec + rec)


def compute_bleu(pred: str, gt: str):
    smoother = SmoothingFunction().method1
    hyp = pred.lower().split()
    ref = [gt.lower().split()]
    if not hyp:
        return 0.0, 0.0
    b1 = sentence_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=smoother)
    b4 = sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
    return b1, b4


def compute_meteor(pred: str, gt: str) -> float:
    try:
        return meteor_score([gt.lower().split()], pred.lower().split())
    except Exception:
        return 0.0


def overall_vqa_score(pred: str, gt: str, q_type: str) -> float:
    pred = normalize(pred)
    gt = normalize(gt)
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
    if f1 >= 0.5:
        return 0.5
    if len(overlap) >= 1 and (len(pred_set) > 1 or len(gt_set) > 1):
        return 0.5
    return 0.0


predictions = []
ground_truths = []
question_types = []
questions = []
failed_samples = []
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
            continue

        answer_start = answer_indices[0].item()
        input_ids = sample["input_ids"][:answer_start].unsqueeze(0).to(model.device)
        attention_mask = sample["attention_mask"][:answer_start].unsqueeze(0).to(model.device)
        pixel_values = sample["pixel_values"].to(model.device, dtype=torch.bfloat16)
        question = sample.get("question", "")

        gt_tokens = labels[answer_mask]
        gt_text = normalize(tokenizer.decode(gt_tokens, skip_special_tokens=True))
        closed = is_yes_no(gt_text)
        q_type = "yes_no" if closed else "open"
        prompt_length = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5 if closed else 50,
                min_new_tokens=1,
                do_sample=False,
                num_beams=1,
                eos_token_id=EOS_ID,
                pad_token_id=PAD_ID,
            )

        if output_ids.shape[1] > prompt_length:
            new_tokens = output_ids[0, prompt_length:]
        else:
            new_tokens = output_ids[0]

        raw_pred = normalize(tokenizer.decode(new_tokens, skip_special_tokens=True))
        pred_text = extract_yn(raw_pred) if closed else raw_pred

    except Exception as e:
        print(f"\nError on sample {idx}: {e}")
        failed_samples.append({"idx": idx, "error": str(e)})
        continue

    predictions.append(pred_text)
    ground_truths.append(gt_text)
    question_types.append(q_type)
    questions.append(question)

total_time = time.time() - start_time
N = len(predictions)
if N == 0:
    raise RuntimeError("Evaluation failed for all samples.")

exact_correct = sum(p == g for p, g in zip(predictions, ground_truths))
exact_match_rate = exact_correct / N
overall_scores = [overall_vqa_score(p, g, t) for p, g, t in zip(predictions, ground_truths, question_types)]
overall_score_sum = sum(overall_scores)
overall_accuracy = overall_score_sum / N
overall_full_credit_count = sum(score == 1.0 for score in overall_scores)
overall_half_credit_count = sum(score == 0.5 for score in overall_scores)

yn_pairs = [(p, g) for p, g, t in zip(predictions, ground_truths, question_types) if t == "yes_no"]
yn_correct = sum(p == g for p, g in yn_pairs)
yn_acc = yn_correct / len(yn_pairs) if yn_pairs else 0.0

open_pairs = [(p, g) for p, g, t in zip(predictions, ground_truths, question_types) if t == "open"]
open_correct = sum(p == g for p, g in open_pairs)
open_acc = open_correct / len(open_pairs) if open_pairs else 0.0
open_overall_scores = [overall_vqa_score(p, g, t) for p, g, t in zip(predictions, ground_truths, question_types) if t == "open"]
open_overall_accuracy = sum(open_overall_scores) / len(open_overall_scores) if open_overall_scores else 0.0

bleu1_list, bleu4_list, meteor_list, f1_list = [], [], [], []
for p, g in open_pairs:
    b1, b4 = compute_bleu(p, g)
    bleu1_list.append(b1)
    bleu4_list.append(b4)
    meteor_list.append(compute_meteor(p, g))
    f1_list.append(token_f1(p, g))

avg_bleu1 = sum(bleu1_list) / len(bleu1_list) if bleu1_list else 0.0
avg_bleu4 = sum(bleu4_list) / len(bleu4_list) if bleu4_list else 0.0
avg_meteor = sum(meteor_list) / len(meteor_list) if meteor_list else 0.0
avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results = {
    "timestamp": timestamp,
    "model": MODEL_ID,
    "adapter": ADAPTER_PATH,
    "test_data": str(TEST_DATA_PATH),
    "total_samples": len(data),
    "scored_samples": N,
    "failed_samples": len(failed_samples),
    "strict_exact_match": exact_match_rate,
    "exact_correct": exact_correct,
    "overall_accuracy": overall_accuracy,
    "overall_score_sum": overall_score_sum,
    "overall_full_credit_count": overall_full_credit_count,
    "overall_half_credit_count": overall_half_credit_count,
    "yes_no_accuracy": yn_acc,
    "yes_no_samples": len(yn_pairs),
    "yes_no_correct": yn_correct,
    "open_strict_exact_match": open_acc,
    "open_overall_accuracy": open_overall_accuracy,
    "open_samples": len(open_pairs),
    "open_correct": open_correct,
    "open_bleu1": avg_bleu1,
    "open_bleu4": avg_bleu4,
    "open_meteor": avg_meteor,
    "open_token_f1": avg_f1,
    "total_time_minutes": total_time / 60,
    "samples_per_second": N / total_time,
}

results_file = RESULTS_DIR / f"eval_results_{timestamp}.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

predictions_data = [
    {
        "idx": i,
        "question": q,
        "prediction": p,
        "ground_truth": g,
        "question_type": t,
        "correct": p == g,
        "overall_score": overall_vqa_score(p, g, t),
        "token_f1": token_f1(p, g) if t == "open" else None,
    }
    for i, (q, p, g, t) in enumerate(zip(questions, predictions, ground_truths, question_types))
]
if failed_samples:
    predictions_data.append({"failed_samples": failed_samples[:20]})

predictions_file = RESULTS_DIR / f"eval_predictions_{timestamp}.json"
with open(predictions_file, "w") as f:
    json.dump(predictions_data, f, indent=2)

print("\n" + "=" * 70)
print("FINAL RESULTS — InternVL2.5-8B on PathVQA")
print("=" * 70)
print(f"Total samples:       {len(data)}")
print(f"Scored samples:      {N}")
print(f"Failed samples:      {len(failed_samples)}")
print(f"Strict Exact Match:  {exact_match_rate*100:.2f}%  ({exact_correct}/{N})")
print(f"overall_accuracy:    {overall_accuracy*100:.2f}%")
print(f"Yes/No Accuracy:     {yn_acc*100:.2f}%  ({yn_correct}/{len(yn_pairs)})")
print(f"Open Strict Exact:   {open_acc*100:.2f}%  ({open_correct}/{len(open_pairs)})")
print(f"Open Overall Acc:    {open_overall_accuracy*100:.2f}%")
print(f"BLEU-1 / BLEU-4:     {avg_bleu1*100:.2f}% / {avg_bleu4*100:.2f}%")
print(f"METEOR / Token F1:   {avg_meteor*100:.2f}% / {avg_f1*100:.2f}%")
print(f"\nSaved:")
print(f"  {results_file}")
print(f"  {predictions_file}")
print("=" * 70)
