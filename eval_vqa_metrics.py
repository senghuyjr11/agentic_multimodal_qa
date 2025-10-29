# eval_pathvqa_official.py
# Static evaluation for SimpleBiomedCLIPVQA on PathVQA test split,
# aligned with common practice in PathVQA papers:
# - Yes/No Accuracy
# - Free-form EM Accuracy
# - Overall EM Accuracy
# - Free-form BLEU (SacreBLEU)
# Optional extras:
# - Free-form ROUGE-1 F1
# - Free-form BERTScore F1

import json
from pathlib import Path
from collections import Counter
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from PIL import Image
import open_clip
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ----------------- Static config (EDIT IF NEEDED) -----------------
CSV_PATH = "dataset_pathvqa/test/test.csv"          # test CSV
CHECKPOINT_PATH = "outputs_simple/best_model.pt"    # best checkpoint
OUT_DIR = "eval_official_test"                      # output folder

BATCH_SIZE = 16
MAX_QUESTION_TOKENS = 64
GEN_MAX_LENGTH = 8
NUM_BEAMS = 4
FORCE_CPU = False

# Use instruction prefix if you trained/evaluated with it (set to True to enable)
USE_PROMPT_PREFIX = False

# Route yes/no questions via constrained scoring ("yes" vs "no")
ENABLE_YESNO_CONSTRAINED = True

# Optional extras: set to False to skip (faster)
COMPUTE_ROUGE1 = True
COMPUTE_BERTSCORE = True
BERTSCORE_MODEL = "microsoft/deberta-base-mnli"  # robust default
# ------------------------------------------------------------------


# ================== Text normalization & synonyms ==================
def normalize_number_tokens(s: str) -> str:
    mapping = {
        "zero": "0","one": "1","two": "2","three": "3","four": "4","five": "5",
        "six": "6","seven": "7","eight": "8","nine": "9","ten": "10"
    }
    tokens = re.split(r"(\W+)", s.lower())
    return "".join(mapping.get(t, t) for t in tokens)

# Add/extend as you see patterns in your wrong cases
BIOMED_SYNONYMS = {
    "neutrophil": {"neutrophil", "neutrophils"},
    "esophagus": {"esophagus", "oesophagus"},
    "leukocyte": {"leukocyte", "leucocyte", "white blood cell", "wbc", "white cells"},
    "hemorrhage": {"hemorrhage", "haemorrhage", "bleeding"},
    "arteriole": {"arteriole", "arterioles"},
    "alveolus": {"alveolus", "alveoli"},
    "yes": {"yes", "y", "true"},
    "no": {"no", "n", "false"},
    "right": {"right", "rt"},
    "left": {"left", "lt"},
}

def canonicalize(text: str) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = normalize_number_tokens(s)
    s = re.sub(r"\s+", " ", s).strip()
    for canon, variants in BIOMED_SYNONYMS.items():
        if s in variants:
            return canon
    return s


# ======================= Dataset & Model ===========================
class SimplePathVQADataset(Dataset):
    def __init__(self, csv_path, tokenizer, image_processor, max_length=64, use_prompt_prefix=False):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.use_prompt_prefix = use_prompt_prefix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_processor(image)

        question = str(row["question"]).strip() if pd.notna(row["question"]) else "what is this?"
        answer = str(row["answer"]).strip() if pd.notna(row["answer"]) else "unknown"

        if self.use_prompt_prefix:
            prompt = f"Answer the medical visual question with a short, factual answer.\nQuestion: {question}\nAnswer:"
        else:
            prompt = question

        q_tokens = self.tokenizer(
            prompt, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        a_tokens = self.tokenizer(
            answer, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "image": image_tensor,
            "question_input_ids": q_tokens["input_ids"].squeeze(0),
            "question_attention_mask": q_tokens["attention_mask"].squeeze(0),
            "answer_input_ids": a_tokens["input_ids"].squeeze(0),
            "answer_attention_mask": a_tokens["attention_mask"].squeeze(0),
            "question_text": question,
            "answer_text": answer,
        }


class SimpleBiomedCLIPVQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model, self.preprocess = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.t5.config.pad_token_id is None:
            self.t5.config.pad_token_id = self.tokenizer.pad_token_id

        vision_dim = 512
        t5_dim = self.t5.config.d_model
        self.vision_proj = nn.Linear(vision_dim, t5_dim)
        nn.init.normal_(self.vision_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.vision_proj.bias)

    def set_vision_trainable(self, train: bool):
        for p in self.clip_model.parameters():
            p.requires_grad = train
        self.clip_model.train(train)

    def forward(self, images, question_ids, question_mask, answer_ids=None):
        B = images.shape[0]
        with torch.set_grad_enabled(self.clip_model.training):
            if any(p.requires_grad for p in self.clip_model.parameters()):
                vision_features = self.clip_model.encode_image(images.float())
            else:
                with torch.no_grad():
                    vision_features = self.clip_model.encode_image(images.float())

        vis = self.vision_proj(vision_features).unsqueeze(1)  # [B,1,d]
        q_emb = self.t5.encoder.embed_tokens(question_ids)    # [B,L,d]

        combined = torch.cat([vis, q_emb], dim=1)
        vis_mask = torch.ones(B, 1, device=images.device, dtype=question_mask.dtype)
        attn = torch.cat([vis_mask, question_mask], dim=1)
        enc = self.t5.encoder(inputs_embeds=combined, attention_mask=attn, return_dict=True)

        if answer_ids is not None:
            out = self.t5(encoder_outputs=enc, attention_mask=attn, labels=answer_ids, return_dict=True)
            return out
        else:
            return enc, attn

    @torch.no_grad()
    def generate_freeform(self, images, question_ids, question_mask,
                          max_length=GEN_MAX_LENGTH, num_beams=NUM_BEAMS):
        self.eval()
        enc, attn = self.forward(images, question_ids, question_mask, answer_ids=None)
        gen = self.t5.generate(
            encoder_outputs=enc, attention_mask=attn,
            max_length=max_length, min_length=1,
            num_beams=num_beams, length_penalty=0.0,
            no_repeat_ngram_size=2, early_stopping=True
        )
        self.train(False)
        return gen

    @torch.no_grad()
    def generate_yesno(self, images, question_ids, question_mask):
        preds = []
        for i in range(images.shape[0]):
            img = images[i:i+1]
            qid = question_ids[i:i+1]
            qmsk = question_mask[i:i+1]
            enc, attn = self.forward(img, qid, qmsk, answer_ids=None)
            y_ids = self.tokenizer("yes", return_tensors="pt").input_ids.to(img.device)
            n_ids = self.tokenizer("no",  return_tensors="pt").input_ids.to(img.device)
            ly = self.t5(encoder_outputs=enc, attention_mask=attn, labels=y_ids, return_dict=True).loss.item()
            ln = self.t5(encoder_outputs=enc, attention_mask=attn, labels=n_ids, return_dict=True).loss.item()
            preds.append("yes" if ly <= ln else "no")
        return preds

    def looks_yesno(self, q_texts):
        starts = ("is","are","was","were","does","do","did","has","have","had",
                  "can","could","should","would","will","must")
        return [str(q).strip().lower().startswith(starts) for q in q_texts]


# =========================== Metrics ===============================
def compute_freeform_bleu(preds, refs):
    import sacrebleu
    return sacrebleu.corpus_bleu(preds, [refs]).score

def compute_freeform_rouge1(preds, refs):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    total = 0.0
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)["rouge1"].fmeasure
        total += s
    return 100.0 * total / max(len(preds), 1)

def compute_freeform_bertscore_f1(preds, refs, model_type):
    from bert_score import score as bertscore
    _, _, F1 = bertscore(preds, refs, lang="en", model_type=model_type, rescale_with_baseline=True)
    return 100.0 * float(F1.mean())


# =========================== Main eval =============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
    print(f"Device: {device}")

    # Build model & load checkpoint
    model = SimpleBiomedCLIPVQA().to(device)
    model.set_vision_trainable(False)
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Data
    ds = SimplePathVQADataset(
        csv_path=CSV_PATH,
        tokenizer=model.tokenizer,
        image_processor=model.preprocess,
        max_length=MAX_QUESTION_TOKENS,
        use_prompt_prefix=USE_PROMPT_PREFIX
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Buckets
    all_qs, all_refs, all_preds = [], [], []
    yn_qs, yn_refs, yn_preds = [], [], []          # yes/no subset
    ff_qs, ff_refs, ff_preds = [], [], []          # free-form subset

    with torch.no_grad():
        for batch in tqdm(dl, desc="Evaluating (test)"):
            images = batch["image"].to(device, non_blocking=True)
            q_ids  = batch["question_input_ids"].to(device, non_blocking=True)
            q_mask = batch["question_attention_mask"].to(device, non_blocking=True)

            q_texts = list(batch["question_text"])
            r_texts = list(batch["answer_text"])
            is_yn_flags = model.looks_yesno(q_texts)

            # Yes/No
            preds = [None] * images.shape[0]
            if ENABLE_YESNO_CONSTRAINED and any(is_yn_flags):
                idxs = torch.tensor([i for i,f in enumerate(is_yn_flags) if f], device=device, dtype=torch.long)
                if idxs.numel() > 0:
                    p_yn = model.generate_yesno(images[idxs], q_ids[idxs], q_mask[idxs])
                    for j, iidx in enumerate(idxs.tolist()):
                        preds[iidx] = p_yn[j]

            # Free-form
            ff_idxs = [i for i,f in enumerate(is_yn_flags) if not f]
            if len(ff_idxs) > 0:
                fi = torch.tensor(ff_idxs, device=device, dtype=torch.long)
                gen = model.generate_freeform(images[fi], q_ids[fi], q_mask[fi],
                                              max_length=GEN_MAX_LENGTH, num_beams=NUM_BEAMS)
                p_ff = model.tokenizer.batch_decode(gen, skip_special_tokens=True)
                for j, iidx in enumerate(fi.tolist()):
                    preds[iidx] = p_ff[j]

            # Collect
            all_qs.extend(q_texts)
            all_refs.extend(r_texts)
            all_preds.extend(preds)

            for i, flag in enumerate(is_yn_flags):
                if flag:
                    yn_qs.append(q_texts[i]); yn_refs.append(r_texts[i]); yn_preds.append(preds[i])
                else:
                    ff_qs.append(q_texts[i]); ff_refs.append(r_texts[i]); ff_preds.append(preds[i])

    # ---------- Compute metrics ----------
    # EM supports normalization & synonyms
    def em_score(preds, refs):
        return 100.0 * sum(1 for p, r in zip(preds, refs) if canonicalize(p) == canonicalize(r)) / max(len(preds), 1)

    overall_em = em_score(all_preds, all_refs)
    yn_em = em_score(yn_preds, yn_refs) if len(yn_refs) > 0 else 0.0
    ff_em = em_score(ff_preds, ff_refs) if len(ff_refs) > 0 else 0.0

    # BLEU for free-form
    ff_bleu = compute_freeform_bleu(ff_preds, ff_refs) if len(ff_refs) > 0 else 0.0

    # Optional extras
    ff_rouge1 = compute_freeform_rouge1(ff_preds, ff_refs) if (COMPUTE_ROUGE1 and len(ff_refs) > 0) else None
    ff_bertscore = compute_freeform_bertscore_f1(ff_preds, ff_refs, BERTSCORE_MODEL) if (COMPUTE_BERTSCORE and len(ff_refs) > 0) else None

    # ---------- Print & Save ----------
    counts = {
        "num_total": len(all_refs),
        "num_yesno": len(yn_refs),
        "num_freeform": len(ff_refs),
    }
    metrics = {
        "overall_EM_percent": overall_em,
        "yesno_accuracy_percent": yn_em,
        "freeform_EM_percent": ff_em,
        "freeform_sacrebleu": ff_bleu,
    }
    if ff_rouge1 is not None:
        metrics["freeform_rouge1_f1_percent"] = ff_rouge1
    if ff_bertscore is not None:
        metrics["freeform_bertscore_f1_percent"] = ff_bertscore

    print("\n=== Counts ===")
    print(json.dumps(counts, indent=2))
    print("\n=== Metrics (Primary) ===")
    print(json.dumps(metrics, indent=2))

    # quick error analysis (optional console dump)
    wrong = [(g, p, q) for g, p, q in zip(all_refs, all_preds, all_qs) if canonicalize(g) != canonicalize(p)]
    print("\nTop gold answers among wrong cases (first 20):")
    for gold, cnt in Counter([g for g, _, _ in wrong]).most_common(20):
        print(f"{gold}: {cnt}")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions with split tags
    df = pd.DataFrame({
        "question": all_qs,
        "reference": all_refs,
        "prediction": all_preds,
        "type": ["yesno" if q in yn_qs else "freeform" for q in all_qs]  # quick tag by membership
    })
    df.to_csv(out_dir / "predictions.csv", index=False)

    with (out_dir / "counts.json").open("w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved:\n- {out_dir / 'predictions.csv'}\n- {out_dir / 'counts.json'}\n- {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
