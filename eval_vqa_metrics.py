import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import pandas as pd
import open_clip
from tqdm import tqdm
import re
import json
from collections import Counter
import numpy as np

# Config
TEST_CSV = 'dataset_pathvqa/test/test.csv'
CHECKPOINT = 'outputs_simple/best_model.pt'
OUTPUT_DIR = 'evaluation_results'
BATCH_SIZE = 16

# Create output directory
import os

os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_answer(text):
    """PathVQA standard normalization"""
    text = str(text).lower().strip()
    text = re.sub(r'\b(the|a|an)\b\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "unknown"


class PathVQADataset(Dataset):
    def __init__(self, csv_path, tokenizer, image_processor):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        image_tensor = self.image_processor(image)

        question = str(row['question']).strip()
        answer = normalize_answer(row['answer'])

        q_tokens = self.tokenizer(question, max_length=64, padding='max_length',
                                  truncation=True, return_tensors='pt')

        return {
            'image': image_tensor,
            'question_ids': q_tokens['input_ids'].squeeze(0),
            'question_mask': q_tokens['attention_mask'].squeeze(0),
            'question_text': question,
            'answer_text': answer
        }


class BiomedCLIPVQA(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip_model, self.preprocess = open_clip.create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.t5.config.pad_token_id is None:
            self.t5.config.pad_token_id = self.tokenizer.pad_token_id

        self.projection = nn.Linear(512, 768)

    def forward(self, images, question_ids, question_mask):
        batch_size = images.size(0)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)

        image_features = self.projection(image_features).unsqueeze(1)
        question_embeds = self.t5.encoder.embed_tokens(question_ids)
        combined = torch.cat([image_features, question_embeds], dim=1)

        image_mask = torch.ones(batch_size, 1, device=images.device)
        full_mask = torch.cat([image_mask, question_mask], dim=1)

        encoder_output = self.t5.encoder(inputs_embeds=combined, attention_mask=full_mask)

        generated = self.t5.generate(encoder_outputs=encoder_output,
                                     attention_mask=full_mask,
                                     max_length=32, num_beams=3)
        return generated


def is_yesno_question(question):
    """Detect yes/no questions"""
    q = question.lower().strip()
    return q.startswith(('is ', 'are ', 'does ', 'do ', 'can ', 'was ', 'were ', 'has ', 'have ', 'did '))


# =================== PathVQA Official Metrics ===================

def compute_exact_match(predictions, references):
    """Exact Match - Official PathVQA metric for open-ended questions"""
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return 100.0 * correct / len(predictions) if len(predictions) > 0 else 0.0


def compute_macro_f1(predictions, references):
    """Macro-averaged F1 - Official PathVQA metric for open-ended questions"""
    # Get unique answers
    all_answers = set(predictions + references)

    f1_scores = []
    for answer in all_answers:
        # True positives, false positives, false negatives
        tp = sum(1 for p, r in zip(predictions, references) if p == answer and r == answer)
        fp = sum(1 for p, r in zip(predictions, references) if p == answer and r != answer)
        fn = sum(1 for p, r in zip(predictions, references) if p != answer and r == answer)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    return 100.0 * np.mean(f1_scores) if f1_scores else 0.0


def compute_bleu(predictions, references):
    """BLEU score - Official PathVQA metric for open-ended questions"""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk

    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    smoothie = SmoothingFunction().method4
    bleu_scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = [ref.split()]

        if len(pred_tokens) == 0 or len(ref_tokens[0]) == 0:
            bleu_scores.append(0.0)
        else:
            # BLEU-4 with uniform weights (standard for VQA)
            score = sentence_bleu(ref_tokens, pred_tokens,
                                  weights=(0.25, 0.25, 0.25, 0.25),
                                  smoothing_function=smoothie)
            bleu_scores.append(score)

    return 100.0 * np.mean(bleu_scores) if bleu_scores else 0.0


def compute_rouge(predictions, references):
    """ROUGE-L - Additional metric commonly used in VQA papers"""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        scores = []
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)['rougeL'].fmeasure
            scores.append(score)

        return 100.0 * np.mean(scores) if scores else 0.0
    except ImportError:
        print("Warning: rouge_score not installed. Skipping ROUGE metric.")
        return None


def compute_bertscore(predictions, references):
    """BERTScore - Additional semantic similarity metric"""
    try:
        from bert_score import score as bertscore

        if len(predictions) == 0:
            return None

        # Use a lighter model for faster evaluation
        P, R, F1 = bertscore(predictions, references, lang="en",
                             model_type="microsoft/deberta-base-mnli",
                             rescale_with_baseline=True,
                             verbose=False)

        return 100.0 * float(F1.mean())
    except ImportError:
        print("Warning: bert_score not installed. Skipping BERTScore metric.")
        return None
    except Exception as e:
        print(f"Warning: BERTScore computation failed: {e}")
        return None


# =================== Main Evaluation ===================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from {CHECKPOINT}...")
    model = BiomedCLIPVQA().to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded successfully!")

    # Load test data
    print(f"\nLoading test data from {TEST_CSV}...")
    dataset = PathVQADataset(TEST_CSV, model.tokenizer, model.preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)
    print(f"✓ Test samples: {len(dataset)}")

    # Generate predictions
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)

    all_predictions = []
    all_references = []
    all_questions = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch['image'].to(device)
            q_ids = batch['question_ids'].to(device)
            q_mask = batch['question_mask'].to(device)

            # Generate
            generated = model(images, q_ids, q_mask)
            predictions = model.tokenizer.batch_decode(generated, skip_special_tokens=True)

            # Normalize and collect
            for pred, true, question in zip(predictions, batch['answer_text'], batch['question_text']):
                pred_norm = normalize_answer(pred)
                true_norm = normalize_answer(true)

                all_predictions.append(pred_norm)
                all_references.append(true_norm)
                all_questions.append(question)

    # Separate yes/no and open-ended questions
    yesno_preds, yesno_refs = [], []
    open_preds, open_refs = [], []

    for pred, ref, question in zip(all_predictions, all_references, all_questions):
        if is_yesno_question(question):
            yesno_preds.append(pred)
            yesno_refs.append(ref)
        else:
            open_preds.append(pred)
            open_refs.append(ref)

    # =================== Compute Metrics ===================
    print("\n" + "=" * 80)
    print("COMPUTING METRICS (PathVQA Official Protocol)")
    print("=" * 80)

    results = {}

    # 1. Yes/No Accuracy (official metric)
    if len(yesno_preds) > 0:
        yesno_acc = compute_exact_match(yesno_preds, yesno_refs)
        results['yesno_accuracy'] = round(yesno_acc, 2)
        print(f"\n✓ Yes/No Accuracy: {yesno_acc:.2f}% ({len(yesno_preds)} questions)")

    # 2. Open-ended metrics (official)
    if len(open_preds) > 0:
        print(f"\n✓ Computing open-ended metrics ({len(open_preds)} questions)...")

        # Exact Match
        open_em = compute_exact_match(open_preds, open_refs)
        results['open_exact_match'] = round(open_em, 2)
        print(f"  - Exact Match: {open_em:.2f}%")

        # Macro F1
        open_f1 = compute_macro_f1(open_preds, open_refs)
        results['open_macro_f1'] = round(open_f1, 2)
        print(f"  - Macro-averaged F1: {open_f1:.2f}%")

        # BLEU
        open_bleu = compute_bleu(open_preds, open_refs)
        results['open_bleu'] = round(open_bleu, 2)
        print(f"  - BLEU: {open_bleu:.2f}%")

    # 3. Overall Accuracy
    overall_acc = compute_exact_match(all_predictions, all_references)
    results['overall_accuracy'] = round(overall_acc, 2)
    print(f"\n✓ Overall Accuracy: {overall_acc:.2f}% ({len(all_predictions)} questions)")

    # 4. Official PathVQA Ranking Score (macro average of 4 metrics)
    if len(yesno_preds) > 0 and len(open_preds) > 0:
        official_score = (yesno_acc + open_em + open_f1 + open_bleu) / 4
        results['official_pathvqa_score'] = round(official_score, 2)
        print(f"\n✓ Official PathVQA Score (avg of 4 metrics): {official_score:.2f}%")

    # =================== Additional Metrics (Common in Papers) ===================
    print("\n" + "=" * 80)
    print("ADDITIONAL METRICS (Commonly Reported in Papers)")
    print("=" * 80)

    if len(open_preds) > 0:
        # ROUGE-L
        rouge = compute_rouge(open_preds, open_refs)
        if rouge is not None:
            results['open_rougeL'] = round(rouge, 2)
            print(f"  - ROUGE-L: {rouge:.2f}%")

        # BERTScore
        print("  - Computing BERTScore (may take a few minutes)...")
        bertscore = compute_bertscore(open_preds, open_refs)
        if bertscore is not None:
            results['open_bertscore_f1'] = round(bertscore, 2)
            print(f"  - BERTScore F1: {bertscore:.2f}%")

    # =================== Summary ===================
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print("\nOfficial PathVQA Metrics:")
    print(f"  1. Yes/No Accuracy:        {results.get('yesno_accuracy', 'N/A')}%")
    print(f"  2. Open Exact Match:       {results.get('open_exact_match', 'N/A')}%")
    print(f"  3. Open Macro F1:          {results.get('open_macro_f1', 'N/A')}%")
    print(f"  4. Open BLEU:              {results.get('open_bleu', 'N/A')}%")
    print(f"  → Official Score (avg):    {results.get('official_pathvqa_score', 'N/A')}%")
    print(f"\nOverall Accuracy:            {results.get('overall_accuracy', 'N/A')}%")

    if 'open_rougeL' in results or 'open_bertscore_f1' in results:
        print(f"\nAdditional Metrics:")
        if 'open_rougeL' in results:
            print(f"  - ROUGE-L:                 {results['open_rougeL']}%")
        if 'open_bertscore_f1' in results:
            print(f"  - BERTScore F1:            {results['open_bertscore_f1']}%")

    print("=" * 80)

    # =================== Save Results ===================

    # Save metrics JSON
    with open(f'{OUTPUT_DIR}/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Metrics saved to {OUTPUT_DIR}/metrics.json")

    # Save detailed predictions
    predictions_df = pd.DataFrame({
        'question': all_questions,
        'prediction': all_predictions,
        'reference': all_references,
        'question_type': ['yes/no' if is_yesno_question(q) else 'open-ended' for q in all_questions],
        'correct': [p == r for p, r in zip(all_predictions, all_references)]
    })
    predictions_df.to_csv(f'{OUTPUT_DIR}/predictions.csv', index=False)
    print(f"✓ Predictions saved to {OUTPUT_DIR}/predictions.csv")

    # Save error analysis
    errors_df = predictions_df[predictions_df['correct'] == False]
    if len(errors_df) > 0:
        errors_df.to_csv(f'{OUTPUT_DIR}/errors.csv', index=False)
        print(f"✓ Error cases saved to {OUTPUT_DIR}/errors.csv ({len(errors_df)} errors)")

    # Save answer distribution analysis
    answer_dist = {
        'top_predicted_answers': dict(Counter(all_predictions).most_common(20)),
        'top_reference_answers': dict(Counter(all_references).most_common(20))
    }
    with open(f'{OUTPUT_DIR}/answer_distribution.json', 'w') as f:
        json.dump(answer_dist, f, indent=2)
    print(f"✓ Answer distribution saved to {OUTPUT_DIR}/answer_distribution.json")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()