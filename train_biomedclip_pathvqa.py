import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import pandas as pd
import open_clip
from pathlib import Path
from tqdm import tqdm
import re

# Simple config - no fancy stuff
TRAIN_CSV = 'dataset_pathvqa/train/train.csv'
VAL_CSV = 'dataset_pathvqa/validation/validation.csv'
OUTPUT_DIR = 'outputs_simple'
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 2e-5
EARLY_STOP_PATIENCE = 5


def normalize_answer(text):
    """Remove articles and extra spaces"""
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

        # Load image
        image = Image.open(row['image_path']).convert('RGB')
        image_tensor = self.image_processor(image)

        # Get question and answer
        question = str(row['question']).strip()
        answer = normalize_answer(row['answer'])  # Normalize here!

        # Tokenize
        q_tokens = self.tokenizer(question, max_length=64, padding='max_length',
                                  truncation=True, return_tensors='pt')
        a_tokens = self.tokenizer(answer, max_length=32, padding='max_length',
                                  truncation=True, return_tensors='pt')

        return {
            'image': image_tensor,
            'question_ids': q_tokens['input_ids'].squeeze(0),
            'question_mask': q_tokens['attention_mask'].squeeze(0),
            'answer_ids': a_tokens['input_ids'].squeeze(0),
            'answer_mask': a_tokens['attention_mask'].squeeze(0),
            'answer_text': answer,
            'question_text': question,
        }


class BiomedCLIPVQA(nn.Module):
    def __init__(self):
        super().__init__()

        # Load BiomedCLIP (frozen)
        self.clip_model, self.preprocess = open_clip.create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        # Load T5-base
        self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.t5.config.pad_token_id is None:
            self.t5.config.pad_token_id = self.tokenizer.pad_token_id

        # Projection layer (512 -> 768)
        self.projection = nn.Linear(512, 768)

    def forward(self, images, question_ids, question_mask, answer_ids=None):
        batch_size = images.size(0)

        # Get image features (frozen)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)

        # Project and add to question
        image_features = self.projection(image_features).unsqueeze(1)  # [B, 1, 768]
        question_embeds = self.t5.encoder.embed_tokens(question_ids)  # [B, L, 768]
        combined = torch.cat([image_features, question_embeds], dim=1)  # [B, 1+L, 768]

        # Make attention mask
        image_mask = torch.ones(batch_size, 1, device=images.device)
        full_mask = torch.cat([image_mask, question_mask], dim=1)

        # Encode
        encoder_output = self.t5.encoder(inputs_embeds=combined, attention_mask=full_mask)

        # Training: compute loss
        if answer_ids is not None:
            output = self.t5(encoder_outputs=encoder_output, attention_mask=full_mask,
                             labels=answer_ids)
            return output.loss

        # Inference: generate
        else:
            generated = self.t5.generate(encoder_outputs=encoder_output,
                                         attention_mask=full_mask,
                                         max_length=32, num_beams=3)
            return generated


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    count = 0

    for batch in tqdm(loader, desc='Training'):
        images = batch['image'].to(device)
        q_ids = batch['question_ids'].to(device)
        q_mask = batch['question_mask'].to(device)
        a_ids = batch['answer_ids'].to(device)
        a_mask = batch['answer_mask'].to(device)

        # Prepare labels
        labels = a_ids.clone()
        labels[a_mask == 0] = -100

        # Forward
        optimizer.zero_grad()
        loss = model(images, q_ids, q_mask, labels)

        # Skip bad batches
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

        # Calculate training accuracy on this batch
        with torch.no_grad():
            generated = model(images, q_ids, q_mask)
            predictions = model.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for pred, true in zip(predictions, batch['answer_text']):
                pred_norm = normalize_answer(pred)
                true_norm = normalize_answer(true)
                if pred_norm == true_norm:
                    correct += 1
                total += 1

    avg_loss = total_loss / count if count > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0
    return avg_loss, accuracy


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    loss_count = 0
    correct = 0
    total = 0

    # Track yes/no vs free-form separately
    yesno_correct = 0
    yesno_total = 0
    freeform_correct = 0
    freeform_total = 0

    # Store sample predictions
    sample_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Evaluating')):
            images = batch['image'].to(device)
            q_ids = batch['question_ids'].to(device)
            q_mask = batch['question_mask'].to(device)
            a_ids = batch['answer_ids'].to(device)
            a_mask = batch['answer_mask'].to(device)

            # Compute loss
            labels = a_ids.clone()
            labels[a_mask == 0] = -100
            loss = model(images, q_ids, q_mask, labels)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                loss_count += 1

            # Generate predictions
            generated = model(images, q_ids, q_mask)
            predictions = model.tokenizer.batch_decode(generated, skip_special_tokens=True)

            # Compare with ground truth
            for i, (pred, true, question) in enumerate(zip(predictions, batch['answer_text'], batch['question_text'])):
                pred_norm = normalize_answer(pred)
                true_norm = normalize_answer(true)
                is_correct = (pred_norm == true_norm)

                if is_correct:
                    correct += 1
                total += 1

                # Track by question type
                is_yesno = question.lower().strip().startswith(('is ', 'are ', 'does ', 'do ', 'can ', 'was ', 'were '))
                if is_yesno:
                    if is_correct:
                        yesno_correct += 1
                    yesno_total += 1
                else:
                    if is_correct:
                        freeform_correct += 1
                    freeform_total += 1

                # Store first 5 samples
                if batch_idx == 0 and i < 5:
                    sample_predictions.append({
                        'question': question,
                        'prediction': pred_norm,
                        'reference': true_norm,
                        'correct': is_correct
                    })

    avg_loss = total_loss / loss_count if loss_count > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0
    yesno_acc = 100.0 * yesno_correct / yesno_total if yesno_total > 0 else 0
    freeform_acc = 100.0 * freeform_correct / freeform_total if freeform_total > 0 else 0

    return avg_loss, accuracy, yesno_acc, freeform_acc, sample_predictions


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    print("\nLoading model...")
    model = BiomedCLIPVQA().to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Load data
    print("\nLoading data...")
    train_dataset = PathVQADataset(TRAIN_CSV, model.tokenizer, model.preprocess)
    val_dataset = PathVQADataset(VAL_CSV, model.tokenizer, model.preprocess)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Optimizer - simple Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Training loop
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    best_acc = 0
    patience = 0

    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    print("=" * 80)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'=' * 80}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'=' * 80}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)

        # Evaluate
        val_loss, val_acc, val_yesno_acc, val_freeform_acc, samples = evaluate(model, val_loader, device)

        # Print detailed results
        print(f"\n{'─' * 80}")
        print(f"TRAINING RESULTS:")
        print(f"  Train Loss:          {train_loss:.4f}")
        print(f"  Train Accuracy:      {train_acc:.2f}%")
        print(f"\nVALIDATION RESULTS:")
        print(f"  Val Loss:            {val_loss:.4f}")
        print(f"  Val Accuracy:        {val_acc:.2f}%")
        print(f"  Val Yes/No Acc:      {val_yesno_acc:.2f}%")
        print(f"  Val Free-form Acc:   {val_freeform_acc:.2f}%")
        print(f"{'─' * 80}")

        # Print sample predictions
        print(f"\nSAMPLE PREDICTIONS (First 5 from validation):")
        for i, sample in enumerate(samples, 1):
            status = "✓" if sample['correct'] else "✗"
            print(f"\n  {i}. {status} Q: {sample['question']}")
            print(f"       Pred: {sample['prediction']}")
            print(f"       True: {sample['reference']}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_yesno_acc': val_yesno_acc,
                'val_freeform_acc': val_freeform_acc
            }, f'{OUTPUT_DIR}/best_model.pt')
            print(f"\n{'─' * 80}")
            print(f"⭐ NEW BEST MODEL SAVED! (Val Acc: {best_acc:.2f}%)")
            print(f"{'─' * 80}")
        else:
            patience += 1
            print(f"\n{'─' * 80}")
            print(f"No improvement. Patience: {patience}/{EARLY_STOP_PATIENCE}")
            print(f"Best so far: {best_acc:.2f}%")
            print(f"{'─' * 80}")

        # Early stopping
        if patience >= EARLY_STOP_PATIENCE:
            print(f"\n{'=' * 80}")
            print(f"EARLY STOPPING at epoch {epoch}")
            print(f"{'=' * 80}")
            break

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved at: {OUTPUT_DIR}/best_model.pt")
    print("=" * 80)


if __name__ == '__main__':
    main()