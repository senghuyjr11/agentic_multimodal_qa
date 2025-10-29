import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import pandas as pd
import open_clip
from pathlib import Path
from tqdm import tqdm

class SimplePathVQADataset(Dataset):
    """Simple dataset for Path-VQA"""

    def __init__(self, csv_path, tokenizer, image_processor, max_length=64):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image_path = row['image_path']
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_processor(image)  # Returns [C, H, W] tensor

        # Get text
        question = str(row['question']).strip() if pd.notna(row['question']) else "what is this?"
        answer = str(row['answer']).strip() if pd.notna(row['answer']) else "unknown"

        # Tokenize
        question_tokens = self.tokenizer(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        answer_tokens = self.tokenizer(
            answer,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'image': image_tensor,
            'question_input_ids': question_tokens['input_ids'].squeeze(0),
            'question_attention_mask': question_tokens['attention_mask'].squeeze(0),
            'answer_input_ids': answer_tokens['input_ids'].squeeze(0),
            'answer_attention_mask': answer_tokens['attention_mask'].squeeze(0),
            'answer_text': answer
        }


class SimpleBiomedCLIPVQA(nn.Module):
    """Simple BiomedCLIP + T5 model for VQA"""

    def __init__(self):
        super().__init__()

        print("Loading BiomedCLIP...")
        self.clip_model, self.preprocess = open_clip.create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )

        # Freeze CLIP initially (unfreeze later)  # >>> CHANGED
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        print("Loading T5...")
        self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.t5.config.pad_token_id is None:
            self.t5.config.pad_token_id = self.tokenizer.pad_token_id

        # Simple projection layer: CLIP (512) -> T5 (d_model)
        vision_dim = 512
        t5_dim = self.t5.config.d_model
        self.vision_proj = nn.Linear(vision_dim, t5_dim)
        nn.init.normal_(self.vision_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.vision_proj.bias)

        print("Model ready!")

    # Allow toggling CLIP trainability later  # >>> CHANGED
    def set_vision_trainable(self, train: bool):
        for p in self.clip_model.parameters():
            p.requires_grad = train
        self.clip_model.train(train)

    def forward(self, images, question_ids, question_mask, answer_ids=None):
        batch_size = images.shape[0]

        # Extract vision features with CLIP
        # If CLIP is frozen, keep no_grad; if unfrozen, enable grad  # >>> CHANGED
        if any(p.requires_grad for p in self.clip_model.parameters()):
            vision_features = self.clip_model.encode_image(images.float())  # [B, 512]
        else:
            with torch.no_grad():
                vision_features = self.clip_model.encode_image(images.float())  # [B, 512]

        # Project to T5 space
        vision_features = self.vision_proj(vision_features)  # [B, t5_dim]
        vision_features = vision_features.unsqueeze(1)  # [B, 1, t5_dim]

        # Get question embeddings
        question_embeds = self.t5.encoder.embed_tokens(question_ids)  # [B, L, t5_dim]

        # Concatenate vision and question
        combined_embeds = torch.cat([vision_features, question_embeds], dim=1)  # [B, 1+L, t5_dim]

        # Create attention mask
        vision_mask = torch.ones(batch_size, 1, device=images.device, dtype=question_mask.dtype)
        combined_mask = torch.cat([vision_mask, question_mask], dim=1)

        # Encode
        encoder_outputs = self.t5.encoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            return_dict=True
        )

        # If training, compute loss
        if answer_ids is not None:
            outputs = self.t5(
                encoder_outputs=encoder_outputs,
                attention_mask=combined_mask,
                labels=answer_ids,
                return_dict=True
            )
            return outputs
        else:
            return encoder_outputs, combined_mask

    @torch.no_grad()
    def generate(self, images, question_ids, question_mask, max_length=64):
        self.eval()
        encoder_outputs, combined_mask = self.forward(images, question_ids, question_mask)

        generated = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=combined_mask,
            max_length=max_length,
            num_beams=2,
            early_stopping=True
        )
        self.train()
        return generated


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch in progress_bar:
        # Move to device
        images = batch['image'].to(device)
        question_ids = batch['question_input_ids'].to(device)
        question_mask = batch['question_attention_mask'].to(device)
        answer_ids = batch['answer_input_ids'].to(device)
        answer_mask = batch['answer_attention_mask'].to(device)

        # Prepare labels (mask padding tokens)
        labels = answer_ids.clone()
        labels[answer_mask == 0] = -100

        # Check if we have valid labels
        valid_labels = (labels != -100).any(dim=1)
        if not valid_labels.any():
            continue

        # Filter to valid samples
        images = images[valid_labels]
        question_ids = question_ids[valid_labels]
        question_mask = question_mask[valid_labels]
        labels = labels[valid_labels]

        # Forward pass
        optimizer.zero_grad(set_to_none=True)  # >>> CHANGED (minor hygiene)
        outputs = model(images, question_ids, question_mask, labels)
        loss = outputs.loss

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def evaluate(model, dataloader, device, epoch):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Eval]')

    with torch.no_grad():
        for batch in progress_bar:
            images = batch['image'].to(device)
            question_ids = batch['question_input_ids'].to(device)
            question_mask = batch['question_attention_mask'].to(device)
            answer_ids = batch['answer_input_ids'].to(device)
            answer_mask = batch['answer_attention_mask'].to(device)

            # Prepare labels
            labels = answer_ids.clone()
            labels[answer_mask == 0] = -100

            valid_labels = (labels != -100).any(dim=1)
            if valid_labels.any():
                # Compute loss
                outputs = model(
                    images[valid_labels],
                    question_ids[valid_labels],
                    question_mask[valid_labels],
                    labels[valid_labels]
                )
                total_loss += outputs.loss.item()
                num_batches += 1

            # Generate predictions
            generated = model.generate(images, question_ids, question_mask, max_length=32)
            pred_texts = model.tokenizer.batch_decode(generated, skip_special_tokens=True)

            # Calculate accuracy
            for pred, true in zip(pred_texts, batch['answer_text']):
                if pred.strip().lower() == true.strip().lower():
                    correct += 1
                total += 1

            # Update progress bar
            progress_bar.set_postfix({'acc': f'{(correct / total * 100 if total > 0 else 0):.2f}%'})

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = (correct / total * 100) if total > 0 else 0

    return avg_loss, accuracy, pred_texts


def main():
    # Simple config  # >>> CHANGED (more epochs + unfreeze plan + diff LRs)
    config = {
        'train_csv': 'dataset_pathvqa/train/train.csv',
        'val_csv': 'dataset_pathvqa/validation/validation.csv',
        'batch_size': 16,
        'num_epochs': 15,
        'base_lr': 5e-5,  # T5 + projection
        'vision_lr': 5e-6,  # CLIP (smaller)
        'weight_decay': 0.01,
        'output_dir': 'outputs_simple',
        'unfreeze_epoch': 2  # unfreeze CLIP from this epoch
    }

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = SimpleBiomedCLIPVQA().to(device)
    model.set_vision_trainable(False)  # start frozen  # >>> CHANGED

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (initially frozen vision): {trainable_params:,}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = SimplePathVQADataset(
        config['train_csv'],
        model.tokenizer,
        model.preprocess
    )

    val_dataset = SimplePathVQADataset(
        config['val_csv'],
        model.tokenizer,
        model.preprocess
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Optimizer with parameter groups (diff LR for vision)  # >>> CHANGED
    vision_params = []
    proj_params = []
    t5_params = []
    for n, p in model.named_parameters():
        if n.startswith('clip_model.'):
            vision_params.append(p)
        elif n.startswith('vision_proj'):
            proj_params.append(p)
        else:
            t5_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {'params': vision_params, 'lr': config['vision_lr'], 'weight_decay': config['weight_decay']},
            {'params': proj_params, 'lr': config['base_lr'], 'weight_decay': config['weight_decay']},
            {'params': t5_params, 'lr': config['base_lr'], 'weight_decay': config['weight_decay']},
        ],
        eps=1e-8, betas=(0.9, 0.999)
    )

    # Create output directory
    Path(config['output_dir']).mkdir(exist_ok=True)

    # Training loop
    print(f"\nStarting training for {config['num_epochs']} epochs...\n")
    best_accuracy = 0

    for epoch in range(1, config['num_epochs'] + 1):
        # Unfreeze CLIP from chosen epoch  # >>> CHANGED
        if epoch >= config['unfreeze_epoch']:
            model.set_vision_trainable(True)
        else:
            model.set_vision_trainable(False)

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate
        val_loss, val_accuracy, predictions = evaluate(model, val_loader, device, epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{config['num_epochs']} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_accuracy:.2f}%")

        # Print sample predictions
        print(f"\n  Sample Predictions:")
        for i in range(min(3, len(predictions))):
            print(f"    Q: {val_dataset.df.iloc[i]['question']}")
            print(f"    Pred: {predictions[i]}")
            print(f"    True: {val_dataset.df.iloc[i]['answer']}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'config': config
        }

        checkpoint_path = Path(config['output_dir']) / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_path = Path(config['output_dir']) / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  ⭐ New best model saved! (Acc: {best_accuracy:.2f}%)")

        print()  # Empty line between epochs

    print("=" * 60)
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
