import os
from multiprocessing import freeze_support

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
import pandas as pd

from torchvision import transforms
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from peft import LoraConfig, get_peft_model

torch.backends.cudnn.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "ieee"
torch.backends.cudnn.rnn.fp32_precision = "ieee"


class PathVQADataset(Dataset):
    def __init__(self, csv_path, tokenizer, img_context_token, num_image_tokens, img_size=448):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.img_context_token = img_context_token
        self.num_image_tokens = num_image_tokens
        print(f"Loaded {len(self.data)} samples from {csv_path}")

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Prefix: <IMG_CONTEXT> repeated num_image_tokens times
        self.img_ctx_prefix = " ".join(
            [self.img_context_token] * self.num_image_tokens
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["image_path"]

        # 1) Load and transform image
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.transform(image)  # (3, H, W)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            pixel_values = torch.zeros(3, 448, 448)

        question = str(row["question"])
        answer = str(row["answer"])

        # 2) Text: [<IMG_CONTEXT> x num_image_tokens] + question + answer
        text = (
            f"{self.img_ctx_prefix}\n"
            f"Question: {question}\n"
            f"Answer: {answer}"
        )

        enc = self.tokenizer(
            text,
            max_length=1024,
            padding=False,    # pad in data collator
            truncation=True,
            return_tensors=None,
        )

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)

        # 3) LM-style: labels = input_ids
        labels = input_ids.clone()

        # 4) image_flags: (1,) → after stack → (B, 1) → squeeze(-1) → (B,)
        image_flags = torch.tensor([1], dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image_flags": image_flags,
        }


class VLDataCollator:
    """Top-level, picklable data collator for InternVL VQA training."""

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, batch):
        # pixel_values: stack to (B, 3, H, W)
        pixel_values = torch.stack(
            [item["pixel_values"] for item in batch],
            dim=0
        )

        # variable-length sequences → pad
        input_ids_list = [item["input_ids"] for item in batch]
        attention_mask_list = [item["attention_mask"] for item in batch]
        labels_list = [item["labels"] for item in batch]
        image_flags_list = [item["image_flags"] for item in batch]

        input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )
        # model's loss has no ignore_index, so pad labels with token_id
        labels = pad_sequence(
            labels_list, batch_first=True, padding_value=self.pad_token_id
        )

        # image_flags → (B, 1)
        image_flags = torch.stack(image_flags_list, dim=0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image_flags": image_flags,
        }


def infer_num_image_tokens(model, img_size=448):
    """
    Run model.extract_feature on a dummy image to see
    how many visual tokens per image the ViT produces.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    dummy = torch.zeros(1, 3, img_size, img_size, device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        vit_embeds = model.extract_feature(dummy)

    if vit_embeds.dim() == 3:
        _, F, _ = vit_embeds.shape
    elif vit_embeds.dim() == 2:
        F = vit_embeds.shape[1]
    else:
        raise RuntimeError(f"Unexpected vit_embeds shape: {vit_embeds.shape}")

    print(f"Inferred num_image_tokens per image: {F}")
    return F


def main():
    # Configuration
    MODEL_NAME = "OpenGVLab/InternVL2-1B"
    TRAIN_CSV = "dataset_pathvqa/train/train.csv"
    VAL_CSV = "dataset_pathvqa/validation/validation.csv"
    OUTPUT_DIR = "./internvl_lora_pathvqa"
    NUM_EPOCHS = 15
    BATCH_SIZE = 2
    LEARNING_RATE = 2e-4
    IMG_SIZE = 448

    print("=" * 50)
    print("InternVL2-1B LoRA Fine-tuning")
    print("=" * 50)

    # 1. Tokenizer + special token
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    img_context_token = "<IMG_CONTEXT>"

    # Ensure <IMG_CONTEXT> is in tokenizer.additional_special_tokens
    add_specials = []
    if img_context_token not in tokenizer.additional_special_tokens:
        add_specials.append(img_context_token)

    if add_specials:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens":
                    tokenizer.additional_special_tokens + add_specials
            }
        )
        print(f"Added special tokens: {add_specials}")

    # 2. Load model
    print("Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Resize LM embeddings if vocab size changed
    lm_embeddings = model.language_model.get_input_embeddings()
    if len(tokenizer) != lm_embeddings.num_embeddings:
        model.language_model.resize_token_embeddings(len(tokenizer))
        print("Resized token embeddings to", len(tokenizer))

    # Set img_context_token_id
    img_ctx_id = tokenizer.convert_tokens_to_ids(img_context_token)
    model.img_context_token_id = img_ctx_id
    if hasattr(model.config, "img_context_token_id"):
        model.config.img_context_token_id = img_ctx_id
    print(f"img_context_token_id set to {img_ctx_id} for token {img_context_token!r}")

    # 3. Infer how many image tokens the ViT produces
    num_image_tokens = infer_num_image_tokens(model, img_size=IMG_SIZE)

    # 4. Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["mlp1"],
        # generic PEFT; avoid PeftModelForCausalLM injecting inputs_embeds
        task_type=None,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Datasets
    print("Loading datasets...")
    train_dataset = PathVQADataset(
        TRAIN_CSV,
        tokenizer,
        img_context_token=img_context_token,
        num_image_tokens=num_image_tokens,
        img_size=IMG_SIZE,
    )

    val_dataset = PathVQADataset(
        VAL_CSV,
        tokenizer,
        img_context_token=img_context_token,
        num_image_tokens=num_image_tokens,
        img_size=IMG_SIZE,
    )

    data_collator = VLDataCollator(tokenizer)

    # 6. Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,  # now 15
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",  # required for early stopping
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",

        # ⭐ For early stopping to restore best weights
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # default metric name
        greater_is_better=False,  # lower loss is better
    )

    # 7. Trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training...")
    trainer.train()

    # 8. Save
    print("Saving model...")
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("Training completed!")


if __name__ == "__main__":
    freeze_support()
    main()
