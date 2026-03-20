import os
import json
import torch
import numpy as np
from PIL import Image
from peft import get_peft_model, LoraConfig
from transformers import (
    Qwen2VLForConditionalGeneration, 
    Qwen2VLProcessor, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer
)
from torch.utils.data import Dataset
from evaluate import load

# Print CUDA information
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Model configuration
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"



# Quantization config for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load model with quantization
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # Let the model decide how to distribute across GPUs
)

# Load processor
processor = Qwen2VLProcessor.from_pretrained(MODEL_ID)





# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print model device distribution if available
if hasattr(model, 'hf_device_map'):
    print(f"Model device map: {model.hf_device_map}")
else:
    print("Model not using explicit device map")

# Print trainable parameters
model.print_trainable_parameters()

# Dataset class
class ImageCaptionDataset(Dataset):
    def __init__(self, dataset, processor, img_folder_path, target_image_size=(224, 224)):
        self.dataset = dataset
        self.processor = processor
        self.img_folder_path = img_folder_path
        self.target_image_size = target_image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            user_msg = item["messages"][0]["content"]
            assistant_msg = item["messages"][1]["content"]

            # Extract image path and question
            relative_image_path = next(c["image"] for c in user_msg if c["type"] == "image")
            question = next(c["text"] for c in user_msg if c["type"] == "text")
            
            # Load and resize image
            image_path = os.path.join(self.img_folder_path, relative_image_path)
            image = Image.open(image_path).convert("RGB")
            image = image.resize(self.target_image_size, Image.Resampling.LANCZOS)

            # Create combined text
            combined_text = question + " " + assistant_msg
            
            # Process inputs with tensors
            inputs = self.processor(
                text=combined_text,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            
            # Remove batch dimension
            for k in inputs:
                if isinstance(inputs[k], torch.Tensor) and inputs[k].dim() > 0:
                    inputs[k] = inputs[k].squeeze(0)
            
            # Add labels for training
            inputs["labels"] = inputs["input_ids"].clone()

            # Compute grid_thw with time dimension (t=1 for single images)
            patch_size = 14  # Confirm this matches the model's patch size
            height, width = self.target_image_size
            grid_h, grid_w = height // patch_size, width // patch_size
            inputs["image_grid_thw"] = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)  # Shape: [1, 3]

            return inputs
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            raise
# Data collator function
def data_collator(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    images = [item["pixel_values"] for item in batch]
    grid_thw = [item["image_grid_thw"] for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id
    )
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    pixel_values = torch.stack(images)
    image_grid_thw = torch.cat(grid_thw, dim=0)  # Shape: [batch_size, 3]

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
# Evaluation metrics
bertscore = load("bertscore")
bleu = load("bleu")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    
    # Get predictions from logits
    predictions = logits.argmax(-1)
    
    # Decode predictions and labels to text
    decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels with pad token id
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    
    # Calculate metrics
    bertscore_results = bertscore.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        model_type="distilbert-base-uncased",
        lang="en"
    )
    bleu_results = bleu.compute(predictions=decoded_preds, references=decoded_labels)

    # Return aggregated metrics
    return {
        "bleu-1": bleu_results["bleu"],
        "bertscore_precision": sum(bertscore_results["precision"]) / len(bertscore_results["precision"]),
        "bertscore_recall": sum(bertscore_results["recall"]) / len(bertscore_results["recall"]),
        "bertscore_f1": sum(bertscore_results["f1"]) / len(bertscore_results["f1"])
    }

# Load datasets
json_file_path_train = "/home/ali/storage1/Abdullah_Folder/GENAI/all_data/Data_Processing/Train_dataset_for_qwen.json"
json_file_path_val = "/home/ali/storage1/Abdullah_Folder/GENAI/all_data/Data_Processing/val_dataset_for_qwen.json"
img_folder_path = "/home/ali/storage1/Abdullah_Folder/GENAI/all_data/all_without_split/"

# Load JSON data
with open(json_file_path_train, 'r') as f:
    dataset_train = json.load(f)

with open(json_file_path_val, 'r') as f:
    dataset_val = json.load(f)
    dataset_val = dataset_val[:100]  # Using first 500 samples for validation

# Create datasets
train_dataset = ImageCaptionDataset(dataset_train, processor, img_folder_path, target_image_size=(224, 224))
val_dataset = ImageCaptionDataset(dataset_val, processor, img_folder_path, target_image_size=(224, 224))
# Print dataset sizes
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Training arguments
training_args = TrainingArguments(
    output_dir="./qwen2vl_qlora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=5,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    logging_steps=50,
    save_steps=300,
    eval_steps=300,
    eval_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="bertscore_f1",
    greater_is_better=True,
    fp16=True,
    remove_unused_columns=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=False,  # Important for DDP
    # Distributed training settings
    local_rank=-1,  # Let the TrainingArguments handle this
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Start training
print("Starting training...")
trainer.train()

# Save the trained model adapters
model.save_pretrained("./qwen2vl_qloraAdapters")
print("Training complete and model saved!")
