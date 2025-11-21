import os
import torch
from datasets import DatasetDict
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from PIL import Image
import pandas as pd
from typing import Dict, List
import evaluate

OUTPUT_DIR = "./qwen2vl-pathvqa-lora-r32"  # same as training
DATASET_PATH = "dataset_pathvqa"
MAX_NEW_TOKENS = 64


# -------------------------------
# 1. Load dataset (same as train)
# -------------------------------
def load_pathvqa_dataset(base_path=DATASET_PATH):
    def load_split(split_name):
        csv_path = os.path.join(base_path, split_name, f"{split_name}.csv")
        image_root_dir = os.path.join(base_path, split_name, "images")

        df = pd.read_csv(csv_path)
        valid = {"image_path": [], "question": [], "answer": []}
        for _, row in df.iterrows():
            img_filename = row["image_path"].split("/")[-1]
            img_path = os.path.join(image_root_dir, img_filename)
            if not os.path.exists(img_path):
                continue
            valid["image_path"].append(img_path)
            valid["question"].append(str(row["question"]))
            valid["answer"].append(str(row["answer"]))
        return Dataset.from_dict(valid)

    from datasets import Dataset
    dataset = DatasetDict({
        "train": load_split("train"),
        "validation": load_split("validation"),
        "test": load_split("test"),
    })
    return dataset


# -------------------------------
# 2. Load trained model + processor
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(OUTPUT_DIR)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    OUTPUT_DIR,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()

# -------------------------------
# 3. Metric objects (BLEU, ROUGE)
# -------------------------------
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


# -------------------------------
# 4. Evaluation loop
# -------------------------------
def evaluate_split(dataset, max_samples=None):
    preds = []
    refs = []

    for i, example in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        image = Image.open(example["image_path"]).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": example["question"]},
                ],
            }
        ]

        # apply chat template and tokenize
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        # remove the prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        pred_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        ref_text = str(example["answer"]).strip()

        preds.append(pred_text)
        refs.append(ref_text)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} examples...")

    # BLEU expects tokenized references: List[List[str]] and predictions: List[str]
    bleu_result = bleu.compute(
        predictions=preds,
        references=[[r] for r in refs],  # list of list
    )
    rouge_result = rouge.compute(
        predictions=preds,
        references=refs,
    )

    print("\nEvaluation results:")
    print("BLEU:", bleu_result)
    print("ROUGE:", rouge_result)

    return bleu_result, rouge_result


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_pathvqa_dataset()
    print("Running evaluation on TEST split...")
    evaluate_split(dataset["test"], max_samples=None)  # or set a small number first
