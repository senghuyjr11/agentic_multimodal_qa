"""
Inference Script for PathVQA (v2 Compatible)
Matches the Preprocessing v2 logic:
1. Resizes input image to 384x384
2. Uses exact chat template structure
"""
import torch
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from PIL import Image

# ================= CONFIGURATION =================
# Must match your training script
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
# Path to where your training script saved the adapter
ADAPTER_PATH = "./qwen2vl_pathvqa_adapters"

# Input file settings
IMAGE_PATH = "dataset_pathvqa/test/images/test_00001.jpg"  # Change this to your image
QUESTION = "how are the histone subunits charged?"


# =================================================

def load_model_and_processor():
    print("Loading model and processor...")

    # 1. Load Processor
    processor = Qwen2VLProcessor.from_pretrained(MODEL_ID)

    # 2. Load Base Model (4-bit quantization to match training environment)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # 3. Load the Trained Adapter
    print(f"Loading adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("✓ Model loaded successfully\n")
    return model, processor


def predict(model, processor, image_path, question):
    # 1. Load and Resize Image
    # CRITICAL: We must resize to 384x384 because your preprocessing script did this.
    # The model learned features at this specific resolution.
    try:
        raw_image = Image.open(image_path).convert("RGB")
        image = raw_image.resize((384, 384), Image.Resampling.BILINEAR)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 2. Prepare Prompt (Chat Template)
    # This matches the 'user_messages' structure in your preprocessing code
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]

    # Generate the text prompt with special tokens
    text_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    # 3. Process Inputs
    # We pass the text prompt and the resized image to the processor
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    # 4. Move inputs to GPU
    inputs = inputs.to(model.device)

    # 5. Generate
    print("Generating answer...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,  # Allow enough space for the answer
            do_sample=False,  # Deterministic for testing
            num_beams=1,
            use_cache=True,
            # Ensure we use the correct stop tokens
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # 6. Decode
    # Clip the input tokens to see only the new generated text
    output_text_ids = generated_ids[0][len(inputs.input_ids[0]):]
    output_text = processor.decode(output_text_ids, skip_special_tokens=True)

    return output_text


if __name__ == "__main__":
    # Load resources
    model, processor = load_model_and_processor()

    # Run prediction
    print("=" * 60)
    print(f"Image:    {IMAGE_PATH}")
    print(f"Question: {QUESTION}")
    print("=" * 60)

    answer = predict(model, processor, IMAGE_PATH, QUESTION)

    print("\n" + "=" * 60)
    print(f"🤖 Model Answer: {answer}")
    print("=" * 60)

    # Clean up
    torch.cuda.empty_cache()