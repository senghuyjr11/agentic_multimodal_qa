import torch
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    ViTImageProcessor,
    ViTForImageClassification,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration
)


class ModelType(Enum):
    PATHVQA = "pathvqa"
    VQA_RAD = "vqa_rad"


@dataclass
class ModelConfig:
    base_model_id: str
    adapter_path: str
    model_class: type  # Qwen2VLForConditionalGeneration or Qwen3VLForConditionalGeneration


class ImageAgent:

    DEFAULT_QUESTIONS = {
        ModelType.PATHVQA: "Describe the key findings in this pathology image.",
        ModelType.VQA_RAD: "Describe the key findings in this medical image."
    }

    def __init__(
        self,
        pathvqa_config: ModelConfig,
        vqa_rad_config: ModelConfig,
        classifier_path: str
    ):
        self.pathvqa_config = pathvqa_config
        self.vqa_rad_config = vqa_rad_config

        # Load classifier
        print("Loading modality classifier...")
        self._clf_processor = ViTImageProcessor.from_pretrained(classifier_path)
        self._clf_model = ViTForImageClassification.from_pretrained(classifier_path)
        self._clf_model.eval()
        if torch.cuda.is_available():
            self._clf_model.cuda()
        print("✓ Classifier loaded")

        # Lazy load
        self._pathvqa = None
        self._vqa_rad = None

    def route(self, image: Image.Image) -> tuple[ModelType, float]:
        inputs = self._clf_processor(image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            probs = torch.softmax(self._clf_model(**inputs).logits, dim=-1)
            idx = probs.argmax().item()

        model_type = ModelType.PATHVQA if idx == 0 else ModelType.VQA_RAD
        return model_type, probs[0, idx].item()

    def _load_model(self, config: ModelConfig):
        print(f"Loading: {config.base_model_id}")

        processor = AutoProcessor.from_pretrained(config.base_model_id)

        base_model = config.model_class.from_pretrained(
            config.base_model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map="auto",
        )

        model = PeftModel.from_pretrained(base_model, config.adapter_path)
        model.eval()
        print(f"✓ Loaded {config.adapter_path}")

        return processor, model

    def predict(self, image_path: str, question: Optional[str] = None) -> dict:
        image = Image.open(image_path).convert("RGB")
        image_resized = image.resize((384, 384), Image.Resampling.BILINEAR)

        # Route
        model_type, confidence = self.route(image)

        print(f"\n{'='*50}")
        print(f"ROUTING: {model_type.value.upper()} ({confidence:.1%})")
        print(f"{'='*50}")

        # Get model
        if model_type == ModelType.PATHVQA:
            if self._pathvqa is None:
                self._pathvqa = self._load_model(self.pathvqa_config)
            processor, model = self._pathvqa
        else:
            if self._vqa_rad is None:
                self._vqa_rad = self._load_model(self.vqa_rad_config)
            processor, model = self._vqa_rad

        # Prepare input
        final_question = question.strip() if question else self.DEFAULT_QUESTIONS[model_type]

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": final_question}]}]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image_resized], padding=True, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        answer = processor.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        return {
            "question": final_question,
            "answer": answer,
            "model": model_type.value,
            "confidence": confidence
        }


if __name__ == "__main__":
    # Your configs
    pathvqa_config = ModelConfig(
        base_model_id="Qwen/Qwen2-VL-7B-Instruct",
        adapter_path="../qwen2vl_7b_pathvqa_adapters",
        model_class=Qwen2VLForConditionalGeneration
    )

    vqa_rad_config = ModelConfig(
        base_model_id="Qwen/Qwen3-VL-2B-Instruct",
        adapter_path="../qwen3vl_2b_vqa_rad_adapters",
        model_class=Qwen3VLForConditionalGeneration
    )

    agent = ImageAgent(
        pathvqa_config=pathvqa_config,
        vqa_rad_config=vqa_rad_config,
        classifier_path="../modality_classifier"
    )

    # Test PathVQA or VQA-RAD
    result = agent.predict(
        "dataset_pathvqa/test/images/test_00001.jpg",
        "how are the histone subunits charged?"
    )
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")