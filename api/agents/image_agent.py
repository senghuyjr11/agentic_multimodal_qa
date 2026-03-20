import torch
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    ViTImageProcessor,
    ViTForImageClassification,
    Qwen3VLForConditionalGeneration,
)


class ModelType(Enum):
    PATHVQA = "pathvqa"
    VQA_RAD = "vqa_rad"
    REJECT  = "reject"


@dataclass
class ModelConfig:
    base_model_id: str
    adapter_path: str
    model_class: type


class ImageAgent:
    DEFAULT_QUESTIONS = {
        ModelType.PATHVQA: (
            "What pathological findings do you observe? "
            "Describe the specific location and characteristics."
        ),
        ModelType.VQA_RAD: (
            "What abnormalities are visible in this scan? "
            "Specify the anatomical location and extent."
        ),
    }

    REJECTION_MESSAGE = (
        "I can only analyze medical images (X-rays, CT scans, MRIs, ultrasounds, "
        "or pathology slides). This image doesn't appear to be a medical scan. "
        "Please upload a valid medical image."
    )

    def __init__(
        self,
        pathvqa_config: ModelConfig,
        vqa_rad_config: ModelConfig,
        classifier_path: str = "modality_classifier_v4",
    ):
        self.pathvqa_config = pathvqa_config
        self.vqa_rad_config = vqa_rad_config

        print("Loading modality classifier...")
        self._clf_processor = ViTImageProcessor.from_pretrained(classifier_path)
        self._clf_model = ViTForImageClassification.from_pretrained(classifier_path)
        self._clf_model.eval()
        if torch.cuda.is_available():
            self._clf_model.cuda()
        print("✓ Classifier loaded")

        self._pathvqa = None
        self._vqa_rad = None

    def preload_models(self):
        print("\nPre-loading VQA models...")
        self._load_pathvqa()
        self._load_vqa_rad()
        print("✓ All VQA models pre-loaded\n")

    def _load_pathvqa(self):
        if self._pathvqa is None:
            print("  Loading PathVQA (Qwen3-VL-8B)...")
            self._pathvqa = self._load_model(self.pathvqa_config)
            print("  ✓ PathVQA loaded")

    def _load_vqa_rad(self):
        if self._vqa_rad is None:
            print("  Loading SLAKE (Qwen3-VL-8B)...")
            self._vqa_rad = self._load_model(self.vqa_rad_config)
            print("  ✓ VQA-RAD loaded")

    def route(self, image: Image.Image) -> Tuple[ModelType, float]:
        inputs = self._clf_processor(image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._clf_model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)

        idx  = int(idx.item())
        conf = float(conf.item())

        if idx == 0:
            return ModelType.PATHVQA, conf
        elif idx == 1:
            return ModelType.VQA_RAD, conf
        else:
            return ModelType.REJECT, conf

    def _load_model(self, config: ModelConfig):
        processor = AutoProcessor.from_pretrained(config.base_model_id)

        base_model = config.model_class.from_pretrained(
            config.base_model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map={"": 0},
        )

        model = PeftModel.from_pretrained(base_model, config.adapter_path)
        model.eval()
        print(f"  ✓ Loaded adapter: {config.adapter_path}")

        return processor, model

    def predict(self, image_path: str, question: Optional[str] = None) -> dict:
        image = Image.open(image_path).convert("RGB")
        image_resized = image.resize((384, 384), Image.Resampling.BILINEAR)

        model_type, confidence = self.route(image)

        print(f"\n{'='*60}")
        print(f"[CLASSIFIER] {model_type.value.upper()} (confidence={confidence:.1%})")
        print(f"{'='*60}")

        if model_type == ModelType.REJECT:
            print(f"[REJECT] Non-medical image detected")
            return {
                "question": question or "N/A",
                "answer": self.REJECTION_MESSAGE,
                "model": "reject",
                "confidence": confidence,
                "ood": True,
            }

        if model_type == ModelType.PATHVQA:
            if self._pathvqa is None:
                self._load_pathvqa()
            processor, model = self._pathvqa
        else:
            if self._vqa_rad is None:
                self._load_vqa_rad()
            processor, model = self._vqa_rad

        final_question = question.strip() if question else self.DEFAULT_QUESTIONS[model_type]
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": final_question}]}
        ]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = processor(
            text=[text_prompt],
            images=[image_resized],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        answer = processor.decode(
            output_ids[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()

        print(f"[VQA] Answer: {answer[:100]}...")

        return {
            "question": final_question,
            "answer": answer,
            "model": model_type.value,
            "confidence": confidence,
            "ood": False,
        }


if __name__ == "__main__":
    pathvqa_config = ModelConfig(
        base_model_id="Qwen/Qwen3-VL-8B-Instruct",
        adapter_path="../pathvqa_qwen3vl_pipeline/adapters",
        model_class=Qwen3VLForConditionalGeneration,
    )

    vqa_rad_config = ModelConfig(
        base_model_id="Qwen/Qwen3-VL-8B-Instruct",
        adapter_path="../slake_qwen3vl_pipeline/adapters",
        model_class=Qwen3VLForConditionalGeneration,
    )

    agent = ImageAgent(
        pathvqa_config=pathvqa_config,
        vqa_rad_config=vqa_rad_config,
        classifier_path="../modality_classifier_v4",
    )

    result = agent.predict(
        "dataset_pathvqa/test/images/test_00001.jpg",
        "what do you see in this image?"
    )
    print(f"\nResult : {'REJECTED' if result['ood'] else 'ACCEPTED'}")
    print(f"Model  : {result['model']}")
    print(f"Answer : {result['answer'][:100]}...")