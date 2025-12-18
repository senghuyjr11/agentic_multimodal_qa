"""
image_agent.py - PathVQA vs VQA-RAD router with OOD rejection (2-class classifier)

Key features:
- Adds ModelType.UNKNOWN
- Uses OOD rejection with MSP (max softmax prob) + entropy + energy
- Blocks Qwen inference when input is out-of-domain
- Lazy-loads PATHVQA and VQA-RAD Qwen adapters as before

Usage:
  python image_agent.py

Notes:
- Thresholds MUST be tuned for your classifier using a small OOD calibration set.
- Default thresholds are conservative starting points.
"""

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
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)


class ModelType(Enum):
    PATHVQA = "pathvqa"
    VQA_RAD = "vqa_rad"
    UNKNOWN = "unknown"


@dataclass
class ModelConfig:
    base_model_id: str
    adapter_path: str
    model_class: type  # Qwen2VLForConditionalGeneration or Qwen3VLForConditionalGeneration

@dataclass
class OODConfig:
    msp_threshold: float = 0.80
    entropy_threshold: float = 0.55
    energy_threshold: float = -2.0
    energy_temperature: float = 1.0

def softmax_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: [B, C]
    returns: [B]
    """
    return -(probs * (probs + 1e-12).log()).sum(dim=-1)


def energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """
    Energy-based OOD score.
    logits: [B, C]
    returns: [B]
    Typical behavior: OOD => higher energy (often less negative).
    """
    return -T * torch.logsumexp(logits / T, dim=-1)


class ImageAgent:
    DEFAULT_QUESTIONS = {
        ModelType.PATHVQA: "What do you see in this pathology image?",
        ModelType.VQA_RAD: "What abnormalities are visible in this scan?",
    }

    def __init__(
            self,
            pathvqa_config: ModelConfig,
            vqa_rad_config: ModelConfig,
            classifier_path: str,
            ood_config: OODConfig = None  # Add this
    ):
        self.ood_config = ood_config or OODConfig()
        self.pathvqa_config = pathvqa_config
        self.vqa_rad_config = vqa_rad_config

        self.msp_threshold = ood_config.msp_threshold
        self.entropy_threshold = ood_config.entropy_threshold
        self.energy_threshold = ood_config.energy_threshold
        self.energy_temperature = ood_config.energy_temperature

        # Load classifier
        print("Loading modality classifier...")
        self._clf_processor = ViTImageProcessor.from_pretrained(classifier_path)
        self._clf_model = ViTForImageClassification.from_pretrained(classifier_path)
        self._clf_model.eval()
        if torch.cuda.is_available():
            self._clf_model.cuda()
        print("✓ Classifier loaded")

        # Lazy load VLMs
        self._pathvqa = None
        self._vqa_rad = None

    def route(self, image: Image.Image) -> Tuple[ModelType, float]:
        """
        Returns:
          (model_type, confidence)
        confidence is MSP (max softmax probability).
        """
        inputs = self._clf_processor(image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._clf_model(**inputs).logits  # [1, 2]
            probs = torch.softmax(logits, dim=-1)      # [1, 2]

            msp, idx = probs.max(dim=-1)               # [1], [1]
            ent = softmax_entropy(probs)               # [1]
            eng = energy_score(logits, T=self.energy_temperature)  # [1]

            msp = float(msp.item())
            idx = int(idx.item())
            ent = float(ent.item())
            eng = float(eng.item())

        # OOD rejection (OR is conservative)
        is_ood = (
            (msp < self.msp_threshold)
            or (ent > self.entropy_threshold)
            or (eng > self.energy_threshold)
        )

        if is_ood:
            return ModelType.UNKNOWN, msp

        model_type = ModelType.PATHVQA if idx == 0 else ModelType.VQA_RAD
        return model_type, msp

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
        print(f"✓ Loaded adapter: {config.adapter_path}")

        return processor, model

    def predict(self, image_path: str, question: Optional[str] = None) -> dict:
        image = Image.open(image_path).convert("RGB")

        # Keep the classifier view as the original image (often better for routing)
        # Use a resized image for the VLM to keep consistent behavior
        image_resized = image.resize((384, 384), Image.Resampling.BILINEAR)

        # Route with OOD
        model_type, confidence = self.route(image)

        print(f"\n{'='*50}")
        if model_type == ModelType.UNKNOWN:
            print(f"ROUTING: UNKNOWN/OOD (MSP={confidence:.1%})")
        else:
            print(f"ROUTING: {model_type.value.upper()} (MSP={confidence:.1%})")
        print(f"{'='*50}")

        # Reject if OOD
        if model_type == ModelType.UNKNOWN:
            return {
                "question": question.strip() if question else "N/A",
                "answer": (
                    "Rejected (out-of-domain image). This system supports only pathology images "
                    "(histology/microscopy) and radiology scans (X-ray/CT/MRI). "
                    "Please upload a valid medical image."
                ),
                "model": "unknown",
                "confidence": confidence,
                "ood": True,
                "ood_reason": "classifier_ood_gate",
                "ood_rule": {
                    "msp_threshold": self.msp_threshold,
                    "entropy_threshold": self.entropy_threshold,
                    "energy_threshold": self.energy_threshold,
                    "energy_temperature": self.energy_temperature,
                },
            }

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

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        answer = processor.decode(
            output_ids[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        return {
            "question": final_question,
            "answer": answer,
            "model": model_type.value,
            "confidence": confidence,
            "ood": False,
        }


if __name__ == "__main__":
    # Your configs
    pathvqa_config = ModelConfig(
        base_model_id="Qwen/Qwen2-VL-7B-Instruct",
        adapter_path="../qwen2vl_7b_pathvqa_adapters",
        model_class=Qwen2VLForConditionalGeneration,
    )

    vqa_rad_config = ModelConfig(
        base_model_id="Qwen/Qwen3-VL-2B-Instruct",
        adapter_path="../qwen3vl_2b_vqa_rad_adapters",
        model_class=Qwen3VLForConditionalGeneration,
    )

    agent = ImageAgent(
        pathvqa_config=pathvqa_config,
        vqa_rad_config=vqa_rad_config,
        classifier_path="../modality_classifier",
        # Starting thresholds (tune!)
        msp_threshold=0.80,
        entropy_threshold=0.55,
        energy_threshold=-2.0,
        energy_temperature=1.0,
    )

    # Example
    result = agent.predict(
        "dataset_pathvqa/test/images/test_00001.jpg",
        "how are the histone subunits charged?"
    )
    print(f"Model: {result['model']} | Conf: {result['confidence']:.3f} | OOD: {result.get('ood')}")
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
