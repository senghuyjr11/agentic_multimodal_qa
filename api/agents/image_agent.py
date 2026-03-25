import gc
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
            "Analyze this pathology image carefully. Identify the most likely pathological "
            "abnormality or disease pattern visible in the tissue. If the finding is uncertain "
            "or no clear abnormality is visible, say so clearly. Describe the key microscopic "
            "features and tissue region that support your conclusion."
        ),
        ModelType.VQA_RAD: (
            "Analyze this medical scan carefully. Identify the most likely abnormality or disease "
            "visible in the image. If the scan appears normal or the finding is uncertain, state "
            "that clearly. Describe the main imaging findings and the anatomical location that "
            "support your conclusion."
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
        classifier_path: str = "../modality_classifier_pipeline/model",
    ):
        self.pathvqa_config = pathvqa_config
        self.vqa_rad_config = vqa_rad_config

        # ViT runs on CPU — keeps full VRAM free for Qwen
        print("Loading modality classifier (CPU)...")
        self._clf_processor = ViTImageProcessor.from_pretrained(classifier_path)
        self._clf_model = ViTForImageClassification.from_pretrained(classifier_path)
        self._clf_model.eval()
        print("✓ Classifier loaded (CPU)")

        self._processor = None
        self._model     = None

    def preload_models(self):
        print("\nLoading base model (Qwen3-VL-8B) once...")
        base_model_id = self.pathvqa_config.base_model_id

        self._processor = AutoProcessor.from_pretrained(base_model_id)

        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map={"": 0},
        )

        print("  Loading PathVQA adapter...")
        self._model = PeftModel.from_pretrained(
            base_model,
            self.pathvqa_config.adapter_path,
            adapter_name="pathvqa",
        )

        print("  Loading SLAKE adapter...")
        self._model.load_adapter(
            self.vqa_rad_config.adapter_path,
            adapter_name="slake",
        )

        self._model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        print("✓ Base model + both adapters loaded")

        self._warmup()
        print("✓ GPU warmed up — ready\n")

    def _warmup(self):
        print("  Warming up GPU (compiling CUDA kernels)...")
        dummy_image = Image.new("RGB", (384, 384), color=(128, 128, 128))
        dummy_text  = self._processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "test"}]}],
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[dummy_text],
            images=[dummy_image],
            return_tensors="pt",
        ).to(self._model.device)

        for adapter in ("pathvqa", "slake"):
            self._model.set_adapter(adapter)
            with torch.no_grad():
                self._model.generate(**inputs, max_new_tokens=5, do_sample=False)

        torch.cuda.empty_cache()

    def route(self, image: Image.Image) -> Tuple[ModelType, float]:
        inputs = self._clf_processor(image, return_tensors="pt")
        # ViT stays on CPU intentionally

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

    def predict(self, image_path: str, question: Optional[str] = None) -> dict:
        image = Image.open(image_path).convert("RGB")
        image_resized = image.resize((384, 384), Image.Resampling.BILINEAR)

        model_type, confidence = self.route(image)

        print(f"\n{'='*60}")
        print(f"[CLASSIFIER] {model_type.value.upper()} (confidence={confidence:.1%})")
        print(f"{'='*60}")

        if model_type == ModelType.REJECT:
            print("[REJECT] Non-medical image detected")
            return {
                "question": question or "N/A",
                "answer": self.REJECTION_MESSAGE,
                "model": "reject",
                "confidence": confidence,
                "ood": True,
            }

        # Switch adapter — no model reload needed
        adapter_name = "pathvqa" if model_type == ModelType.PATHVQA else "slake"
        self._model.set_adapter(adapter_name)
        print(f"[ADAPTER] Switched to {adapter_name}")

        torch.cuda.empty_cache()

        final_question = question.strip() if question else self.DEFAULT_QUESTIONS[model_type]
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": final_question}]}
        ]
        text_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self._processor(
            text=[text_prompt],
            images=[image_resized],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                pad_token_id=self._processor.tokenizer.pad_token_id,
            )

        answer = self._processor.decode(
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
        classifier_path="../modality_classifier_pipeline/model",
    )
    agent.preload_models()

    result = agent.predict(
        "dataset_pathvqa/test/images/test_00001.jpg",
        "what do you see in this image?"
    )
    print(f"\nResult : {'REJECTED' if result['ood'] else 'ACCEPTED'}")
    print(f"Model  : {result['model']}")
    print(f"Answer : {result['answer'][:100]}...")
