import gc
import os
import time
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
            "Analyze this pathology image and answer in 1 to 2 complete sentences. "
            "State the most likely pathological abnormality or disease pattern. "
            "If uncertain or no clear abnormality is visible, say that clearly. "
            "Briefly mention the key microscopic features supporting the answer."
        ),
        ModelType.VQA_RAD: (
            "Analyze this medical image and answer in 1 to 2 complete sentences. "
            "State the most likely abnormality or disease. "
            "If the scan appears normal or the finding is uncertain, say that clearly. "
            "Briefly mention the main imaging finding and location supporting the answer."
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
        self._models_preloaded = False
        self.max_new_tokens = int(os.getenv("APP_VQA_MAX_NEW_TOKENS", "64"))
        self.image_size = int(os.getenv("APP_VQA_IMAGE_SIZE", "336"))
        self.min_pixels = int(os.getenv("APP_VQA_MIN_PIXELS", str(64 * 28 * 28)))
        self.max_pixels = int(os.getenv("APP_VQA_MAX_PIXELS", str(144 * 28 * 28)))
        self.attn_implementation = os.getenv("APP_VQA_ATTN_IMPL", "sdpa").strip() or None

    @property
    def models_preloaded(self) -> bool:
        return self._models_preloaded and self._processor is not None and self._model is not None

    def preload_models(self):
        if self.models_preloaded:
            print("✓ VQA models already preloaded; skipping reload")
            return

        print("\nLoading base model (Qwen3-VL-8B) once...")
        print(
            f"  Inference config: max_new_tokens={self.max_new_tokens}, "
            f"image_size={self.image_size}, "
            f"min_pixels={self.min_pixels}, max_pixels={self.max_pixels}, "
            f"attn={self.attn_implementation or 'default'}, compute_dtype=float16"
        )
        base_model_id = self.pathvqa_config.base_model_id

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self._processor = AutoProcessor.from_pretrained(
            base_model_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        model_kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            ),
            "device_map": {"": 0},
        }
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_id,
            **model_kwargs,
        )

        print("  Loading PathVQA adapter...")
        self._model = PeftModel.from_pretrained(
            base_model,
            self.pathvqa_config.adapter_path,
            adapter_name="pathvqa",
        )

        print("  Loading VQA-RAD adapter...")
        self._model.load_adapter(
            self.vqa_rad_config.adapter_path,
            adapter_name="vqa_rad",
        )

        self._model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        print("✓ Base model + both adapters loaded")

        self._warmup()
        self._models_preloaded = True
        print("✓ GPU warmed up — ready\n")

    def _warmup(self):
        print("  Warming up GPU (compiling CUDA kernels)...")
        dummy_image = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))
        dummy_text  = self._processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "test"}]}],
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[dummy_text],
            images=[dummy_image],
            return_tensors="pt",
        ).to(self._model.device)

        for adapter in ("pathvqa", "vqa_rad"):
            self._model.set_adapter(adapter)
            with torch.inference_mode():
                self._model.generate(**inputs, max_new_tokens=5, do_sample=False)

        torch.cuda.empty_cache()

    def route(self, image: Image.Image) -> Tuple[ModelType, float]:
        inputs = self._clf_processor(image, return_tensors="pt")
        # ViT stays on CPU intentionally

        with torch.inference_mode():
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
        if not self.models_preloaded:
            raise RuntimeError(
                "VQA models are not preloaded. Start the API through api_refactored:app "
                "and wait for the startup preload to finish before sending image requests."
            )

        total_start = time.perf_counter()

        load_start = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        image_resized = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        load_seconds = time.perf_counter() - load_start

        classify_start = time.perf_counter()
        model_type, confidence = self.route(image)
        classify_seconds = time.perf_counter() - classify_start

        print(f"\n{'='*60}")
        print(f"[CLASSIFIER] {model_type.value.upper()} (confidence={confidence:.1%})")
        print(f"{'='*60}")

        if model_type == ModelType.REJECT:
            print("[REJECT] Non-medical image detected")
            total_seconds = time.perf_counter() - total_start
            return {
                "question": question or "N/A",
                "answer": self.REJECTION_MESSAGE,
                "model": "reject",
                "confidence": confidence,
                "ood": True,
                "timing": {
                    "total_seconds": round(total_seconds, 4),
                },
            }

        # Switch adapter — no model reload needed
        adapter_name = "pathvqa" if model_type == ModelType.PATHVQA else "vqa_rad"
        self._model.set_adapter(adapter_name)
        print(f"[ADAPTER] Switched to {adapter_name}")

        if question and question.strip():
            final_question = (
                f"{question.strip()} "
                "Answer briefly in 1 to 2 complete sentences."
            )
            prompt_source = "user_question"
        else:
            final_question = self.DEFAULT_QUESTIONS[model_type]
            prompt_source = f"default_{model_type.value}"

        print(f"[PROMPT] Source: {prompt_source}")
        print(f"[PROMPT] Text: {final_question}")

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": final_question}]}
        ]
        text_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)

        preprocess_start = time.perf_counter()
        inputs = self._processor(
            text=[text_prompt],
            images=[image_resized],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        preprocess_seconds = time.perf_counter() - preprocess_start

        generate_start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                pad_token_id=self._processor.tokenizer.pad_token_id,
            )
        generate_seconds = time.perf_counter() - generate_start

        answer = self._processor.decode(
            output_ids[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()

        print(f"[VQA] Answer: {answer[:100]}...")
        total_seconds = time.perf_counter() - total_start
        print(
            f"[TIMING] load_image={load_seconds:.2f}s "
            f"classify={classify_seconds:.2f}s "
            f"preprocess={preprocess_seconds:.2f}s "
            f"generate={generate_seconds:.2f}s "
            f"total={total_seconds:.2f}s "
            f"max_new_tokens={self.max_new_tokens} "
            f"image_size={self.image_size}"
        )

        return {
            "question": final_question,
            "answer": answer,
            "model": model_type.value,
            "confidence": confidence,
            "ood": False,
            "timing": {
                "total_seconds": round(total_seconds, 4),
            },
        }


if __name__ == "__main__":
    pathvqa_config = ModelConfig(
        base_model_id="Qwen/Qwen3-VL-8B-Instruct",
        adapter_path="../pathvqa_qwen3vl_pipeline/adapters",
        model_class=Qwen3VLForConditionalGeneration,
    )

    vqa_rad_config = ModelConfig(
        base_model_id="Qwen/Qwen3-VL-8B-Instruct",
        adapter_path="../vqa_rad_qwen3vl_pipeline/adapters",
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
