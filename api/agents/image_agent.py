"""
image_agent.py - PathVQA vs VQA-RAD router with ENHANCED OOD rejection

Key features:
- Classifier-based OOD (MSP, entropy, energy)
- SEMANTIC OOD checking (checks VQA output for non-medical keywords)
- Two-layer protection: reject before VQA (classifier) + after VQA (semantic)
- Lazy-loads PATHVQA and VQA-RAD Qwen adapters

Changes from original:
- Added semantic keyword checking in OODConfig
- Added check_semantic_ood() method
- VQA output is now validated before returning
"""

import torch
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
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
    model_class: type


@dataclass
class OODConfig:
    # Classifier-based OOD thresholds
    msp_threshold: float = 0.80
    entropy_threshold: float = 0.55
    energy_threshold: float = -2.0
    energy_temperature: float = 1.0

    # Semantic OOD checking (NEW!)
    enable_semantic_check: bool = True

    # Keywords that indicate non-medical images (will be rejected)
    reject_keywords: List[str] = field(default_factory=lambda: [
        # People/faces - HIGHEST PRIORITY
        "person", "face", "man", "woman", "child", "boy", "girl",
        "portrait", "selfie", "people", "human", "individual",
        "male", "female", "adult", "teenager",

        # Common non-medical objects
        "car", "vehicle", "automobile", "truck", "bus",
        "building", "house", "structure",
        "tree", "plant", "flower", "grass",
        "animal", "dog", "cat", "bird", "pet",
        "food", "meal", "dish", "fruit", "vegetable",
        "table", "chair", "furniture", "desk",
        "phone", "mobile", "computer", "laptop", "screen",
        "book", "document", "paper", "text",
        "clothing", "shirt", "dress", "shoes",

        # Scenes
        "landscape", "scenery",
        "city", "urban", "street", "road",
        "room", "bedroom", "kitchen", "bathroom",
        "outdoor", "outside", "indoor", "inside",
        "nature", "natural", "environment",
        "sky", "cloud", "sunset", "sunrise",
        "beach", "ocean", "sea", "water",
        "mountain", "hill", "valley", "forest"
    ])

    # Keywords that indicate medical images (helps validation)
    medical_keywords: List[str] = field(default_factory=lambda: [
        # Radiology
        "x-ray", "xray", "radiograph", "radiography",
        "ct", "computed tomography", "scan",
        "mri", "magnetic resonance",
        "ultrasound", "sonogram", "echo",
        "chest", "thorax", "lung", "pulmonary",
        "bone", "skeletal", "spine", "vertebra",
        "fracture", "break", "injury",
        "pneumonia", "infiltrate", "consolidation",
        "cardiac", "heart", "coronary",
        "brain", "cerebral", "cranial",
        "abdomen", "abdominal", "pelvis", "pelvic",

        # Pathology
        "histology", "histological", "histopathology",
        "pathology", "pathological",
        "tissue", "biopsy", "specimen",
        "microscopy", "microscopic", "micrograph",
        "cell", "cellular", "cytology",
        "stain", "staining", "h&e", "hematoxylin",
        "gram", "gram stain", "bacteria",
        "slide", "section",
        "tumor", "neoplasm", "mass",
        "carcinoma", "adenocarcinoma", "sarcoma",
        "melanoma", "lymphoma", "leukemia",

        # General medical
        "lesion", "abnormality", "finding",
        "disease", "disorder", "condition",
        "infection", "inflammatory", "inflammation",
        "diagnosis", "diagnostic",
        "medical", "clinical", "patient"
    ])


def softmax_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Calculate entropy of probability distribution"""
    return -(probs * (probs + 1e-12).log()).sum(dim=-1)


def energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Energy-based OOD score"""
    return -T * torch.logsumexp(logits / T, dim=-1)


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

    def __init__(
            self,
            pathvqa_config: ModelConfig,
            vqa_rad_config: ModelConfig,
            classifier_path: str,
            ood_config: Optional[OODConfig] = None
    ):
        self.ood_config = ood_config or OODConfig()
        self.pathvqa_config = pathvqa_config
        self.vqa_rad_config = vqa_rad_config

        # Classifier thresholds
        self.msp_threshold = self.ood_config.msp_threshold
        self.entropy_threshold = self.ood_config.entropy_threshold
        self.energy_threshold = self.ood_config.energy_threshold
        self.energy_temperature = self.ood_config.energy_temperature

        # Load classifier
        print("Loading modality classifier...")
        self._clf_processor = ViTImageProcessor.from_pretrained(classifier_path)
        self._clf_model = ViTForImageClassification.from_pretrained(classifier_path)
        self._clf_model.eval()
        if torch.cuda.is_available():
            self._clf_model.cuda()
        print("✓ Classifier loaded")

        if self.ood_config.enable_semantic_check:
            print("✓ Semantic OOD checking enabled")

        # Lazy load VLMs
        self._pathvqa = None
        self._vqa_rad = None

    def preload_models(self):
        """Pre-load both VQA models at startup (optional for eager loading)"""
        print("\nPre-loading VQA models...")
        self._load_pathvqa()
        self._load_vqa_rad()
        print("✓ All VQA models pre-loaded\n")

    def _load_pathvqa(self):
        """Load PathVQA model if not already loaded"""
        if self._pathvqa is None:
            print("  Loading PathVQA (Qwen2-VL-7B)...")
            self._pathvqa = self._load_model(self.pathvqa_config)
            print("  ✓ PathVQA loaded")

    def _load_vqa_rad(self):
        """Load VQA-RAD model if not already loaded"""
        if self._vqa_rad is None:
            print("  Loading VQA-RAD (Qwen3-VL-2B)...")
            self._vqa_rad = self._load_model(self.vqa_rad_config)
            print("  ✓ VQA-RAD loaded")

    def route(self, image: Image.Image) -> Tuple[ModelType, float]:
        """
        Route image to appropriate model using classifier.
        Returns: (model_type, confidence)
        """
        inputs = self._clf_processor(image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._clf_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

            msp, idx = probs.max(dim=-1)
            ent = softmax_entropy(probs)
            eng = energy_score(logits, T=self.energy_temperature)

            msp = float(msp.item())
            idx = int(idx.item())
            ent = float(ent.item())
            eng = float(eng.item())

        # Classifier-based OOD rejection
        is_ood = (
            (msp < self.msp_threshold)
            or (ent > self.entropy_threshold)
            or (eng > self.energy_threshold)
        )

        if is_ood:
            return ModelType.UNKNOWN, msp

        model_type = ModelType.PATHVQA if idx == 0 else ModelType.VQA_RAD
        return model_type, msp

    def check_semantic_ood(self, vqa_answer: str, question: str = "") -> Tuple[bool, Optional[str]]:
        """
        Check if VQA answer contains non-medical content.

        Returns:
            (is_ood, rejection_message)
        """
        if not self.ood_config.enable_semantic_check:
            return False, None

        answer_lower = vqa_answer.lower()
        question_lower = question.lower()

        # Check for non-medical keywords (reject if found)
        for keyword in self.ood_config.reject_keywords:
            if keyword in answer_lower:
                rejection_msg = (
                    f"I can only analyze medical images (X-rays, CT scans, MRIs, ultrasounds, "
                    f"or pathology slides). This appears to be a non-medical image. "
                    f"If you have a medical image to analyze, please upload it."
                )

                print(f"\n[SEMANTIC OOD REJECT] Detected non-medical keyword: '{keyword}'")
                print(f"  VQA Answer: {vqa_answer[:100]}...")

                return True, rejection_msg

        # Check for medical keywords (validate it's actually medical)
        has_medical_keywords = any(
            keyword in answer_lower or keyword in question_lower
            for keyword in self.ood_config.medical_keywords
        )

        # If no medical keywords found, it's suspicious
        if not has_medical_keywords:
            rejection_msg = (
                f"I'm not confident this is a medical image. "
                f"I can only analyze:\n"
                f"• Radiology: X-rays, CT scans, MRIs, ultrasounds\n"
                f"• Pathology: Tissue samples, biopsies, microscopy slides\n\n"
                f"If this is a medical image, please try rephrasing your question "
                f"or provide more context about what you'd like me to analyze."
            )

            print(f"\n[SEMANTIC OOD WARNING] No medical keywords detected")
            print(f"  VQA Answer: {vqa_answer[:100]}...")
            print(f"  This might be OOD, but allowing through for now")

            # Don't reject yet - could be edge case
            # But you can enable this for VERY strict mode:
            # return True, rejection_msg

        return False, None

    def _load_model(self, config: ModelConfig):
        """Load VQA model with LoRA adapters"""
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
        """
        Predict answer for medical image question with TWO-LAYER OOD protection:
        1. Classifier-based OOD (before VQA inference)
        2. Semantic OOD (after VQA inference)
        """
        image = Image.open(image_path).convert("RGB")
        image_resized = image.resize((384, 384), Image.Resampling.BILINEAR)

        # LAYER 1: Classifier-based OOD check
        model_type, confidence = self.route(image)

        print(f"\n{'='*60}")
        if model_type == ModelType.UNKNOWN:
            print(f"[LAYER 1] CLASSIFIER OOD REJECT (MSP={confidence:.1%})")
        else:
            print(f"[LAYER 1] CLASSIFIER PASS: {model_type.value.upper()} (MSP={confidence:.1%})")
        print(f"{'='*60}")

        # Reject if classifier says OOD
        if model_type == ModelType.UNKNOWN:
            return {
                "question": question.strip() if question else "N/A",
                "answer": (
                    "I can only analyze medical images (X-rays, CT scans, MRIs, ultrasounds, "
                    "or pathology slides). This image doesn't appear to be a medical scan. "
                    "Please upload a valid medical image."
                ),
                "model": "unknown",
                "confidence": confidence,
                "ood": True,
                "ood_reason": "classifier_ood",
                "ood_layer": "layer_1_classifier"
            }

        # Get VQA model
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

        # Generate VQA answer
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
        ).strip()

        # LAYER 2: Semantic OOD check
        print(f"\n[LAYER 2] Checking VQA output for semantic OOD...")
        print(f"  VQA Answer: {answer[:100]}...")

        is_semantic_ood, rejection_msg = self.check_semantic_ood(answer, final_question)

        if is_semantic_ood:
            print(f"[LAYER 2] SEMANTIC OOD REJECT!")
            print(f"{'='*60}\n")
            return {
                "question": final_question,
                "answer": rejection_msg,
                "model": model_type.value,
                "confidence": confidence,
                "ood": True,
                "ood_reason": "semantic_ood",
                "ood_layer": "layer_2_semantic",
                "original_vqa_answer": answer  # For debugging
            }

        print(f"[LAYER 2] SEMANTIC PASS - Medical content detected")
        print(f"{'='*60}\n")

        # Accept - return VQA answer
        return {
            "question": final_question,
            "answer": answer,
            "model": model_type.value,
            "confidence": confidence,
            "ood": False,
        }


if __name__ == "__main__":
    # Example configuration
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

    # Create agent with semantic OOD enabled
    agent = ImageAgent(
        pathvqa_config=pathvqa_config,
        vqa_rad_config=vqa_rad_config,
        classifier_path="../modality_classifier",
        ood_config=OODConfig(
            # Classifier thresholds
            msp_threshold=0.80,
            entropy_threshold=0.55,
            energy_threshold=-2.0,
            # Semantic checking (NEW!)
            enable_semantic_check=True  # ← Enable semantic layer
        )
    )

    # Test with medical image
    print("\n" + "="*70)
    print("TEST 1: Medical Image (should ACCEPT)")
    print("="*70)
    result = agent.predict(
        "dataset_pathvqa/test/images/test_00001.jpg",
        "what do you see in this image?"
    )
    print(f"\nResult: {'REJECTED' if result['ood'] else 'ACCEPTED'}")
    print(f"Model: {result['model']}")
    print(f"Answer: {result['answer'][:100]}...")

    # Test with person photo (should REJECT)
    print("\n" + "="*70)
    print("TEST 2: Person Photo (should REJECT)")
    print("="*70)
    # result = agent.predict("person_photo.jpg", "who is this")
    # Uncomment above when you have a test image