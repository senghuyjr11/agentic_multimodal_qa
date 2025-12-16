https://huggingface.co/datasets/flaviagiammarino/path-vqa
https://huggingface.co/datasets/flaviagiammarino/vqa-rad

Main Pipeline Flow:
User Input (Image + Question / Image Only / Text Only)
                    │
                    ▼
            ┌───────────────┐
            │   Main.py     │
            │   (Router)    │
            └───────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   Has Image?               Text Only?
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  Image Agent  │       │ Text-Only     │
│  (ViT + VQA)  │       │ Agent         │
└───────────────┘       └───────────────┘
        │                       │
        │               ┌───────┴───────┐
        │               ▼               ▼
        │           Casual?         Medical?
        │               │               │
        │               ▼               ▼
        │         Gemma 4B/12B    Gemma 4B/12B
        │               │               │
        │               ▼               │
        │         Final Response        │
        │            (End)              │
        │                               │
        ▼                               ▼
┌───────────────────────────────────────────┐
│              PubMed Agent                 │
│         (Knowledge Augmentation)          │
└───────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│            Reasoning Agent                │
│      (Gemma 4B/12B + Multi-language)      │
└───────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│           Session Manager                 │
│         (Save to JSON + Image)            │
└───────────────────────────────────────────┘
                    │
                    ▼
              Final Output

Image Agent Detail:
Image Input
      │
      ▼
┌─────────────┐
│ViT Classifier│
│ (Modality)   │
└─────────────┘
      │
      ├──── Pathology ────▶ PathVQA Model (Qwen2-VL-7B)
      │
      └──── Radiology ────▶ VQA-RAD Model (Qwen3-VL-2B)
                                   │
                                   ▼
                              VQA Answer

Text-Only Agent Detail:
Text Input
      │
      ▼
┌──────────────┐
│Keyword Check │
└──────────────┘
      │
      ├──── Casual ────▶ Gemma 4B/12B ────▶ Simple Response (End)
      │
      └──── Medical ───▶ Gemma 4B/12B ────▶ Initial Response
                                                   │
                                                   ▼
                                            PubMed Agent
                                                   │
                                                   ▼
                                           Reasoning Agent
                                                   │
                                                   ▼
                                           Final Response

Reasoning Agent Detail:
Inputs:
├── Question
├── VQA Answer (or "Text-only")
└── PubMed Articles
          │
          ▼
    ┌───────────┐
    │ Language? │
    └───────────┘
          │
          ├──── English ────▶ Gemma 4B (Fast)
          │
          └──── Other ──────▶ Gemma 12B (Multilingual)
                                   │
                                   ▼
                          Enhanced Response
                          ├── Answer
                          ├── Simple Explanation
                          ├── Clinical Context
                          └── Summary

Session Storage Structure:
sessions/
└── {username}/
    └── {session_id}/
        ├── input_image.jpg
        └── session_data.json
              │
              ├── input (question, image_path)
              ├── image_agent (routed_to, confidence)
              ├── vqa_agent (question, answer)
              ├── pubmed_agent (query, articles)
              └── reasoning_agent (language, response)

Files Structure:
api/
├── image_agent.py        → ViT + PathVQA/VQA-RAD
├── text_only_agent.py    → Casual/Medical routing
├── pubmed_agent.py       → PubMed knowledge
├── reasoning_agent.py    → Final explanation
├── session_manager.py    → Storage
├── main.py               → Pipeline orchestration
└── sessions/             → User data

Summary Table:
┌──────────────────┬──────────────────┬─────────────────────────────┐
│ Agent            │ Model            │ Purpose                     │
├──────────────────┼──────────────────┼─────────────────────────────┤
│ Image Agent      │ ViT + Qwen2/3-VL │ Route & predict from image  │
│ Text-Only Agent  │ Gemma 4B/12B     │ Handle text-only questions  │
│ PubMed Agent     │ API (no model)   │ Fetch medical literature    │
│ Reasoning Agent  │ Gemma 4B/12B     │ Explain + translate         │
│ Session Manager  │ None             │ Store outputs               │
└──────────────────┴──────────────────┴─────────────────────────────┘