# Agentic Multimodal Medical QA

A medical VQA system that demonstrates how agentic AI works with multimodal inputs. The main idea is that instead of one monolithic model, the system routes each user request through a chain of specialized agents — each handling one job — to produce a grounded, context-aware answer.

---

## What it does

Users can upload a medical image (X-ray, pathology slide) and ask questions about it in natural language. The system analyzes the image, searches PubMed for relevant literature, and returns a response with citations. It supports multi-turn conversations with memory, JWT authentication, and 18 languages.

---

## Agentic Pipeline

All agents live in `api/agents/`. The orchestrator in `api/main_simple.py` runs them in sequence per request.

| Agent | Job |
|---|---|
| `TranslationAgent` | Detects language, translates input to English, translates output back |
| `RouterAgent` | Decides which agents to call and what mode to use (Gemma-3-4B-IT) |
| `ImageAgent` | Classifies image modality, runs the right fine-tuned VQA model |
| `PubMedAgent` | Searches NCBI PubMed, scores articles by relevance using embeddings |
| `ResponseGenerator` | Synthesizes VQA answer + literature into a final response |
| `MemoryManager` | Holds conversation history in RAM for follow-up context |
| `SessionManager` | Persists sessions and images to disk |

A typical flow: translate input → load memory → route → run VQA → search PubMed → generate response → translate output → save session.

The `RouterAgent` also handles follow-up detection, so if you ask "explain that in simpler terms" after a medical answer, it skips VQA and PubMed and just rewrites the previous response.

---

## Models

- **Pathology (PathVQA)**: Qwen3-VL-8B-Instruct + DoRA adapters (r=64, attention + MLP)
- **Radiology (SLAKE)**: Qwen3-VL-8B-Instruct + DoRA adapters (r=32, attention only)
- **Routing**: Gemma-3-4B-IT via Gemini API
- **Translation**: facebook/nllb-200-distilled-1.3B (runs on CPU)
- **Image classifier**: ViT (google/vit-base-patch16-224), trained on PathVQA + SLAKE + ImageNet-Mini reject class

Both VQA models share the same base model loaded once at startup. Adapters are swapped on demand via PEFT named adapters — no model reload between requests.

---

## Dataset & Training

Fine-tuned on three medical VQA datasets: PathVQA, SLAKE, and VQA-RAD. The deployed system uses only the PathVQA and SLAKE adapters. Each dataset has its own numbered pipeline (`1_download → 2_preprocess → 3_train → 4_evaluate`).

---

## API

Runs on FastAPI at port 8000. Main endpoints:

```
POST /auth/register
POST /auth/login
POST /chat/new              # start session, accepts image + question
POST /chat/{id}/message     # continue conversation
GET  /chat/history
```

All chat endpoints require a JWT token in the `Authorization` header.

---

## Environment

```
GOOGLE_API_KEY=...      # Gemini (routing + generation + embeddings)
NCBI_EMAIL=...
NCBI_API_KEY=...
JWT_SECRET_KEY=...
```

---

## Run

```bash
cd api
uvicorn api_refactored:app --host 0.0.0.0 --port 8000
```
