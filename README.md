# Agentic Multimodal Medical QA

## What it does

Users can upload a medical image (X-ray, pathology slide) and ask questions about it in natural language. The system analyzes the image, searches PubMed for relevant literature, and returns a response with citations.

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
| `ConversationSummarizer` | Compresses older turns into a rolling summary when memory grows too large |
| `MemoryManager` | Holds active conversation history in RAM and tracks summary state |
| `SessionManager` | Persists sessions and images to disk |

A typical flow: translate input → load memory → route → run VQA → search PubMed → generate response → translate output → save session.

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

Fine-tuned on three medical VQA datasets: PathVQA, and SLAKE. The deployed system uses only the PathVQA and SLAKE adapters. Each dataset has its own numbered pipeline (`1_download → 2_preprocess → 3_train → 4_evaluate`).

---

## API

Runs on FastAPI at port 8000. Main endpoints:

```
POST /auth/register
POST /auth/login
POST /chat/new              # start session, accepts image + question
POST /chat/{id}/message     # continue conversation
GET  /chat/history
GET  /chat/{id}/memory-status
GET  /chat/{id}/summary
POST /chat/{id}/summarize
```

All chat endpoints require a JWT token in the `Authorization` header.

The `/memory-status` and `/summary` endpoints expose the rolling-summary state for long sessions. `/summarize` can be used to force manual compaction for testing or debugging, while normal chat requests can also trigger compaction automatically.

---

## Frontend

The frontend for this project is available here:

[medical-vqa-frontend](https://github.com/senghuyjr11/medical-vqa-frontend.git)

---

## Environment

```
GOOGLE_API_KEY=...      # Gemma (routing + generation + embeddings)
NCBI_EMAIL=...
NCBI_API_KEY=...
JWT_SECRET_KEY=...
APP_ENV=dev             # use 'prod' on server
```

---

## Run

```bash
cd api
uvicorn api_refactored:app --host 0.0.0.0 --port 8000
```
