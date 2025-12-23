import os
import tempfile
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from transformers import Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration

from image_agent import ModelConfig
from main import MedicalVQAPipeline
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
REQUIRED_KEYS = ["GOOGLE_API_KEY", "NCBI_EMAIL", "NCBI_API_KEY"]

missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Missing environment variables: {missing}")

# ============== APP SETUP ==============

app = FastAPI(
    title="Medical VQA API",
    description="Conversational Medical Visual Question Answering System with Multi-turn Memory (Clean Turn Schema)",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[MedicalVQAPipeline] = None


def _save_upload_to_tempfile(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(upload.file.read())
        return f.name


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline

    print("=" * 60)
    print("INITIALIZING CONVERSATIONAL MEDICAL VQA API (CLEAN TURN SCHEMA)")
    print("=" * 60)

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

    pipeline = MedicalVQAPipeline(
        ncbi_email=os.getenv("NCBI_EMAIL"),
        ncbi_api_key=os.getenv("NCBI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        pathvqa_config=pathvqa_config,
        vqa_rad_config=vqa_rad_config,
        classifier_path="../modality_classifier"
    )

    # Preload VQA models (optional but recommended)
    print("\n[Preloading VQA Models]")
    print("Loading PathVQA model...")
    pipeline.image_agent._pathvqa = pipeline.image_agent._load_model(pathvqa_config)
    print("✓ PathVQA loaded")

    print("Loading VQA-RAD model...")
    pipeline.image_agent._vqa_rad = pipeline.image_agent._load_model(vqa_rad_config)
    print("✓ VQA-RAD loaded")

    print("\n" + "=" * 60)
    print("ALL MODELS READY - CONVERSATIONAL MODE ENABLED (CLEAN TURN SCHEMA)")
    print("=" * 60)


# ============== HEALTH ==============

@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "features": ["multi-turn conversations", "auto language detection", "memory persistence", "per-turn citations/meta"],
        "schema": "clean_turn_meta"
    }


# ============== CHAT ENDPOINTS ==============

@app.post("/chat/new")
async def start_new_chat(
    username: str = Form(...),
    question: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Start a NEW conversation.

    - username: required
    - question: optional
    - image: optional file upload

    Returns:
    - session_id
    - turn
    - response
    - output_language (for UI)
    - full_session (clean schema)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not image and not question:
        raise HTTPException(status_code=400, detail="Must provide image or question")

    image_path = None
    try:
        if image:
            suffix = os.path.splitext(image.filename)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                content = await image.read()
                f.write(content)
                image_path = f.name

        result = pipeline.run(
            username=username,
            question=question,
            image_path=image_path
        )

        session_data = pipeline.session_manager.load(username, result["session_id"])

        return {
            "message": "New conversation started",
            "session_id": result["session_id"],
            "turn": result["turn"],
            "response": result["enhanced_response"],
            "output_language": result.get("output_language", result.get("original_language", "English")),
            "source_language": result.get("source_language", "English"),
            "full_session": session_data
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)


@app.post("/chat/{session_id}/message")
async def send_message(
    session_id: int,
    username: str = Form(...),
    question: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Continue an EXISTING conversation.

    - session_id: required
    - username: required
    - question: optional
    - image: optional upload

    Returns:
    - session_id
    - turn
    - response
    - output_language
    - full_session
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not image and not question:
        raise HTTPException(status_code=400, detail="Must provide image or question")

    if not pipeline.session_manager.session_exists(username, session_id):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found for user {username}")

    image_path = None
    try:
        if image:
            suffix = os.path.splitext(image.filename)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                content = await image.read()
                f.write(content)
                image_path = f.name

        result = pipeline.run(
            username=username,
            question=question,
            image_path=image_path,
            session_id=session_id
        )

        session_data = pipeline.session_manager.load(username, result["session_id"])

        return {
            "message": "Message sent",
            "session_id": result["session_id"],
            "turn": result["turn"],
            "response": result["enhanced_response"],
            "output_language": result.get("output_language", result.get("original_language", "English")),
            "source_language": result.get("source_language", "English"),
            "full_session": session_data
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)


# ============== HISTORY ENDPOINTS ==============

@app.get("/chat/history/{username}")
async def list_user_chats(username: str):
    """List all chat sessions for a user."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    session_ids = pipeline.session_manager.list_user_sessions(username)

    chats = []
    for sid in session_ids:
        data = pipeline.session_manager.load(username, sid)

        first_question = data.get("input", {}).get("question", "[Image uploaded]")
        turns_count = len(data.get("conversation_history", []))

        # In clean schema, language should be read from latest turn meta.translation if available
        history = data.get("conversation_history", [])
        if history:
            last_meta = history[-1].get("meta", {})
            lang = (last_meta.get("translation", {}) or {}).get("output_language", "English")
        else:
            lang = "English"

        chats.append({
            "session_id": sid,
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at", data.get("created_at")),
            "first_message": first_question[:50] + "..." if len(first_question) > 50 else first_question,
            "turns": turns_count,
            "language": lang
        })

    return {
        "username": username,
        "total_chats": len(chats),
        "chats": chats
    }


@app.get("/chat/{session_id}")
async def get_chat_session(username: str, session_id: int):
    """Get full conversation history for a session."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not pipeline.session_manager.session_exists(username, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return pipeline.session_manager.load(username, session_id)


@app.delete("/chat/{session_id}")
async def delete_chat_session(username: str, session_id: int):
    """Delete a chat session."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not pipeline.session_manager.session_exists(username, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    import shutil
    session_dir = pipeline.session_manager._get_session_dir(username, session_id)
    shutil.rmtree(session_dir)

    # Remove from active RAM memory if present
    if session_id in pipeline.active_conversations:
        del pipeline.active_conversations[session_id]

    return {"message": f"Session {session_id} deleted"}


@app.get("/users")
async def list_users():
    """List all users."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return pipeline.session_manager.list_users()


@app.get("/debug/memory/{session_id}")
async def debug_memory(session_id: int, username: str):
    """
    Debug endpoint: Show what's in LangChain RAM for a session.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not pipeline.session_manager.session_exists(username, session_id):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found for user {username}")

    if session_id not in pipeline.active_conversations:
        return {
            "session_id": session_id,
            "username": username,
            "in_ram": False,
            "message": "Session not in RAM cache (will be loaded from JSON on next message)"
        }

    memory = pipeline.active_conversations[session_id]
    messages = memory.messages

    debug_info = {
        "session_id": session_id,
        "username": username,
        "in_ram": True,
        "total_messages": len(messages),
        "turns": len(messages) // 2,
        "messages": []
    }

    for i, msg in enumerate(messages):
        debug_info["messages"].append({
            "index": i,
            "type": "human" if isinstance(msg, HumanMessage) else "ai",
            "content": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        })

    return debug_info


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
