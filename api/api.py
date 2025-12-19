"""
api.py - Conversational API with LangChain Memory
"""
import os
import tempfile
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration

from image_agent import ModelConfig
from main import MedicalVQAPipeline

# ============== APP SETUP ==============

app = FastAPI(
    title="Medical VQA API",
    description="Conversational Medical Visual Question Answering System with Multi-turn Memory",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline

    print("=" * 60)
    print("INITIALIZING CONVERSATIONAL MEDICAL VQA API")
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
        ncbi_email="senghuymit007@gmail.com",
        ncbi_api_key="92da6b3a9eb8f5916e252e7fbc9d9aed3609",
        google_api_key="GOOGLE_API_KEY_REMOVED",
        pathvqa_config=pathvqa_config,
        vqa_rad_config=vqa_rad_config,
        classifier_path="../modality_classifier"
    )

    # Preload all VQA models
    print("\n[Preloading VQA Models]")
    print("Loading PathVQA model...")
    pipeline.image_agent._pathvqa = pipeline.image_agent._load_model(pathvqa_config)
    print("âœ“ PathVQA loaded")

    print("Loading VQA-RAD model...")
    pipeline.image_agent._vqa_rad = pipeline.image_agent._load_model(vqa_rad_config)
    print("âœ“ VQA-RAD loaded")

    print("\n" + "=" * 60)
    print("ALL MODELS READY - CONVERSATIONAL MODE ENABLED")
    print("=" * 60)


# ============== CHAT ENDPOINTS ==============

@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "features": ["multi-turn conversations", "auto language detection", "memory persistence"]
    }


@app.post("/chat/new")
async def start_new_chat(
    username: str = Form(...),
    question: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Start a NEW conversation.

    - username: required
    - question: optional (any language - auto-detected)
    - image: optional file upload

    Returns session_id for continuing the conversation.
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

        # Start new conversation (no session_id)
        result = pipeline.run(
            username=username,
            question=question,
            image_path=image_path
        )

        # Load full session data
        session_data = pipeline.session_manager.load(username, result["session_id"])

        return {
            "message": "New conversation started",
            "session_id": result["session_id"],
            "turn": result["turn"],
            "response": result["enhanced_response"],
            "original_language": result["original_language"],
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

    Automatically handles:
    - If memory is in RAM â†’ uses it (instant)
    - If memory not in RAM â†’ loads from JSON (resume after app restart)

    - session_id: required (from /chat/new response)
    - username: required
    - question: optional (any language - auto-detected)
    - image: optional file upload
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not image and not question:
        raise HTTPException(status_code=400, detail="Must provide image or question")

    # Check if session exists
    if not pipeline.session_manager.session_exists(username, session_id):
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found for user {username}"
        )

    image_path = None

    try:
        if image:
            suffix = os.path.splitext(image.filename)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                content = await image.read()
                f.write(content)
                image_path = f.name

        # Continue conversation (with session_id)
        result = pipeline.run(
            username=username,
            question=question,
            image_path=image_path,
            session_id=session_id  # This triggers memory loading if needed
        )

        # Load full session data
        session_data = pipeline.session_manager.load(username, result["session_id"])

        return {
            "message": "Message sent",
            "session_id": result["session_id"],
            "turn": result["turn"],
            "response": result["enhanced_response"],
            "original_language": result["original_language"],
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

        # Get first message as preview
        first_question = data["input"].get("question", "[Image uploaded]")
        turns_count = len(data.get("conversation_history", []))

        chats.append({
            "session_id": sid,
            "created_at": data["created_at"],
            "updated_at": data.get("updated_at", data["created_at"]),
            "first_message": first_question[:50] + "..." if len(first_question) > 50 else first_question,
            "turns": turns_count,
            "language": data.get("translation", {}).get("original_language", "English")
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

    # Remove from active memory if present
    if session_id in pipeline.active_conversations:
        del pipeline.active_conversations[session_id]

    return {"message": f"Session {session_id} deleted"}


@app.get("/users")
async def list_users():
    """List all users."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return pipeline.session_manager.list_users()


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)