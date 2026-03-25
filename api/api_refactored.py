"""
api_refactored.py - Complete FastAPI with all endpoints + authentication

Features:
- JWT Authentication
- New chat / Continue chat endpoints
- Full conversation history
- Memory debugging
- Model preloading
- Clean agent structure
"""

import os
import shutil
from pathlib import Path
from typing import Optional
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from runtime_config import (
    CLASSIFIER_MODEL_DIR,
    PATHVQA_ADAPTER_DIR,
    SLAKE_ADAPTER_DIR,
    SESSIONS_DIR,
    TEMP_UPLOADS_DIR,
    configure_runtime_environment,
)

configure_runtime_environment()

from transformers import Qwen3VLForConditionalGeneration

from agents.image_agent import ModelConfig
from main_simple import MedicalVQAPipeline
from auth import router as auth_router, get_current_user

# Load environment variables
load_dotenv()

# Validate required keys
REQUIRED_KEYS = ["GOOGLE_API_KEY", "NCBI_EMAIL", "NCBI_API_KEY"]
missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Missing environment variables: {missing}")

# ============== APP SETUP ==============

app = FastAPI(
    title="Medical VQA API - Clean Structure",
    description="Conversational Medical Visual Question Answering with Multi-turn Memory",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/sessions", StaticFiles(directory=str(SESSIONS_DIR)), name="sessions")

# Include authentication router
app.include_router(auth_router)

# Global pipeline
pipeline: Optional[MedicalVQAPipeline] = None


# ============== STARTUP ==============

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline and preload models on startup."""
    global pipeline

    print("\n" + "=" * 70)
    print("INITIALIZING MEDICAL VQA API (CLEAN AGENT STRUCTURE)")
    print("=" * 70)

    # Model configurations
    pathvqa_config = ModelConfig(
        base_model_id="Qwen/Qwen3-VL-8B-Instruct",
        adapter_path=str(PATHVQA_ADAPTER_DIR),
        model_class=Qwen3VLForConditionalGeneration
    )

    vqa_rad_config = ModelConfig(
        base_model_id="Qwen/Qwen3-VL-8B-Instruct",
        adapter_path=str(SLAKE_ADAPTER_DIR),
        model_class=Qwen3VLForConditionalGeneration
    )

    # Initialize pipeline
    pipeline = MedicalVQAPipeline(
        ncbi_email=os.getenv("NCBI_EMAIL"),
        ncbi_api_key=os.getenv("NCBI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        pathvqa_config=pathvqa_config,
        vqa_rad_config=vqa_rad_config,
        classifier_path=str(CLASSIFIER_MODEL_DIR)
    )

    # Preload VQA models
    print("\n" + "=" * 70)
    print("PRE-LOADING VQA MODELS...")
    print("=" * 70)

    pipeline.image_agent.preload_models()

    print("=" * 70)
    print("✓ ALL MODELS LOADED - API READY!")
    print("=" * 70 + "\n")


# ============== HEALTH CHECK ==============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "structure": "clean_agents",
        "models_preloaded": True,
        "features": [
            "JWT Authentication",
            "Multi-turn Conversations",
            "Auto Language Detection",
            "Memory Persistence",
            "LLM-driven Routing"
        ],
        "agents": {
            "router": "✓",
            "response_generator": "✓",
            "memory_manager": "✓",
            "image_agent": "✓",
            "pubmed_agent": "✓",
            "translation_agent": "✓",
            "session_manager": "✓"
        }
    }


# ============== CHAT ENDPOINTS ==============

@app.post("/chat/new")
async def start_new_chat(
    current_user: str = Depends(get_current_user),
    question: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Start a NEW conversation (Protected - requires authentication).

    Returns:
    - session_id
    - response
    - metadata
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not image and not question:
        raise HTTPException(status_code=400, detail="Must provide image or question")

    image_path = None
    try:
        # Handle image upload
        if image:
            suffix = os.path.splitext(image.filename)[1] or ".jpg"
            TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            image_path = TEMP_UPLOADS_DIR / f"{uuid4().hex}{suffix}"
            content = await image.read()
            image_path.write_bytes(content)
            image_path = str(image_path)

        # Run pipeline
        result = pipeline.run(
            username=current_user,
            question=question or "What do you see in this image?",
            image_path=image_path,
            session_id=None  # New conversation
        )

        # Get full session data
        session_data = pipeline.session_mgr.load(current_user, result["session_id"])

        return {
            "message": "New conversation started",
            "session_id": result["session_id"],
            "response": result["response"],
            "metadata": result.get("metadata", {}),
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
    current_user: str = Depends(get_current_user),
    question: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Continue an EXISTING conversation (Protected - requires authentication).

    Returns:
    - session_id
    - response
    - metadata
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not image and not question:
        raise HTTPException(status_code=400, detail="Must provide image or question")

    if not pipeline.session_mgr.session_exists(current_user, session_id):
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found for user {current_user}"
        )

    image_path = None
    try:
        # Handle image upload
        if image:
            suffix = os.path.splitext(image.filename)[1] or ".jpg"
            TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            image_path = TEMP_UPLOADS_DIR / f"{uuid4().hex}{suffix}"
            content = await image.read()
            image_path.write_bytes(content)
            image_path = str(image_path)

        # Run pipeline
        result = pipeline.run(
            username=current_user,
            question=question or "What do you see in this image?",
            image_path=image_path,
            session_id=session_id
        )

        # Get full session data
        session_data = pipeline.session_mgr.load(current_user, result["session_id"])

        return {
            "message": "Message sent",
            "session_id": result["session_id"],
            "response": result["response"],
            "metadata": result.get("metadata", {}),
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

@app.get("/chat/history")
async def list_user_chats(current_user: str = Depends(get_current_user)):
    """List all chat sessions for current user (Protected)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    session_ids = pipeline.session_mgr.list_user_sessions(current_user)

    chats = []
    for sid in session_ids:
        try:
            data = pipeline.session_mgr.load(current_user, sid)

            # Get first question
            first_question = data.get("input", {}).get("question", "[Image uploaded]")

            # Get turns count
            history = data.get("conversation_history", [])
            turns_count = len(history)

            # Get language
            if history:
                last_meta = history[-1].get("meta", {})
                lang = (last_meta.get("translation", {}) or {}).get("output_language", "English")
            else:
                lang = "English"

            chats.append({
                "session_id": sid,
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at", data.get("created_at")),
                "first_message": (
                    first_question[:50] + "..."
                    if len(first_question) > 50
                    else first_question
                ),
                "turns": turns_count,
                "language": lang
            })
        except Exception as e:
            print(f"Error loading session {sid}: {e}")
            continue

    return {
        "username": current_user,
        "total_chats": len(chats),
        "chats": sorted(chats, key=lambda x: x["updated_at"], reverse=True)
    }


@app.get("/chat/{session_id}")
async def get_chat_session(
    session_id: int,
    current_user: str = Depends(get_current_user)
):
    """Get full conversation history for a session (Protected)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not pipeline.session_mgr.session_exists(current_user, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return pipeline.session_mgr.load(current_user, session_id)


@app.delete("/chat/{session_id}")
async def delete_chat_session(
    session_id: int,
    current_user: str = Depends(get_current_user)
):
    """Delete a chat session (Protected)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not pipeline.session_mgr.session_exists(current_user, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete from disk
    session_dir = SESSIONS_DIR / current_user / str(session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)

    # Remove from active RAM memory
    if session_id in pipeline.memory.active_sessions:
        pipeline.memory.clear_session(session_id)

    return {"message": f"Session {session_id} deleted"}


# ============== MEMORY ENDPOINTS ==============

@app.get("/memory/stats")
async def get_memory_stats():
    """Get memory statistics"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        stats = pipeline.memory.get_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{session_id}/memory-status")
async def get_session_memory_status(
    session_id: int,
    current_user: str = Depends(get_current_user)
):
    """Return estimated prompt-context usage for a session."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not pipeline.session_mgr.session_exists(current_user, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    history = pipeline.session_mgr.get_conversation_history(current_user, session_id)
    memory_state = pipeline.session_mgr.get_memory_state(current_user, session_id)
    pipeline.memory.get_or_create(session_id, history, memory_state)

    return pipeline.memory.get_context_status(session_id)


@app.get("/chat/{session_id}/summary")
async def get_session_summary(
    session_id: int,
    current_user: str = Depends(get_current_user)
):
    """Return the rolling summary and related compaction metadata."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not pipeline.session_mgr.session_exists(current_user, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    history = pipeline.session_mgr.get_conversation_history(current_user, session_id)
    memory_state = pipeline.session_mgr.get_memory_state(current_user, session_id)
    pipeline.memory.get_or_create(session_id, history, memory_state)

    return {
        "session_id": session_id,
        **pipeline.memory.get_summary(session_id),
        "status": pipeline.memory.get_context_status(session_id)
    }


@app.post("/chat/{session_id}/summarize")
async def summarize_session_memory(
    session_id: int,
    current_user: str = Depends(get_current_user)
):
    """Force summarization of older turns for this session."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not pipeline.session_mgr.session_exists(current_user, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    history = pipeline.session_mgr.get_conversation_history(current_user, session_id)
    memory_state = pipeline.session_mgr.get_memory_state(current_user, session_id)
    pipeline.memory.get_or_create(session_id, history, memory_state)

    result = pipeline.memory.force_compact(
        session_id=session_id,
        summarizer=pipeline.summarizer,
        full_conversation_history=history,
        persist_callback=lambda state: pipeline.session_mgr.update_memory_state(
            current_user, session_id, state
        )
    )

    return {
        "session_id": session_id,
        **result,
        "summary": pipeline.memory.get_summary(session_id)
    }


@app.get("/debug/memory_check/{session_id}")
async def memory_check(
    session_id: int,
    current_user: str = Depends(get_current_user)
):
    """Debug endpoint to check memory state (Protected)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not pipeline.session_mgr.session_exists(current_user, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # JSON source of truth
    session_data = pipeline.session_mgr.load(current_user, session_id)
    json_turns = len(session_data.get("conversation_history", []))

    # RAM memory
    in_ram = session_id in pipeline.memory.active_sessions
    ram_messages = []
    ram_count = 0

    if in_ram:
        mem = pipeline.memory.active_sessions[session_id]
        ram_count = len(mem.messages)

        # Last 6 messages (up to 3 turns)
        last_msgs = mem.messages[-6:] if len(mem.messages) > 6 else mem.messages
        for m in last_msgs:
            ram_messages.append({
                "type": m.role,
                "content": (
                    m.content[:200] + "..."
                    if len(m.content) > 200
                    else m.content
                )
            })

    return {
        "username": current_user,
        "session_id": session_id,
        "json_turns": json_turns,
        "in_ram": in_ram,
        "ram_message_count": ram_count,
        "ram_tail": ram_messages
    }


# ============== ADMIN ENDPOINTS ==============

@app.get("/users")
async def list_users():
    """List all users (Public - for admin/debug purposes)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    users = pipeline.session_mgr.list_users()
    return {"users": users, "count": len(users)}


# ============== LEGACY ENDPOINT (for backward compatibility) ==============

@app.post("/chat")
async def chat_legacy(
    username: str = Form(...),
    question: str = Form(...),
    image: Optional[UploadFile] = File(None),
    session_id: Optional[int] = Form(None)
):
    """
    Legacy chat endpoint (no authentication required for backward compatibility)

    Use /chat/new or /chat/{session_id}/message for new implementations
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    image_path = None
    try:
        # Handle image upload
        if image:
            temp_dir = TEMP_UPLOADS_DIR
            temp_dir.mkdir(exist_ok=True)
            image_path = temp_dir / image.filename

            with image_path.open("wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            image_path = str(image_path)

        # Run pipeline
        result = pipeline.run(
            username=username,
            question=question,
            image_path=image_path,
            session_id=session_id
        )

        # Clean up temp image
        if image_path and Path(image_path).exists():
            Path(image_path).unlink()

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api_refactored:app", host="0.0.0.0", port=8000, reload=True)
