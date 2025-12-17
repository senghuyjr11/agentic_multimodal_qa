"""
api.py - FastAPI endpoints for Medical VQA Pipeline
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
    description="Multi-agent Medical Visual Question Answering System",
    version="1.0.0"
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
    print("INITIALIZING MEDICAL VQA API")
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
    print("ALL MODELS READY")
    print("=" * 60)



# ============== ENDPOINTS ==============

@app.get("/health")
async def health_check():
    """Health check."""
    return {"status": "healthy", "pipeline_ready": pipeline is not None}


@app.post("/predict")
async def predict(
    username: str = Form(...),
    question: Optional[str] = Form(None),
    language: str = Form("English"),
    image: Optional[UploadFile] = File(None)
):
    """
    Main prediction endpoint.

    - username: required
    - question: optional (uses default if image provided)
    - language: default "English"
    - image: optional file upload
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
            image_path=image_path,
            language=language
        )

        return pipeline.session_manager.load(username, result["session_id"])

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)


@app.get("/sessions/{username}")
async def list_sessions(username: str):
    """List all sessions for a user."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    session_ids = pipeline.session_manager.list_user_sessions(username)

    sessions = []
    for sid in session_ids:
        data = pipeline.session_manager.load(username, sid)
        sessions.append({
            "session_id": sid,
            "created_at": data["created_at"],
            "input_type": data["input"].get("input_type", "unknown"),
            "question": data["input"].get("question")
        })

    return sessions


@app.get("/sessions/{username}/{session_id}")
async def get_session(username: str, session_id: int):
    """Get session detail."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not pipeline.session_manager.session_exists(username, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return pipeline.session_manager.load(username, session_id)


@app.delete("/sessions/{username}/{session_id}")
async def delete_session(username: str, session_id: int):
    """Delete a session."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not pipeline.session_manager.session_exists(username, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    import shutil
    session_dir = pipeline.session_manager._get_session_dir(username, session_id)
    shutil.rmtree(session_dir)

    return {"message": f"Session {username}/{session_id} deleted"}


@app.get("/users")
async def list_users():
    """List all users."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return pipeline.session_manager.list_users()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)