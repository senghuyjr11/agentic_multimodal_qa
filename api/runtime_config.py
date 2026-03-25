"""
runtime_config.py - Centralized runtime paths and cache setup
"""

import os
from pathlib import Path


API_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = API_ROOT.parent
APP_ENV = os.getenv("APP_ENV", "dev").strip().lower()


def _resolve_path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


HF_CACHE_DIR = _resolve_path("APP_HF_CACHE_DIR", PROJECT_ROOT / ".hf_cache")
SESSIONS_DIR = _resolve_path("APP_SESSIONS_DIR", API_ROOT / "sessions")
TEMP_UPLOADS_DIR = _resolve_path("APP_TEMP_UPLOADS_DIR", API_ROOT / "temp_uploads")
USERS_FILE = _resolve_path("APP_USERS_FILE", API_ROOT / "users.json")

PATHVQA_ADAPTER_DIR = _resolve_path(
    "APP_PATHVQA_ADAPTER_DIR",
    PROJECT_ROOT / "pathvqa_qwen3vl_pipeline" / "adapters"
)
SLAKE_ADAPTER_DIR = _resolve_path(
    "APP_SLAKE_ADAPTER_DIR",
    PROJECT_ROOT / "slake_qwen3vl_pipeline" / "adapters"
)
CLASSIFIER_MODEL_DIR = _resolve_path(
    "APP_CLASSIFIER_DIR",
    PROJECT_ROOT / "modality_classifier_pipeline" / "model"
)


def configure_runtime_environment():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    force_project_cache = _bool_env(
        "APP_FORCE_PROJECT_CACHE",
        default=(APP_ENV == "prod")
    )

    if force_project_cache:
        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
        os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))
