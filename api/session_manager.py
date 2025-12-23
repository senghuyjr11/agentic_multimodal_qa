"""
session_manager.py - Manages session storage per user (CLEAN TURN-BASED SCHEMA)
"""
import os
import json
import shutil
from datetime import datetime
from dataclasses import asdict, is_dataclass


class SessionManager:
    """Creates and manages session folders per user."""

    def __init__(self, base_dir: str = "sessions"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _now_iso(self) -> str:
        return datetime.now().isoformat()

    def _get_user_dir(self, username: str) -> str:
        return os.path.join(self.base_dir, username)

    def _get_session_dir(self, username: str, session_id: int) -> str:
        return os.path.join(self._get_user_dir(username), str(session_id))

    def _get_next_session_id(self, username: str) -> int:
        user_dir = self._get_user_dir(username)
        if not os.path.exists(user_dir):
            return 1
        existing = [int(d) for d in os.listdir(user_dir) if d.isdigit()]
        return max(existing, default=0) + 1

    def _get_input_type(self, image_path: str, question: str) -> str:
        if image_path and question:
            return "image_and_text"
        elif image_path:
            return "image_only"
        else:
            return "text_only"

    def create_session(
        self,
        username: str,
        question: str,
        image_path: str = None
    ) -> int:
        """Create new session for user (clean schema)."""
        user_dir = self._get_user_dir(username)
        os.makedirs(user_dir, exist_ok=True)

        session_id = self._get_next_session_id(username)
        session_dir = self._get_session_dir(username, session_id)
        os.makedirs(session_dir)

        # Copy initial image if provided
        saved_image_path = None
        if image_path:
            image_ext = os.path.splitext(image_path)[1] or ".jpg"
            filename = f"input_image{image_ext}"
            shutil.copy(image_path, os.path.join(session_dir, filename))
            saved_image_path = f"sessions/{username}/{session_id}/{filename}"

        session_data = {
            "username": username,
            "session_id": session_id,
            "created_at": self._now_iso(),
            "updated_at": self._now_iso(),
            "input": {
                "image_path": saved_image_path,
                "question": question,
                "input_type": self._get_input_type(image_path, question)
            },
            "conversation_history": []
        }

        self._save(username, session_id, session_data)
        print(f"✓ Created session: {username}/{session_id}")
        return session_id

    def load(self, username: str, session_id: int) -> dict:
        """Load session data."""
        path = os.path.join(self._get_session_dir(username, session_id), "session_data.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, username: str, session_id: int, data: dict):
        """Save session data."""
        path = os.path.join(self._get_session_dir(username, session_id), "session_data.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def session_exists(self, username: str, session_id: int) -> bool:
        """Check if session exists."""
        return os.path.exists(self._get_session_dir(username, session_id))

    def list_user_sessions(self, username: str) -> list[int]:
        """List all session IDs for a user."""
        user_dir = self._get_user_dir(username)
        if not os.path.exists(user_dir):
            return []
        return sorted([int(d) for d in os.listdir(user_dir) if d.isdigit()])

    def list_users(self) -> list[str]:
        """List all usernames."""
        return [
            d for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d))
        ]

    def _normalize(self, obj):
        """Ensure object is JSON serializable; converts dataclasses recursively."""
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, list):
            return [self._normalize(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._normalize(v) for k, v in obj.items()}
        return obj

    def add_conversation_turn(
        self,
        username: str,
        session_id: int,
        user_message: str,
        assistant_message: str,
        image_path: str = None,
        meta: dict = None
    ):
        """
        Add a single Q&A turn to conversation history with optional image + per-turn meta.
        CLEAN: all agent outputs live ONLY inside turn.meta (no top-level agent fields).
        """
        session_data = self.load(username, session_id)

        if "conversation_history" not in session_data:
            session_data["conversation_history"] = []

        turn_number = len(session_data["conversation_history"]) + 1

        # Copy image for this turn if provided
        saved_image_path = None
        if image_path:
            session_dir = self._get_session_dir(username, session_id)
            image_ext = os.path.splitext(image_path)[1] or ".jpg"
            filename = f"input_image_turn{turn_number}{image_ext}"
            shutil.copy(image_path, os.path.join(session_dir, filename))
            saved_image_path = f"sessions/{username}/{session_id}/{filename}"

        meta = meta or {}
        meta = self._normalize(meta)

        session_data["conversation_history"].append({
            "turn": turn_number,
            "user": user_message,
            "assistant": assistant_message,
            "image_path": saved_image_path,
            "timestamp": self._now_iso(),
            "meta": meta
        })

        session_data["updated_at"] = self._now_iso()
        self._save(username, session_id, session_data)
        print(f"✓ Added turn {turn_number} to conversation")

    def get_conversation_history(self, username: str, session_id: int) -> list:
        """Get all conversation turns from a session."""
        session_data = self.load(username, session_id)
        return session_data.get("conversation_history", [])
