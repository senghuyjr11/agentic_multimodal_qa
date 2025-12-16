"""
session_manager.py - Manages session storage per user
"""
import os
import json
import shutil
from datetime import datetime


class SessionManager:
    """Creates and manages session folders per user."""

    def __init__(self, base_dir: str = "sessions"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

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

    def create_session(
        self,
        username: str,
        question: str,
        image_path: str = None  # Optional now
    ) -> int:
        """Create new session for user."""
        user_dir = self._get_user_dir(username)
        os.makedirs(user_dir, exist_ok=True)

        session_id = self._get_next_session_id(username)
        session_dir = self._get_session_dir(username, session_id)
        os.makedirs(session_dir)

        # Copy image if provided
        saved_image_path = None
        original_image_path = None
        if image_path:
            image_ext = os.path.splitext(image_path)[1]
            saved_image_path = f"input_image{image_ext}"
            original_image_path = image_path
            shutil.copy(image_path, os.path.join(session_dir, saved_image_path))

        session_data = {
            "username": username,
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "input": {
                "image_path": saved_image_path,
                "original_image_path": original_image_path,
                "question": question,
                "input_type": self._get_input_type(image_path, question)
            },
            "image_agent": None,
            "vqa_agent": None,
            "text_agent": None,
            "reasoning_agent": None
        }

        self._save(username, session_id, session_data)
        print(f"✓ Created session: {username}/{session_id}")

        return session_id

    def _get_input_type(self, image_path: str, question: str) -> str:
        if image_path and question:
            return "image_and_text"
        elif image_path:
            return "image_only"
        else:
            return "text_only"

    def update(self, username: str, session_id: int, agent_name: str, data: dict):
        """Update session with agent output."""
        session_data = self.load(username, session_id)
        session_data[agent_name] = data
        session_data["updated_at"] = datetime.now().isoformat()
        self._save(username, session_id, session_data)
        print(f"✓ Updated {agent_name}")

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
        return [d for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d))]