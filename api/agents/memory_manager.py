"""
memory_manager.py - Manages conversation memory ONLY

ONE JOB: Handle conversation history in RAM

Simple implementation without LangChain dependency.
"""

from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any, Optional

from langchain_community.chat_message_histories import ChatMessageHistory


class InMemoryConversation:
    def __init__(self):
        self.history = ChatMessageHistory()  # ← LangChain here
        self.pubmed_articles = []
        self.rolling_summary = ""
        self.summarized_turn_count = 0
        self.compression_count = 0
        self.last_compaction_at = None

    def add_user_message(self, content: str):
        self.history.add_user_message(content)  # ← LangChain method

    def add_ai_message(self, content: str):
        self.history.add_ai_message(content)  # ← LangChain method

    @property
    def messages(self):
        return self.history.messages  # ← returns LangChain HumanMessage/AIMessage object

    def approximate_tokens(self) -> int:
        total_chars = len(self.rolling_summary or "")
        total_chars += sum(len(getattr(msg, "content", "")) for msg in self.messages)
        return max(1, total_chars // 4)

    def memory_state(self) -> dict:
        return {
            "rolling_summary": self.rolling_summary,
            "summarized_turn_count": self.summarized_turn_count,
            "compression_count": self.compression_count,
            "last_compaction_at": self.last_compaction_at,
            "estimated_tokens": self.approximate_tokens(),
        }

class MemoryManager:
    """
    Manages conversation memory in RAM.

    Does NOT persist to disk - that's session_manager.py's job.
    """

    def __init__(self):
        # Active conversations: {session_id: memory}
        self.active_sessions: Dict[int, InMemoryConversation] = {}
        self.max_context_tokens = 12000
        self.compact_trigger_ratio = 0.8
        self.keep_recent_turns = 6

    def get_or_create(
        self,
        session_id: int,
        conversation_history: List[dict] = None,
        memory_state: Optional[dict] = None
    ) -> InMemoryConversation:
        """
        ONE JOB: Get memory for a session (create if new).

        Args:
            session_id: Session identifier
            conversation_history: Past turns to restore (from JSON)

        Returns:
            InMemoryConversation for this session
        """

        # Already in cache?
        if session_id in self.active_sessions:
            print(f"✓ Using cached memory for session {session_id}")
            return self.active_sessions[session_id]

        # Create new memory
        memory = InMemoryConversation()

        if memory_state:
            memory.rolling_summary = memory_state.get("rolling_summary", "") or ""
            memory.summarized_turn_count = int(memory_state.get("summarized_turn_count", 0) or 0)
            memory.compression_count = int(memory_state.get("compression_count", 0) or 0)
            memory.last_compaction_at = memory_state.get("last_compaction_at")

        # Restore from history if provided
        if conversation_history:
            turns_to_restore = conversation_history[memory.summarized_turn_count:]

            for turn in turns_to_restore:
                memory.add_user_message(turn["user"])
                memory.add_ai_message(turn["assistant"])
            print(
                f"✓ Restored {len(turns_to_restore)} active turns for session {session_id}"
                f" ({memory.summarized_turn_count} summarized)"
            )
        else:
            print(f"✓ Created new memory for session {session_id}")

        # Cache it
        self.active_sessions[session_id] = memory

        return memory

    def add_turn(
        self,
        session_id: int,
        user_message: str,
        ai_message: str
    ):
        """
        Add a conversation turn to memory.

        Args:
            session_id: Session identifier
            user_message: What user said
            ai_message: What assistant said
        """

        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found in memory")

        memory = self.active_sessions[session_id]
        memory.add_user_message(user_message)
        memory.add_ai_message(ai_message)

        print(f"✓ Added turn to memory (session {session_id})")

    def get_last_ai_message(self, session_id: int) -> str:
        """Get the last assistant message (for modify mode)"""

        if session_id not in self.active_sessions:
            return None

        memory = self.active_sessions[session_id]

        # Find last AI message (LangChain AIMessage has type == "ai")
        for msg in reversed(memory.messages):
            if msg.type == "ai":
                return msg.content

        return None

    # ========== NEW: PubMed Article Caching ==========

    def store_pubmed_articles(self, session_id: int, articles: List[Any]):
        """
        Store PubMed articles for this session.

        Args:
            session_id: Session identifier
            articles: List of Article objects from PubMed search
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found in memory")

        self.active_sessions[session_id].pubmed_articles = articles
        print(f"✓ Cached {len(articles)} PubMed articles for session {session_id}")

    def get_pubmed_articles(self, session_id: int) -> List[Any]:
        """
        Get cached PubMed articles for this session.

        Args:
            session_id: Session identifier

        Returns:
            List of Article objects (empty if none cached)
        """
        if session_id not in self.active_sessions:
            return []

        return self.active_sessions[session_id].pubmed_articles

    def clear_pubmed_articles(self, session_id: int):
        """
        Clear cached articles (call when user uploads new image or changes topic).

        Args:
            session_id: Session identifier
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id].pubmed_articles = []
            print(f"✓ Cleared PubMed cache for session {session_id}")

    # ================================================

    def get_context_window(
        self,
        session_id: int,
        num_turns: int
    ) -> List[Tuple[str, str]]:
        """
        Get recent conversation turns.

        Args:
            session_id: Session identifier
            num_turns: How many turns to retrieve

        Returns:
            List of (user_msg, ai_msg) tuples
        """

        if session_id not in self.active_sessions:
            return []

        memory = self.active_sessions[session_id]
        messages = memory.messages[-(num_turns * 2):]

        turns = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                turns.append((messages[i].content, messages[i+1].content))

        return turns

    def clear_session(self, session_id: int):
        """Remove session from cache"""

        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            print(f"✓ Cleared memory for session {session_id}")

    def clear_all(self):
        """Clear all sessions (e.g., on app restart)"""

        count = len(self.active_sessions)
        self.active_sessions.clear()
        print(f"✓ Cleared all memory ({count} sessions)")

    def get_stats(self) -> dict:
        """Get memory statistics"""

        return {
            "active_sessions": len(self.active_sessions),
            "session_ids": list(self.active_sessions.keys()),
            "total_messages": sum(
                len(mem.messages)
                for mem in self.active_sessions.values()
            ),
            "sessions_with_articles": sum(  # NEW
                1 for mem in self.active_sessions.values()
                if mem.pubmed_articles
            ),
            "sessions_with_summary": sum(
                1 for mem in self.active_sessions.values()
                if mem.rolling_summary
            )
        }

    def get_summary(self, session_id: int) -> dict:
        if session_id not in self.active_sessions:
            return {
                "rolling_summary": "",
                "summarized_turn_count": 0,
                "compression_count": 0,
                "last_compaction_at": None
            }

        memory = self.active_sessions[session_id]
        return {
            "rolling_summary": memory.rolling_summary,
            "summarized_turn_count": memory.summarized_turn_count,
            "compression_count": memory.compression_count,
            "last_compaction_at": memory.last_compaction_at
        }

    def get_context_status(self, session_id: int) -> dict:
        if session_id not in self.active_sessions:
            return {
                "session_id": session_id,
                "estimated_tokens": 0,
                "max_context_tokens": self.max_context_tokens,
                "usage_ratio": 0.0,
                "percent_used": 0,
                "percent_left": 100,
                "active_context_tokens": 0,
                "active_context_usage_ratio": 0.0,
                "active_context_percent_used": 0,
                "active_context_percent_left": 100,
                "message_count": 0,
                "active_turns": 0,
                "summarized_turn_count": 0,
                "compression_count": 0,
                "has_summary": False,
                "should_compact": False,
            }

        memory = self.active_sessions[session_id]
        estimated_tokens = memory.approximate_tokens()
        usage_ratio = min(estimated_tokens / self.max_context_tokens, 1.0)
        active_context_tokens = max(
            1,
            sum(len(getattr(msg, "content", "")) for msg in memory.messages) // 4
        )
        active_context_usage_ratio = min(
            active_context_tokens / self.max_context_tokens,
            1.0
        )

        return {
            "session_id": session_id,
            "estimated_tokens": estimated_tokens,
            "max_context_tokens": self.max_context_tokens,
            "usage_ratio": round(usage_ratio, 4),
            "percent_used": int(round(usage_ratio * 100)),
            "percent_left": max(0, 100 - int(round(usage_ratio * 100))),
            "active_context_tokens": active_context_tokens,
            "active_context_usage_ratio": round(active_context_usage_ratio, 4),
            "active_context_percent_used": int(round(active_context_usage_ratio * 100)),
            "active_context_percent_left": max(0, 100 - int(round(active_context_usage_ratio * 100))),
            "message_count": len(memory.messages),
            "active_turns": len(memory.messages) // 2,
            "summarized_turn_count": memory.summarized_turn_count,
            "compression_count": memory.compression_count,
            "has_summary": bool(memory.rolling_summary),
            "last_compaction_at": memory.last_compaction_at,
            "should_compact": usage_ratio >= self.compact_trigger_ratio,
        }

    def compact_if_needed(
        self,
        session_id: int,
        summarizer,
        full_conversation_history: List[dict],
        persist_callback: Optional[Callable[[dict], None]] = None
    ) -> dict:
        status = self.get_context_status(session_id)
        if not status["should_compact"]:
            return {
                "compacted": False,
                "reason": "below_threshold",
                "status": status
            }

        return self.force_compact(
            session_id=session_id,
            summarizer=summarizer,
            full_conversation_history=full_conversation_history,
            persist_callback=persist_callback
        )

    def force_compact(
        self,
        session_id: int,
        summarizer,
        full_conversation_history: List[dict],
        persist_callback: Optional[Callable[[dict], None]] = None
    ) -> dict:
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found in memory")

        memory = self.active_sessions[session_id]
        unsummarized_history = full_conversation_history[memory.summarized_turn_count:]

        if len(unsummarized_history) <= self.keep_recent_turns:
            return {
                "compacted": False,
                "reason": "not_enough_turns",
                "status": self.get_context_status(session_id)
            }

        turns_to_compact = unsummarized_history[:-self.keep_recent_turns]
        recent_turns = unsummarized_history[-self.keep_recent_turns:]

        updated_summary = summarizer.summarize(
            existing_summary=memory.rolling_summary,
            turns=turns_to_compact
        )

        replacement_memory = InMemoryConversation()
        replacement_memory.rolling_summary = updated_summary
        replacement_memory.summarized_turn_count = memory.summarized_turn_count + len(turns_to_compact)
        replacement_memory.compression_count = memory.compression_count + 1
        replacement_memory.last_compaction_at = datetime.now().isoformat()
        replacement_memory.pubmed_articles = memory.pubmed_articles

        for turn in recent_turns:
            replacement_memory.add_user_message(turn["user"])
            replacement_memory.add_ai_message(turn["assistant"])

        self.active_sessions[session_id] = replacement_memory

        if persist_callback:
            persist_callback(replacement_memory.memory_state())

        print(
            f"✓ Compacted session {session_id}: "
            f"{len(turns_to_compact)} older turns summarized, "
            f"{len(recent_turns)} recent turns kept"
        )

        return {
            "compacted": True,
            "summarized_turns_added": len(turns_to_compact),
            "status": self.get_context_status(session_id),
            "summary_preview": updated_summary[:240]
        }


if __name__ == "__main__":
    # Test
    manager = MemoryManager()

    # Test 1: Create new memory
    memory = manager.get_or_create(session_id=1)
    print(f"Memory created: {memory}")

    # Test 2: Add turn
    manager.add_turn(
        session_id=1,
        user_message="What is diabetes?",
        ai_message="Diabetes is a chronic condition..."
    )

    # Test 3: Get context
    context = manager.get_context_window(session_id=1, num_turns=1)
    print(f"Context: {context}")

    # Test 4: Cache articles (NEW)
    @dataclass
    class MockArticle:
        title: str
        pmid: str

    mock_articles = [
        MockArticle(title="Article 1", pmid="123"),
        MockArticle(title="Article 2", pmid="456")
    ]
    manager.store_pubmed_articles(session_id=1, articles=mock_articles)

    # Test 5: Retrieve articles (NEW)
    cached = manager.get_pubmed_articles(session_id=1)
    print(f"Cached articles: {len(cached)}")

    # Test 6: Stats
    stats = manager.get_stats()
    print(f"Stats: {stats}")

    # Test 7: Restore from history
    history = [
        {"user": "Hello", "assistant": "Hi there!"},
        {"user": "What is diabetes?", "assistant": "Diabetes is..."}
    ]
    memory2 = manager.get_or_create(session_id=2, conversation_history=history)
    print(f"Restored turns: {len(memory2.messages) // 2}")
