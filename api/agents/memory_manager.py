"""
memory_manager.py - Manages conversation memory ONLY

ONE JOB: Handle conversation history in RAM

Simple implementation without LangChain dependency.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    """Simple message representation"""
    role: str  # "user" or "assistant"
    content: str


class InMemoryConversation:
    """
    Simple in-memory conversation history.

    Replacement for LangChain's InMemoryChatMessageHistory.
    """

    def __init__(self):
        self.messages: List[ChatMessage] = []
        self.pubmed_articles: List[Any] = []  # NEW: Cache PubMed articles

    def add_user_message(self, content: str):
        """Add user message"""
        self.messages.append(ChatMessage(role="user", content=content))

    def add_ai_message(self, content: str):
        """Add AI message"""
        self.messages.append(ChatMessage(role="assistant", content=content))

    def get_messages(self) -> List[ChatMessage]:
        """Get all messages"""
        return self.messages

    def clear(self):
        """Clear all messages"""
        self.messages.clear()
        self.pubmed_articles.clear()  # NEW: Also clear articles


class MemoryManager:
    """
    Manages conversation memory in RAM.

    Does NOT persist to disk - that's session_manager.py's job.
    """

    def __init__(self):
        # Active conversations: {session_id: memory}
        self.active_sessions: Dict[int, InMemoryConversation] = {}

    def get_or_create(
        self,
        session_id: int,
        conversation_history: List[dict] = None
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

        # Restore from history if provided
        if conversation_history:
            for turn in conversation_history:
                memory.add_user_message(turn["user"])
                memory.add_ai_message(turn["assistant"])
            print(f"✓ Restored {len(conversation_history)} turns for session {session_id}")
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

        # Find last AI message
        for msg in reversed(memory.messages):
            if msg.role == "assistant":
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
            )
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