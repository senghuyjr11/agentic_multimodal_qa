"""
Agents package - Clean agent structure

Each agent has ONE clear responsibility.
"""

from .image_agent import ImageAgent, ModelConfig
from .memory_manager import MemoryManager, InMemoryConversation
from .pubmed_agent import PubMedAgent, Article
from .response_generator import ResponseGenerator
from .router_agent import RouterAgent, RoutingDecision
from .session_manager import SessionManager
from .translation_agent import TranslationAgent

__all__ = [
    # Image Agent
    'ImageAgent',
    'ModelConfig',

    # Memory
    'MemoryManager',
    'InMemoryConversation',

    # PubMed
    'PubMedAgent',
    'Article',

    # Response
    'ResponseGenerator',

    # Router
    'RouterAgent',
    'RoutingDecision',

    # Session
    'SessionManager',

    # Translation
    'TranslationAgent',
]