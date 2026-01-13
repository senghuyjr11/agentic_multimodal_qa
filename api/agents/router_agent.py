"""
router_agent.py - Makes routing decisions ONLY

ONE JOB: Decide what agents are needed for this message

NO response generation - that's response_generator.py's job
"""

import google.generativeai as genai
from dataclasses import dataclass
from typing import Optional

# Import from our simple memory manager (relative import)
from .memory_manager import InMemoryConversation


@dataclass
class RoutingDecision:
    """What agents should be called?"""
    needs_vqa: bool
    needs_pubmed: bool
    search_query: Optional[str]
    response_mode: str  # "medical_answer" | "casual_chat" | "modify_previous"
    reasoning: str


class RouterAgent:
    """
    Decides what to do with a message.

    Does NOT generate responses - only decides routing.
    """

    def __init__(self, google_api_key: str):
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel("gemma-3-4b-it")

    def decide(
        self,
        message: str,
        has_image: bool,
        memory: InMemoryConversation
    ) -> RoutingDecision:
        """
        ONE JOB: Decide what agents are needed.

        Args:
            message: User's question
            has_image: Does user have an image?
            memory: Conversation history

        Returns:
            RoutingDecision with what to do
        """

        # Get recent conversation context (last 5 turns)
        context = self._get_context(memory, num_turns=5)

        # Build prompt for LLM
        prompt = f"""You are a routing agent. Decide what's needed for this message.

CONVERSATION HISTORY:
{context}

CURRENT MESSAGE:
User: {message}
Has image: {has_image}

ROUTING OPTIONS:
1. needs_vqa: true if image needs analysis
2. needs_pubmed: true if medical literature needed
3. response_mode:
   - "medical_answer": Medical question needing explanation
   - "casual_chat": Greeting, thanks, user info questions
   - "modify_previous": "remove references", "make shorter", "translate to X", "explain in X language"

IMPORTANT RULES:
- User info questions (name, remember me) → casual_chat, no PubMed
- System questions (who are you) → casual_chat, no PubMed
- Medical questions → medical_answer + PubMed
- Image uploads → needs_vqa=true
- Translation requests ("translate to X", "in Spanish", "explain in French") → modify_previous, no PubMed, no VQA

OUTPUT (JSON only):
{{
  "needs_vqa": true/false,
  "needs_pubmed": true/false,
  "search_query": "search terms" or null,
  "response_mode": "medical_answer"|"casual_chat"|"modify_previous",
  "reasoning": "why this decision"
}}"""

        # Ask LLM
        response = self.model.generate_content(prompt)
        text = response.text.strip()

        # Parse JSON
        import json
        import re

        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*|\s*```', '', text)
        decision_dict = json.loads(text)

        # Create decision object
        decision = RoutingDecision(
            needs_vqa=decision_dict.get("needs_vqa", False),
            needs_pubmed=decision_dict.get("needs_pubmed", False),
            search_query=decision_dict.get("search_query"),
            response_mode=decision_dict.get("response_mode", "medical_answer"),
            reasoning=decision_dict.get("reasoning", "")
        )

        # Log decision
        print(f"\n[ROUTER DECISION]")
        print(f"  Mode: {decision.response_mode}")
        print(f"  VQA needed: {decision.needs_vqa}")
        print(f"  PubMed needed: {decision.needs_pubmed}")
        if decision.search_query:
            print(f"  Search: {decision.search_query}")
        print(f"  Reasoning: {decision.reasoning}")

        return decision

    def _get_context(
        self,
        memory: InMemoryConversation,
        num_turns: int
    ) -> str:
        """Extract recent conversation context"""

        messages = memory.messages[-(num_turns * 2):] if memory.messages else []

        context_lines = []
        for msg in messages:
            if msg.role == "user":
                context_lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                # Truncate long responses
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                context_lines.append(f"Assistant: {content}")

        return "\n".join(context_lines) if context_lines else "(New conversation)"


if __name__ == "__main__":
    import os
    from memory_manager import MemoryManager

    # Test
    router = RouterAgent(google_api_key=os.getenv("GOOGLE_API_KEY"))

    memory_mgr = MemoryManager()
    memory = memory_mgr.get_or_create(session_id=1)

    # Test 1: Medical question
    decision = router.decide(
        message="What is diabetes?",
        has_image=False,
        memory=memory
    )
    print(f"\nTest 1 Result: {decision}")

    # Test 2: Casual question
    decision = router.decide(
        message="Hello, my name is John",
        has_image=False,
        memory=memory
    )
    print(f"\nTest 2 Result: {decision}")