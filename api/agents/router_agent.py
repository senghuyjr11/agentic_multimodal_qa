"""
router_agent.py - Makes routing decisions ONLY

ONE JOB: Decide what agents are needed for this message

NO response generation - that's response_generator.py's job
"""

import json
import re

from dataclasses import dataclass
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI

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
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=google_api_key
        )

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

        message_lower = message.lower().strip()

        # Get recent conversation context (last 5 turns)
        context = self._get_context(memory, num_turns=5)

        prompt = f"""Route this message for a medical assistant.

Conversation:
{context}

User message: {message}
Has image: {has_image}

Return JSON only:
{{
  "response_mode": "casual_chat" | "medical_answer" | "modify_previous",
  "needs_pubmed": true | false,
  "search_query": "2-6 medical keywords or null",
  "reasoning": "short reason"
}}

Guidance:
- casual_chat: greetings, thanks, social chat
- modify_previous: asks to rewrite/translate/shorten/remove refs from prior answer
- medical_answer: medical Q/A or follow-up explanation
- keep search_query as short keywords, never full sentence"""

        decision_dict = self._llm_json(prompt)

        # Create decision object
        decision = RoutingDecision(
            needs_vqa=False,  # Will be set by hard rule below
            needs_pubmed=decision_dict.get("needs_pubmed", False),
            search_query=decision_dict.get("search_query"),
            response_mode=decision_dict.get("response_mode", "medical_answer"),
            reasoning=decision_dict.get("reasoning", "")
        )

        return self._normalize_decision(decision, message_lower, has_image)

    def should_search_pubmed_for_vqa(
        self,
        vqa_answer: str,
        original_question: str
    ) -> bool:
        """
        Decide if VQA answer needs PubMed literature search.

        Returns False if answer is simple (yes/no/numbers/colors/simple locations)
        Returns True if answer contains medical terminology that needs explanation
        """

        prompt = f"""Decide if PubMed search is needed for this VQA result.

Question: {original_question}
VQA answer: {vqa_answer}

Return JSON only:
{{
  "needs_pubmed": true | false,
  "reasoning": "short reason"
}}

Use true when answer contains a medical finding/diagnosis that benefits from evidence.
Use false for trivial outputs (yes/no, numbers, color, side, basic location only)."""
        try:
            parsed = self._llm_json(prompt)
            if "needs_pubmed" in parsed:
                return bool(parsed.get("needs_pubmed"))
        except Exception:
            pass

        answer_lower = (vqa_answer or "").strip().lower()
        if re.fullmatch(r"(yes|no|normal|abnormal|\d+|left|right|upper|lower)\.?", answer_lower):
            return False
        return bool(answer_lower)

    def extract_medical_terms(self, vqa_answer: str) -> str:
        """
        Extract searchable medical terms from VQA answer.

        For complex answers, extracts key medical terminology.
        For simple diagnoses, returns the answer as-is.
        """

        prompt = f"""Convert this VQA answer into a short PubMed keyword query.

VQA answer: {vqa_answer}

Return JSON only:
{{
  "query": "2-6 keywords"
}}

Rules: keyword phrase only, no sentence, no punctuation."""
        parsed = self._llm_json(prompt)
        query = (parsed.get("query") or "").strip()
        query = re.sub(r"[^A-Za-z0-9\s\-]", " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        words = query.split()
        return " ".join(words[:6]) if words else vqa_answer

    def _get_context(
        self,
        memory: InMemoryConversation,
        num_turns: int
    ) -> str:
        """Extract recent conversation context"""

        messages = memory.messages[-(num_turns * 2):] if memory.messages else []

        context_lines = []
        if memory.rolling_summary:
            context_lines.append("Earlier conversation summary:")
            context_lines.append(memory.rolling_summary)
            context_lines.append("")

        for msg in messages:
            if msg.type == "human":
                context_lines.append(f"User: {msg.content}")
            elif msg.type == "ai":
                # Truncate long responses
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                context_lines.append(f"Assistant: {content}")

        return "\n".join(context_lines) if context_lines else "(New conversation)"

    def detect_followup_needs_pubmed(
            self,
            message: str,
            memory: InMemoryConversation
    ) -> tuple[bool, Optional[str]]:
        """
        Detect if user is asking for explanation of previous VQA answer.

        Returns: (needs_pubmed, search_query)
        """

        if len(memory.messages) < 2:
            return False, None

        last_ai_msg = None

        # Find last AI message (LangChain: type == "ai")
        for msg in reversed(memory.messages):
            if msg.type == "ai" and last_ai_msg is None:
                last_ai_msg = msg.content
            if last_ai_msg:
                break

        if not last_ai_msg:
            return False, None

        prompt = f"""Detect if this is a medical follow-up that needs PubMed.

LAST ASSISTANT ANSWER:
{last_ai_msg}

CURRENT USER MESSAGE:
{message}

Return JSON only:
{{
  "needs_pubmed_followup": true/false,
  "search_query": "2-6 keywords" or null,
  "reasoning": "brief reason"
}}

Rules:
- true only when user asks for explanation/detail/clarification of medical content
- false for casual chat, greetings, emotion, social messages, name/memory questions
- false for pure modification/translation requests
- keep search_query as short keywords, not full sentence
"""
        parsed = self._llm_json(prompt)
        needs_pubmed_followup = bool(parsed.get("needs_pubmed_followup", False))
        search_query = parsed.get("search_query")

        if needs_pubmed_followup and not search_query:
            search_query = self._build_followup_search_query(last_ai_msg)

        if not needs_pubmed_followup or not search_query:
            return False, None

        print(f"\n[FOLLOW-UP DETECTED]")
        print(f"  Previous answer: {last_ai_msg}")
        print(f"  User asking: {message}")
        print(f"  Search query: {search_query}")
        print(f"  → Will search PubMed for explanation")

        return True, search_query

    def _build_followup_search_query(self, last_ai_msg: str) -> Optional[str]:
        llm_query = self._extract_search_keywords_with_llm(last_ai_msg)
        if llm_query:
            return llm_query

        return self._fallback_followup_search_query(last_ai_msg)

    def _extract_search_keywords_with_llm(self, last_ai_msg: str) -> Optional[str]:
        if not last_ai_msg:
            return None

        prompt = f"""Convert medical text into short PubMed keywords.

SOURCE TEXT:
{last_ai_msg}

Rules:
- Return JSON only: {{"query":"2-6 keywords"}}
- No full sentence, no markdown"""

        try:
            parsed = self._llm_json(prompt)
            query = (parsed.get("query") or "").strip()

            query = re.sub(r'```.*?```', '', query, flags=re.DOTALL).strip()
            query = re.sub(r'[^A-Za-z0-9\s\-]', ' ', query)
            query = re.sub(r'\s+', ' ', query).strip()
            if not query:
                return None

            words = query.split()
            if len(words) > 6:
                query = " ".join(words[:6])

            return query.strip() or None
        except Exception as e:
            print(f"  LLM keyword extraction failed: {e}")
            return None

    def _fallback_followup_search_query(self, last_ai_msg: str) -> Optional[str]:
        if not last_ai_msg:
            return None

        cleaned = re.sub(r'\s+', ' ', last_ai_msg).strip()
        cleaned = re.sub(r'\*+', '', cleaned)
        cleaned = re.sub(r'\[[^\]]+\]\([^)]+\)', '', cleaned).strip()
        if not cleaned:
            return None

        tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", cleaned.lower())
        if not tokens:
            return None

        common = {
            "the", "and", "for", "with", "this", "that", "from", "into", "shows",
            "showing", "image", "likely", "could", "would", "there", "their"
        }
        keywords = []
        for token in tokens:
            if token in common or len(token) < 3:
                continue
            if token not in keywords:
                keywords.append(token)
            if len(keywords) == 6:
                break

        if keywords:
            return " ".join(keywords)
        return " ".join(tokens[:6]) if tokens else None

    def is_asking_about_previous_references(self, message: str) -> bool:
        prompt = f"""Does this message ask about earlier references/articles?

Message: {message}

Return JSON only:
{{
  "asks_previous_references": true/false
}}
"""
        try:
            parsed = self._llm_json(prompt)
            return bool(parsed.get("asks_previous_references", False))
        except Exception:
            message_lower = message.lower()
            return "reference" in message_lower and ("previous" in message_lower or "those" in message_lower)

    def _llm_json(self, prompt: str) -> dict:
        raw = self.llm.invoke(prompt)
        text = str(raw.content if hasattr(raw, "content") else raw).strip()
        return self._parse_json_response(text)

    def _parse_json_response(self, text: str) -> dict:
        cleaned = re.sub(r"```json\s*|\s*```", "", text, flags=re.IGNORECASE).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
        return {}

    def _normalize_decision(self, decision: RoutingDecision, message_lower: str, has_image: bool) -> RoutingDecision:
        if self._is_identity_or_memory_message(message_lower):
            decision.response_mode = "casual_chat"
            decision.needs_pubmed = False
            decision.search_query = None

        allowed_modes = {"casual_chat", "medical_answer", "modify_previous"}
        if decision.response_mode not in allowed_modes:
            decision.response_mode = "medical_answer"

        # Non-negotiable invariants
        if decision.response_mode in {"casual_chat", "modify_previous"}:
            decision.needs_pubmed = False
            decision.search_query = None

        if decision.response_mode == "medical_answer" and decision.needs_pubmed and not decision.search_query:
            decision.search_query = message_lower.strip()[:120] or None

        # Image handling invariant
        if has_image:
            decision.needs_vqa = True
            if decision.response_mode == "casual_chat":
                decision.response_mode = "medical_answer"
                decision.needs_pubmed = False
                decision.search_query = None
        else:
            decision.needs_vqa = False

        return decision

    @staticmethod
    def _is_identity_or_memory_message(message_lower: str) -> bool:
        text = (message_lower or "").strip()
        if not text:
            return False

        memory_patterns = [
            "my name",
            "what is my name",
            "what was my name",
            "what's my name",
            "do you know my name",
            "remember my name",
            "do you remember",
            "what did i ask",
            "first question",
        ]
        if any(pattern in text for pattern in memory_patterns):
            return True

        return bool(
            re.search(r"\b(my name is|i am|i'm)\s+[a-z][a-z\-']{0,39}\b", text, flags=re.IGNORECASE)
        )


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

    # Test 3: Language question
    decision = router.decide(
        message="do you know Khmer?",
        has_image=False,
        memory=memory
    )
    print(f"\nTest 3 Result: {decision}")
