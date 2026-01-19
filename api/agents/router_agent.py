"""
router_agent.py - Makes routing decisions ONLY

ONE JOB: Decide what agents are needed for this message

NO response generation - that's response_generator.py's job
"""

from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai

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
           - "casual_chat": Greeting, thanks, acknowledgments ONLY (no images)
           - "modify_previous": "remove references", "make shorter", "translate to X"

        DECISION RULES (FOLLOW STRICTLY):

        RULE 1 - IMAGE UPLOADS:
        - If has_image=true → response_mode="medical_answer", needs_vqa=true, needs_pubmed=true
        - Image questions like "what do you see" → ALWAYS medical_answer, NOT casual_chat

        RULE 2 - CASUAL RESPONSES (NO PUBMED):
        - Greetings: hi, hello, hey, good morning
        - Thanks: thanks, thank you, appreciate it, grateful
        - Acknowledgments: ok, I see, got it, understood, good, great, nice
        - User info: "my name is X", "I'm X", "call me X", "remember me"
        - Memory questions: "do you remember my name?", "what's my name?", "who am I?"
        - System questions: "who are you?", "what can you do?"
        - Farewells: bye, goodbye, see you
        - ALL casual responses → response_mode="casual_chat", needs_pubmed=FALSE, needs_vqa=FALSE, search_query=null

        RULE 3 - MEDICAL QUESTIONS:
        - Questions about diseases, symptoms, definitions → medical_answer, needs_pubmed=true

        RULE 4 - MODIFICATIONS:
        - "summary", "shorter", "translate to X" → modify_previous

        OUTPUT (JSON only):
        {{
          "needs_vqa": true/false,
          "needs_pubmed": true/false,
          "search_query": "search terms" or null,
          "response_mode": "medical_answer"|"casual_chat"|"modify_previous",
          "reasoning": "why this decision"
        }}

        CRITICAL: If has_image=true, you MUST set needs_vqa=true and response_mode="medical_answer"!"""

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

    def generate_search_from_vqa(
            self,
            vqa_answer: str,
            original_question: str
    ) -> str:
        """
        Generate PubMed search query from VQA answer.

        Called when router decides needs_pubmed=true but couldn't generate
        search_query because VQA hadn't run yet.
        """

        prompt = f"""Extract medical search terms for PubMed from this VQA result.

    ORIGINAL QUESTION: {original_question}
    VQA ANSWER: {vqa_answer}

    Generate concise PubMed search terms (2-5 key medical terms).
    Focus on:
    - Specific conditions/diseases mentioned
    - Anatomical locations
    - Pathological findings
    - Medical terminology

    OUTPUT (search terms only, no explanation):"""

        response = self.model.generate_content(prompt)
        search_query = response.text.strip()

        return search_query

    def should_search_pubmed_for_vqa(
            self,
            vqa_answer: str,
            original_question: str
    ) -> bool:
        """
        Decide if VQA answer needs PubMed literature search.

        Returns False if answer is simple (yes/no/numbers/colors/locations)
        Returns True if answer contains medical terminology that needs explanation
        """

        # Simple answers that don't need literature
        simple_patterns = [
            r'^(yes|no|yeah|nope)\.?$',  # Yes/No
            r'^(normal|abnormal)\.?$',  # Normal/Abnormal
            r'^\d+\.?$',  # Just numbers
            r'^(left|right|upper|lower|anterior|posterior)\.?$',  # Simple locations
            r'^(white|black|red|blue|green|yellow|gray|grey)\.?$',  # Colors
        ]

        answer_lower = vqa_answer.lower().strip()

        import re
        for pattern in simple_patterns:
            if re.match(pattern, answer_lower, re.IGNORECASE):
                return False

        # If answer is very short (1-3 words) and question is yes/no type
        words = vqa_answer.split()
        yes_no_questions = ['is this', 'is there', 'are there', 'does this', 'can you see']

        if len(words) <= 3 and any(q in original_question.lower() for q in yes_no_questions):
            return False

        # If answer contains medical terminology (more than 3 words or medical keywords)
        medical_keywords = [
            'infarct', 'fracture', 'lesion', 'tumor', 'mass', 'opacity',
            'pneumonia', 'carcinoma', 'infiltrate', 'edema', 'hemorrhage',
            'stenosis', 'occlusion', 'thrombosis', 'ischemia', 'necrosis',
            'hyperplasia', 'atrophy', 'hypertrophy', 'inflammation'
        ]

        if any(keyword in answer_lower for keyword in medical_keywords):
            return True

        # Default: if answer is longer than 3 words, probably needs explanation
        return len(words) > 3

    def extract_medical_terms(self, vqa_answer: str) -> str:
        """
        Extract searchable medical terms from VQA answer.

        For complex answers, extracts key medical terminology.
        For simple diagnoses, returns the answer as-is.
        """

        # If answer is already concise (2-4 words), use it directly
        words = vqa_answer.split()
        if 2 <= len(words) <= 4:
            return vqa_answer

        # For longer answers, extract key terms
        prompt = f"""Extract the main medical condition/diagnosis from this VQA answer for PubMed search.

    VQA ANSWER: {vqa_answer}

    Return ONLY the key medical term(s) for searching (2-5 words max).
    Examples:
    - "cerebral infarct in the right hemisphere" → "cerebral infarct"
    - "fracture of the left femur with displacement" → "femur fracture"
    - "large mass in the lung with surrounding opacity" → "lung mass"

    Medical search terms:"""

        response = self.model.generate_content(prompt)
        return response.text.strip()


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