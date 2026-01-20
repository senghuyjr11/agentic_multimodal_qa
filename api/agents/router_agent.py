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
        prompt = f"""You are a routing agent for a Medical Visual Question Answering system. Analyze the user's message and decide what processing is needed.

        CONVERSATION HISTORY:
        {context}

        CURRENT MESSAGE:
        User: {message}
        Has image: {has_image}

        YOUR JOB:
        Decide which agents should process this message based on its intent and content.

        RESPONSE MODES:
        1. "casual_chat" - Conversational messages: greetings, thanks, acknowledgments, personal info, system questions
        2. "medical_answer" - Medical questions needing explanation, diagnosis help, or image analysis
        3. "modify_previous" - Requests to change the previous response (summarize, translate, remove parts)

        AVAILABLE AGENTS:
        - VQA (Visual Question Answering): Analyzes medical images
        - PubMed Search: Finds medical literature for explanations
        - Response Generator: Creates the final answer

        GUIDELINES:
        - Casual conversations (hi, thanks, good, who are you, language questions) → casual_chat, no agents needed
        - Medical images uploaded → medical_answer, use VQA + PubMed to explain findings
        - Medical questions without image → medical_answer, use PubMed for literature
        - Requests to modify previous response → modify_previous, no new agents needed

        IMPORTANT:
        - Images ALWAYS need VQA analysis
        - Medical terms/conditions usually benefit from PubMed literature
        - Casual responses don't need PubMed (even if they mention medical words in context)
        - When unsure, prefer being helpful over being restrictive

        OUTPUT FORMAT (JSON only):
        {{
          "needs_vqa": true/false,
          "needs_pubmed": true/false,
          "search_query": "medical search terms" or null,
          "response_mode": "medical_answer"|"casual_chat"|"modify_previous",
          "reasoning": "brief explanation of your decision"
        }}

        Examples:
        - "good response" → {{"needs_vqa": false, "needs_pubmed": false, "response_mode": "casual_chat", "search_query": null}}
        - "what is diabetes?" → {{"needs_vqa": false, "needs_pubmed": true, "response_mode": "medical_answer", "search_query": "diabetes"}}
        - [image uploaded] "what do you see?" → {{"needs_vqa": true, "needs_pubmed": true, "response_mode": "medical_answer", "search_query": null}}"""


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

        answer_lower = vqa_answer.lower().strip()

        # Step 1: Check for medical keywords first (HIGH PRIORITY)
        medical_keywords = [
            'infarct', 'fracture', 'lesion', 'tumor', 'mass', 'opacity',
            'pneumonia', 'carcinoma', 'infiltrate', 'edema', 'hemorrhage',
            'stenosis', 'occlusion', 'thrombosis', 'ischemia', 'necrosis',
            'hyperplasia', 'atrophy', 'hypertrophy', 'inflammation',
            'fibrosis', 'sclerosis', 'cirrhosis', 'emphysema', 'effusion',
            'nodule', 'cyst', 'abscess', 'aneurysm', 'embolism',
            'metastasis', 'lymphoma', 'melanoma', 'sarcoma', 'adenoma',
            'disease', 'syndrome', 'disorder', 'condition',
            'pneumothorax', 'pleural', 'consolidation', 'atelectasis'
        ]

        # If ANY medical keyword found, ALWAYS search PubMed
        if any(keyword in answer_lower for keyword in medical_keywords):
            return True

        # Step 2: Simple answers that don't need literature (ONLY if no medical keywords)
        import re
        simple_patterns = [
            r'^(yes|no|yeah|nope)\.?$',  # Yes/No
            r'^(normal|abnormal)\.?$',  # Normal/Abnormal
            r'^\d+\.?$',  # Just numbers
            r'^(left|right|upper|lower|anterior|posterior)\.?$',  # Simple locations only
            r'^(white|black|red|blue|green|yellow|gray|grey)\.?$',  # Colors
        ]

        for pattern in simple_patterns:
            if re.match(pattern, answer_lower, re.IGNORECASE):
                return False

        # Step 3: Check question type + answer length
        yes_no_questions = ['is this', 'is there', 'are there', 'does this', 'can you see']
        words = vqa_answer.split()

        # If it's a yes/no question AND answer is 1-2 words AND no medical keywords
        if len(words) <= 2 and any(q in original_question.lower() for q in yes_no_questions):
            return False

        # Step 4: Default - if answer is more than 1 word, likely needs explanation
        return len(words) >= 1

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

    # Test 3: Language question
    decision = router.decide(
        message="do you know Khmer?",
        has_image=False,
        memory=memory
    )
    print(f"\nTest 3 Result: {decision}")