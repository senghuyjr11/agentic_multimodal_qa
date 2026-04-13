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
        prompt = f"""You are a routing agent. Analyze the message and decide what's needed.

        CONVERSATION HISTORY:
        {context}

        CURRENT MESSAGE:
        User: {message}
        Has image: {has_image}

        DECISION TYPES:
        1. "casual_chat" - Greetings, thanks, small talk, system questions
        2. "medical_answer" - Medical questions needing explanation
        3. "modify_previous" - User wants to change the last response

        AGENTS AVAILABLE:
        - PubMed: Search medical literature

        RULES:
        - Casual chat (hi, thanks, good) → casual_chat, no PubMed
        - Medical questions → medical_answer, use PubMed
        - "translate", "summarize", "remove references", "make it shorter", "in English", "back to English" → modify_previous
        - A single language name (e.g. "Korean", "Chinese", "French", "khmer?") after a series of translations → modify_previous (user wants to translate to that language)
        - "make it back to eng/english" or "in English" after non-English responses → modify_previous

        OUTPUT (JSON only):
        {{
          "needs_pubmed": true/false,
          "search_query": "search terms" or null,
          "response_mode": "casual_chat"|"medical_answer"|"modify_previous",
          "reasoning": "why you decided this"
        }}

        Examples:
        - "hi" → {{"needs_pubmed": false, "response_mode": "casual_chat", "search_query": null, "reasoning": "greeting"}}
        - "what is diabetes?" → {{"needs_pubmed": true, "response_mode": "medical_answer", "search_query": "diabetes", "reasoning": "medical question"}}
        - "make it shorter" → {{"needs_pubmed": false, "response_mode": "modify_previous", "search_query": null, "reasoning": "modify request"}}
        - "Korean" (after previous translations) → {{"needs_pubmed": false, "response_mode": "modify_previous", "search_query": null, "reasoning": "user wants translation to Korean"}}
        - "back to English" → {{"needs_pubmed": false, "response_mode": "modify_previous", "search_query": null, "reasoning": "user wants English version"}}
        """


        # Ask LLM
        response = self.model.generate_content(prompt)
        text = response.text.strip()

        # Parse JSON
        import json
        import re

        text = re.sub(r'```json\s*|\s*```', '', text)
        decision_dict = json.loads(text)

        # Create decision object
        decision = RoutingDecision(
            needs_vqa=False,  # Will be set by hard rule below
            needs_pubmed=decision_dict.get("needs_pubmed", False),
            search_query=decision_dict.get("search_query"),
            response_mode=decision_dict.get("response_mode", "medical_answer"),
            reasoning=decision_dict.get("reasoning", "")
        )

        # HARD RULE: Image = VQA required
        if has_image:
            decision.needs_vqa = True
            if decision.response_mode == "casual_chat":
                decision.response_mode = "medical_answer"

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

        # Get last 2 messages (user question + AI answer)
        if len(memory.messages) < 2:
            return False, None

        last_user_msg = None
        last_ai_msg = None

        # Find last user and AI messages (LangChain: type == "ai" / "human")
        for msg in reversed(memory.messages):
            if msg.type == "ai" and last_ai_msg is None:
                last_ai_msg = msg.content
            elif msg.type == "human" and last_user_msg is None:
                last_user_msg = msg.content

            if last_user_msg and last_ai_msg:
                break

        if not last_ai_msg:
            return False, None

        message_lower = message.lower()

        # Patterns that indicate asking for explanation
        explanation_patterns = [
            "what does", "what is", "what are",
            "explain", "meaning", "definition",
            "why", "how", "tell me more",
            "what do you mean", "elaborate",
            "can you explain", "don't understand"
        ]

        # Check if current message is asking for explanation
        is_asking_explanation = any(pattern in message_lower for pattern in explanation_patterns)

        if not is_asking_explanation:
            return False, None

        search_query = self._build_followup_search_query(last_ai_msg)

        if not search_query:
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

        prompt = f"""You convert a medical explanation into a short PubMed keyword query.

SOURCE TEXT:
{last_ai_msg}

Rules:
- Return ONLY keywords, no full sentence
- Prefer diagnosis names, pathology/radiology terms, organ names, and core disease concepts
- Remove filler words and descriptive prose
- Keep it between 2 and 6 keywords
- No punctuation except spaces
- No markdown

Examples:
- "The image demonstrates a heart with a notable dilation of the left ventricle and aortic root." -> left ventricle dilation aortic root dilation
- "This may be a benign mixed mesodermal tumor also known as a teratoma." -> benign mixed mesodermal tumor teratoma
- "Features are consistent with constrictive pericarditis." -> constrictive pericarditis

Keyword query:"""

        try:
            response = self.model.generate_content(prompt)
            query = (response.text or "").strip()

            import re

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

        import re

        cleaned = re.sub(r'\s+', ' ', last_ai_msg).strip()
        cleaned = re.sub(r'\*+', '', cleaned)
        cleaned = re.sub(r'\[[^\]]+\]\([^)]+\)', '', cleaned).strip()

        if not cleaned:
            return None

        patterns = [
            r'consistent with\s+(?:a|an)?\s*([^.;:()]+)',
            r'likely (?:a|an)?\s*([^.;:()]+)',
            r'may be\s+(?:a|an)?\s*([^.;:()]+)',
            r'suggests (?:this may be|it may be)?\s*(?:a|an)?\s*([^.;:()]+)',
            r'also referred to as\s+(?:a|an)?\s*([^.;:()]+)',
            r'also known as\s+(?:a|an)?\s*([^.;:()]+)',
        ]

        extracted_chunks = []
        for pattern in patterns:
            matches = re.findall(pattern, cleaned, flags=re.IGNORECASE)
            for match in matches:
                chunk = re.sub(r'\s+', ' ', match).strip(" .,:;!-")
                if chunk and chunk.lower() not in [c.lower() for c in extracted_chunks]:
                    extracted_chunks.append(chunk)

        def to_keyword_query(text: str) -> Optional[str]:
            stopwords = {
                "the", "a", "an", "and", "or", "with", "without", "of", "to",
                "in", "on", "for", "from", "by", "this", "that", "these", "those",
                "image", "shows", "showing", "reveals", "revealing", "exhibiting",
                "appearance", "surface", "presence", "including", "includes",
                "characteristic", "characterized", "consistent", "likely", "suggests",
                "suggesting", "also", "known", "referred", "referredto", "may", "be",
                "due", "large", "irregular", "fleshy", "multicolored", "lobulated"
            }

            tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", text.lower())
            keywords = []

            for token in tokens:
                normalized = token.replace("-", "")
                if normalized in stopwords:
                    continue
                if len(normalized) < 3:
                    continue
                if normalized not in keywords:
                    keywords.append(normalized)

            if not keywords:
                return None

            return " ".join(keywords[:6])

        if extracted_chunks:
            query = to_keyword_query(" ".join(extracted_chunks[:2]))
            if query:
                return query

        # Fallback: prefer the first sentence when no diagnosis phrase is found.
        first_sentence = re.split(r'(?<=[.!?])\s+', cleaned, maxsplit=1)[0].strip()
        candidate = first_sentence or cleaned
        query = to_keyword_query(candidate)
        return query or candidate.strip(" .,:;!-") or None

    def is_asking_about_previous_references(self, message: str) -> bool:
        """
        Detect if user is asking to explain/elaborate on previous response.

        Patterns:
        - "explain from those resources"
        - "tell me more about that"
        - "elaborate"
        - "can you explain more"
        """
        message_lower = message.lower()

        patterns = [
            "explain", "tell me more", "elaborate", "expand",
            "those resources", "those articles", "those references",
            "from that", "based on that", "using that",
            "go deeper", "more detail", "in detail",
            "from the", "using the", "based on the",
            # NEW - more natural patterns:
            "can you explain", "could you explain",
            "tell me about", "what about",
            "dive deeper", "break it down",
            "give me more", "say more"
        ]

        return any(pattern in message_lower for pattern in patterns)


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
