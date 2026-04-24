"""
response_generator.py - Generates responses ONLY

ONE JOB: Create the final response text

NO routing decisions - that's router_agent.py's job
"""

import re
from typing import Optional, List
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import from our simple memory manager (relative import)
from .memory_manager import InMemoryConversation


@dataclass
class Article:
    """PubMed article"""
    title: str
    abstract: str
    pmid: str
    url: str
    relevance_score: Optional[float] = None


class ResponseGenerator:
    """
    Generates the final response text.

    Does NOT make routing decisions - only writes responses.
    """

    def __init__(self, google_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=google_api_key
        )
        self.parser = StrOutputParser()

    def generate(
            self,
            message: str,
            response_mode: str,
            vqa_answer: Optional[str] = None,
            pubmed_articles: Optional[List[Article]] = None,
            memory: Optional[InMemoryConversation] = None,
            previous_response: Optional[str] = None,
            has_image: bool = False
    ) -> str:
        """Generate response text"""

        # NEW: Check if this is a memory question first
        if memory and response_mode == "casual_chat":
            memory_answer = self._handle_memory_question(message, memory)
            if memory_answer:
                return memory_answer

        # Original logic
        if response_mode == "casual_chat":
            return self._generate_casual(message, memory)

        elif response_mode == "modify_previous":
            return self._generate_modification(message, previous_response, memory)

        else:  # medical_answer
            return self._generate_medical(
                message,
                vqa_answer,
                pubmed_articles,
                memory,
                has_image
            )

    def _generate_modification(
        self,
        message: str,
        previous_response: Optional[str],
        memory: Optional[InMemoryConversation] = None
    ) -> str:
        """Modify previous response based on instruction"""

        if not previous_response:
            return "I don't have a previous response to modify."

        context = self._get_context(memory, num_turns=10) if memory else ""
        intent = self._classify_modification_intent(message, context, previous_response)

        if intent == "reference":
            prompt = ChatPromptTemplate.from_template(
                """You are a medical assistant.

CONVERSATION HISTORY:
{context}

MOST RECENT RESPONSE:
{previous_response}

USER REQUEST:
{message}

Task: answer only about references from the previous response.
- If references exist, restate them clearly.
- If not, say no references were included.
- Do not invent references.

Response:"""
            )

        elif intent == "translate":
            prompt = ChatPromptTemplate.from_template(
                """You are a medical assistant.

CONVERSATION HISTORY:
{context}

MOST RECENT RESPONSE:
{previous_response}

USER REQUEST:
{message}

Task: translate/rephrase the intended previous response.
- Preserve medical meaning.
- Keep references format if references exist.
- Use history to resolve ambiguous requests.

TRANSLATED/REPHRASED RESPONSE:"""
            )
        else:
            prompt = ChatPromptTemplate.from_template(
                """You are a medical assistant.

CONVERSATION HISTORY:
{context}

MOST RECENT RESPONSE:
{previous_response}

USER REQUEST:
{message}

Task: apply the user's requested edit to the relevant previous response.
- Follow the request exactly (shorten/simplify/remove references/rewrite).
- Keep medical meaning unchanged.

MODIFIED RESPONSE:"""
            )

        chain = prompt | self.llm | self.parser
        return chain.invoke({
            "context": context,
            "previous_response": previous_response,
            "message": message
        })

    def _generate_medical(
            self,
            message: str,
            vqa_answer: Optional[str],
            pubmed_articles: Optional[List[Article]],
            memory: Optional[InMemoryConversation],
            has_image: bool
    ) -> str:
        """Generate medical answer with literature"""

        # SCENARIO 1: Image with VQA answer - format and return
        if has_image and vqa_answer:
            prompt = ChatPromptTemplate.from_template(
                """You are a medical assistant.

RAW VQA ANSWER: {vqa_answer}
USER QUESTION: {message}

Rewrite into a clear short answer.
- Keep only information from RAW VQA ANSWER.
- Do not add new facts or citations.

Formatted Response:"""
            )
            chain = prompt | self.llm | self.parser
            return chain.invoke({"vqa_answer": vqa_answer, "message": message})

        # SCENARIO 2: Text question with PubMed articles
        if pubmed_articles:
            context = self._get_context(memory, num_turns=10) if memory else ""

            lit_lines = ["Medical Literature:"]
            for i, article in enumerate(pubmed_articles[:5], 1):
                lit_lines.append(f"[{i}] {article.title}")
                lit_lines.append(f"    {article.abstract[:300]}...")
            lit_section = "\n".join(lit_lines)

            # Check if user wants to remove/hide references
            message_lower = message.lower()
            hide_references = any(word in message_lower for word in
                                  ["don't give reference", "no reference", "without reference",
                                   "remove reference", "hide reference", "skip reference"])

            # Check if user is asking for elaboration
            is_elaboration = any(word in message_lower for word in
                                 ["explain", "elaborate", "tell me more", "detail", "those resources"])
            question_style = self._classify_medical_question_style(message)

            if is_elaboration:
                prompt = ChatPromptTemplate.from_template(
                    """You are a medical assistant.

CONVERSATION HISTORY:
{context}

CURRENT REQUEST:
{message}

MEDICAL LITERATURE:
{lit_section}

Task:
- Give a deeper explanation using only the provided literature.
- If evidence is weak/indirect, say so clearly.
- Be conservative; avoid unsupported certainty.
- {citation_instruction}
- Use this structure only:
Summary:
- 1-2 bullets
Why It Matters:
- 2-4 bullets
Key Evidence:
- 2-4 bullets

Detailed Response:"""
                )
            else:
                prompt = ChatPromptTemplate.from_template(
                    """You are a medical assistant.

{context_section}

CURRENT QUESTION:
{message}

{lit_section}

Answer using only the provided literature.
- First sentence: direct answer to the user question.
- If evidence is weak/indirect, state that clearly.
- Do not overclaim beyond evidence.
- Keep answer focused on question style: {question_style}
{citation_instruction}

Response:"""
                )

            citation_instruction = (
                "Do NOT cite sources. Do NOT include a References or Bibliography section."
                if hide_references else
                "Use inline citations [1][2][3] only. Do NOT generate a References or Bibliography section at the end — references will be added automatically."
            )

            chain = prompt | self.llm | self.parser
            answer = chain.invoke({
                "context": context,
                "context_section": f"CONVERSATION HISTORY:\n{context}" if context else "",
                "message": message,
                "lit_section": lit_section,
                "citation_instruction": citation_instruction,
                "question_style": question_style,
            })

            # Strip any LLM-generated references section to avoid duplication
            import re
            answer = re.sub(
                r'\n{0,2}\*{0,2}(References|Bibliography|Sources)\*{0,2}:?.*',
                '', answer, flags=re.IGNORECASE | re.DOTALL
            ).rstrip()

            # Add references ONLY if user didn't ask to hide them
            if not hide_references:
                answer += "\n\n**References:**"
                for i, article in enumerate(pubmed_articles[:5], 1):
                    score_text = ""
                    if hasattr(article, 'relevance_score') and article.relevance_score:
                        score_pct = int(article.relevance_score * 100)
                        score_text = f" (relevance: {score_pct}%)"

                    title = article.title[:80] + "..." if len(article.title) > 80 else article.title
                    answer += f"\n{i}. [{title}]({article.url}){score_text}"

            return answer

        # SCENARIO 3: Follow-up explanation using conversation memory only
        if memory and self._is_followup_explanation_request(message):
            previous_response = self._get_last_ai_message(memory)
            context = self._get_context(memory, num_turns=10)

            if previous_response:
                prompt = ChatPromptTemplate.from_template(
                    """You are a medical assistant.

CONVERSATION HISTORY:
{context}

MOST RECENT MEDICAL ANSWER:
{previous_response}

USER REQUEST:
{message}

Explain the previous answer more clearly.
- Stay consistent with prior content.
- Keep uncertainty if uncertain.
- No new references.
- Use this structure only:
Summary:
- 1-2 bullets
Key Findings:
- 2-4 bullets
Plain Explanation:
- 2-4 bullets
- Keep under 140 words.

Clear Explanation:"""
                )
                chain = prompt | self.llm | self.parser
                return chain.invoke({
                    "context": context,
                    "previous_response": previous_response,
                    "message": message
                })

        # SCENARIO 4: Topic-aware follow-up using memory when no fresh literature is available
        if memory and self._is_topic_aware_followup(message):
            previous_response = self._get_last_ai_message(memory)
            context = self._get_context(memory, num_turns=10)
            question_style = self._classify_medical_question_style(message)

            if previous_response:
                prompt = ChatPromptTemplate.from_template(
                    """You are a medical assistant.

CONVERSATION HISTORY:
{context}

MOST RECENT MEDICAL ANSWER:
{previous_response}

CURRENT QUESTION:
{message}

Answer the follow-up using conversation context.
- First sentence: direct answer.
- Stay consistent with prior discussion.
- If uncertain, say uncertain; do not overstate.
- No invented references/citations.
- Keep concise.
- Style focus: {question_style}

Response:"""
                )
                chain = prompt | self.llm | self.parser
                return chain.invoke({
                    "context": context,
                    "previous_response": previous_response,
                    "message": message,
                    "question_style": question_style,
                })

        # SCENARIO 5: No data available
        return "I don't have enough information to answer this question. Could you provide more details or upload a medical image?"

    def _is_followup_explanation_request(self, message: str) -> bool:
        prompt = f"""Classify this user message.

Message: {message}

Return JSON only:
{{
  "is_followup_explanation": true | false
}}

true only when user asks to explain/clarify a previous answer."""
        try:
            parsed = self._llm_json(prompt)
            if "is_followup_explanation" in parsed:
                return bool(parsed.get("is_followup_explanation"))
        except Exception:
            pass
        message_lower = message.lower()
        return any(
            pattern in message_lower
            for pattern in ["explain", "more detail", "tell me more", "elaborate", "clearer"]
        )

    def _is_topic_aware_followup(self, message: str) -> bool:
        prompt = f"""Classify this user message.

Message: {message}

Return JSON only:
{{
  "is_topic_followup": true | false
}}

true when it is a follow-up medical question that depends on prior conversation topic."""
        try:
            parsed = self._llm_json(prompt)
            if "is_topic_followup" in parsed:
                return bool(parsed.get("is_topic_followup"))
        except Exception:
            pass
        message_lower = message.lower().strip()
        return any(
            pattern in message_lower
            for pattern in ["diet", "treatment", "symptom", "prognosis", "risk", "recovery", "follow up"]
        )

    def _classify_medical_question_style(self, message: str) -> str:
        prompt = f"""Classify medical question style.

Message: {message}

Return JSON only:
{{
  "style": "severity" | "treatment" | "diet" | "healing_time" | "definition" | "general_medical"
}}"""
        try:
            parsed = self._llm_json(prompt)
            style = (parsed.get("style") or "").strip().lower()
            allowed = {"severity", "treatment", "diet", "healing_time", "definition", "general_medical"}
            if style in allowed:
                return style
        except Exception:
            pass
        return "general_medical"

    def _get_last_ai_message(self, memory: InMemoryConversation) -> Optional[str]:
        for msg in reversed(memory.messages):
            if msg.type == "ai":
                return msg.content
        return None

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
                # Don't truncate - LLM needs full context to remember details
                context_lines.append(f"Assistant: {msg.content}")

        return "\n".join(context_lines) if context_lines else "(No previous conversation)"

    def _get_context_for_memory(
            self,
            memory: InMemoryConversation,
            num_turns: int
    ) -> str:
        """
        Build context for memory questions.
        Keeps user messages full but truncates AI responses (50 chars)
        to prevent context overflow when answers are very long (Korean, Chinese, etc.)
        """
        messages = memory.messages[-(num_turns * 2):] if memory.messages else []

        context_lines = []
        if memory.rolling_summary:
            context_lines.append("Earlier conversation summary:")
            context_lines.append(memory.rolling_summary[:1000])
            context_lines.append("")

        for msg in messages:
            if msg.type == "human":
                context_lines.append(f"User: {msg.content}")
            elif msg.type == "ai":
                truncated = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
                context_lines.append(f"Assistant: {truncated}")

        return "\n".join(context_lines) if context_lines else "(No previous conversation)"

    def _generate_casual(
            self,
            message: str,
            memory: Optional[InMemoryConversation]
    ) -> str:
        """Generate casual conversational response"""

        # Get MORE context for casual chat (up to 10 turns)
        context = self._get_context(memory, num_turns=10) if memory else ""

        prompt = ChatPromptTemplate.from_template(
            """You are a friendly Medical VQA Assistant.

Conversation:
{context}

User: {message}

Instructions:
- Reply warmly in 1-3 sentences.
- Use conversation history for memory/name questions.
- No citations or references.

Response:"""
        )

        chain = prompt | self.llm | self.parser
        text = chain.invoke({"context": context, "message": message})

        # Clean up excessive greetings
        import re
        text = re.sub(r'^(Hi|Hello|Hey),?\s+\w+[,!]?\s+', '', text, flags=re.IGNORECASE)

        return text

    def _handle_memory_question(
            self,
            message: str,
            memory: InMemoryConversation
    ) -> Optional[str]:
        """
        Handle questions about conversation history.
        Returns answer if it's a memory question, None otherwise.
        """

        message_lower = message.lower()

        # 1) Deterministic handling for name-introduction statements.
        # This avoids the "I don't see that..." failure when user just told us their name.
        introduced_name = self._extract_name_from_message(message)
        if introduced_name:
            return f"Nice to meet you, {introduced_name}. I will remember your name in this session."

        # 2) Deterministic handling for name recall questions.
        if self._is_name_question(message_lower):
            remembered_name = self._find_latest_name_in_memory(memory)
            if remembered_name:
                return f"Your name is {remembered_name}."
            return "I don't see your name in our conversation history yet."

        # 3) Generic memory-question handling via LLM prompt.
        memory_patterns = [
            "my name", "what's my name", "do you know my name",
            "my first question", "first question", "what did i ask",
            "do you remember", "what was my", "recall"
        ]

        is_memory_question = any(pattern in message_lower for pattern in memory_patterns)

        if not is_memory_question:
            return None

        # Get conversation history — truncate AI responses to avoid overflowing context window
        # (AI medical responses can be thousands of words; user messages are what matter here)
        context = self._get_context_for_memory(memory, num_turns=50)

        prompt = ChatPromptTemplate.from_template(
            """Answer the user using conversation history only.

Conversation:
{context}

USER QUESTION: {message}

Instructions:
- Answer directly and concisely.
- If info is missing, say: "I don't see that in our conversation history."

Answer:"""
        )

        chain = prompt | self.llm | self.parser
        return chain.invoke({"context": context, "message": message})

    def _classify_modification_intent(
        self,
        message: str,
        context: str,
        previous_response: str,
    ) -> str:
        prompt = f"""Classify user intent for editing previous response.

Conversation:
{context}

Previous response:
{previous_response}

User request:
{message}

Return JSON only:
{{
  "intent": "reference" | "translate" | "edit_general"
}}

- reference: asks about prior references/articles/sources
- translate: asks language change/translate/back to English
- edit_general: shorten/simplify/rewrite/remove sections"""
        try:
            parsed = self._llm_json(prompt)
            intent = (parsed.get("intent") or "").strip().lower()
            if intent in {"reference", "translate", "edit_general"}:
                return intent
        except Exception:
            pass

        lowered = message.lower()
        if "reference" in lowered or "sources" in lowered:
            return "reference"
        if "translate" in lowered or "english" in lowered:
            return "translate"
        return "edit_general"

    def _llm_json(self, prompt: str) -> dict:
        raw = self.llm.invoke(prompt)
        text = str(raw.content if hasattr(raw, "content") else raw).strip()
        cleaned = re.sub(r"```json\s*|\s*```", "", text, flags=re.IGNORECASE).strip()
        try:
            import json
            return json.loads(cleaned)
        except Exception:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                try:
                    import json
                    return json.loads(match.group(0))
                except Exception:
                    return {}
        return {}

    @staticmethod
    def _is_name_question(message_lower: str) -> bool:
        patterns = [
            "what is my name",
            "what's my name",
            "what was my name",
            "do you know my name",
            "remember my name",
            "my name?",
        ]
        return any(pattern in message_lower for pattern in patterns)

    @staticmethod
    def _extract_name_from_message(message: str) -> Optional[str]:
        text = (message or "").strip()
        if not text:
            return None

        patterns = [
            r"\bmy name is\s+([A-Za-z][A-Za-z\-']{0,39})\b",
            r"\bi am\s+([A-Za-z][A-Za-z\-']{0,39})\b",
            r"\bi'm\s+([A-Za-z][A-Za-z\-']{0,39})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                return name[:1].upper() + name[1:]
        return None

    def _find_latest_name_in_memory(self, memory: InMemoryConversation) -> Optional[str]:
        for msg in reversed(memory.messages):
            if msg.type != "human":
                continue
            extracted = self._extract_name_from_message(getattr(msg, "content", "") or "")
            if extracted:
                return extracted
        return None


if __name__ == "__main__":
    import os

    # Test
    generator = ResponseGenerator(google_api_key=os.getenv("GOOGLE_API_KEY"))

    # Test 1: Medical response
    response = generator.generate(
        message="What is diabetes?",
        response_mode="medical_answer",
        pubmed_articles=None
    )
    print(f"Test 1 (Medical):\n{response}\n")

    # Test 2: Casual response
    response = generator.generate(
        message="Hello!",
        response_mode="casual_chat"
    )
    print(f"Test 2 (Casual):\n{response}\n")
