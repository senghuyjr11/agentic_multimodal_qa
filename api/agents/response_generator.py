"""
response_generator.py - Generates responses ONLY

ONE JOB: Create the final response text

NO routing decisions - that's router_agent.py's job
"""

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
            model="gemma-3-4b-it",
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

        # Get conversation context so LLM understands references like
        # "make it back to eng" or "the article we just talked about"
        context = self._get_context(memory, num_turns=10) if memory else ""

        # Check if it's a translation request
        translation_keywords = [
            'translate', 'explain in', 'in spanish', 'in french',
            'in chinese', 'in korean', 'in japanese', 'in vietnamese',
            'in english', 'back to english', 'back to eng',
            'in arabic', 'in hindi', 'in german', 'in russian',
        ]

        is_translation_request = any(
            keyword in message.lower()
            for keyword in translation_keywords
        )

        if is_translation_request:
            prompt = ChatPromptTemplate.from_template(
                """You are a medical assistant helping modify a previous response.

CONVERSATION HISTORY:
{context}

MOST RECENT RESPONSE:
{previous_response}

USER REQUEST:
{message}

Task: Translate or rephrase the response the user is referring to into the requested language.
Use the conversation history to identify which response they mean if it's not clear.
Keep the same meaning and medical accuracy.
If references were included, keep them in the same format.

TRANSLATED/REPHRASED RESPONSE:"""
            )
        else:
            prompt = ChatPromptTemplate.from_template(
                """You are a medical assistant helping modify a previous response.

CONVERSATION HISTORY:
{context}

MOST RECENT RESPONSE:
{previous_response}

USER REQUEST:
{message}

Use the conversation history to understand what the user is referring to.
Common modifications:
- "remove references" → remove reference section
- "make shorter" / "summarize" → condense to key points
- "simplify" → use simpler language
- "back to English" / "in English" → translate the most recent non-English response back to English

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
                """You are a medical assistant. A vision model analyzed a medical image and produced the following raw answer.

RAW VQA ANSWER: {vqa_answer}

USER QUESTION: {message}

Rewrite the answer as a clear, properly formatted medical response:
- Use correct capitalization and punctuation
- Write in complete sentences
- Keep it concise but informative
- Do not add information not present in the raw answer
- Do not reference any literature or add citations

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

            if is_elaboration:
                prompt = ChatPromptTemplate.from_template(
                    """You are a medical assistant. The user wants MORE DETAIL about the previous response.

CONVERSATION HISTORY:
{context}

CURRENT REQUEST:
{message}

MEDICAL LITERATURE:
{lit_section}

Task:
1. Provide a MORE DETAILED explanation using the literature
2. Go deeper into the mechanisms
3. Explain the biological/chemical basis
4. Include specific details from the articles
5. {citation_instruction}
6. Keep it concise and educational

Detailed Response:"""
                )
            else:
                prompt = ChatPromptTemplate.from_template(
                    """You are a medical assistant.

{context_section}

CURRENT QUESTION:
{message}

{lit_section}

Answer the question using the medical literature provided.
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
                    """You are a medical assistant helping the user understand a previous answer.

CONVERSATION HISTORY:
{context}

MOST RECENT MEDICAL ANSWER:
{previous_response}

USER REQUEST:
{message}

Task:
- Explain the previous medical answer more clearly using the conversation context
- Stay consistent with what was already said
- Do not invent new diagnoses or claims
- If the previous answer is uncertain, keep that uncertainty
- Do not add citations or references unless they are explicitly provided

Clear Explanation:"""
                )
                chain = prompt | self.llm | self.parser
                return chain.invoke({
                    "context": context,
                    "previous_response": previous_response,
                    "message": message
                })

        # SCENARIO 3: No data available
        return "I don't have enough information to answer this question. Could you provide more details or upload a medical image?"

    def _is_followup_explanation_request(self, message: str) -> bool:
        message_lower = message.lower()
        patterns = [
            "explain", "more detail", "more details", "tell me more",
            "elaborate", "clearer", "explain clearly", "what do you mean",
            "break it down", "help me understand"
        ]
        return any(pattern in message_lower for pattern in patterns)

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

Previous conversation:
{context}

User: {message}

Instructions:
- Answer naturally and warmly (1-3 sentences)
- If the user asks "what's my name", check the conversation history above
- If the user asks about their first question, check the conversation history above
- If the user asks to remember something, refer to the conversation history
- Be conversational and helpful
- NO citations or references in casual chat

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

        # Detect memory questions
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
            """You are a helpful assistant. Answer the user's question by looking at the conversation history.

FULL CONVERSATION HISTORY:
{context}

USER QUESTION: {message}

Instructions:
- If they ask about their name, look for where they introduced themselves
- If they ask about their first question, find the first User: message
- If they ask "do you remember X", check if X appears in the history
- Answer directly and concisely
- If you can't find the information, say "I don't see that in our conversation history"

Answer:"""
        )

        chain = prompt | self.llm | self.parser
        return chain.invoke({"context": context, "message": message})


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
