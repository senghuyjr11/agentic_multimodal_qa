"""
response_generator.py - Generates responses ONLY

ONE JOB: Create the final response text

NO routing decisions - that's router_agent.py's job
"""

import google.generativeai as genai
from typing import Optional, List
from dataclasses import dataclass

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
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel("gemma-3-4b-it")

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
        """
        ONE JOB: Generate response text.

        Args:
            message: User's question
            response_mode: "medical_answer" | "casual_chat" | "modify_previous"
            vqa_answer: Answer from image agent (if any)
            pubmed_articles: Articles from PubMed (if any)
            memory: Conversation history
            previous_response: Last response (for modify mode)
            has_image: Was there an image?

        Returns:
            Final response text
        """

        if response_mode == "casual_chat":
            return self._generate_casual(message, memory)

        elif response_mode == "modify_previous":
            return self._generate_modification(message, previous_response)

        else:  # medical_answer
            return self._generate_medical(
                message,
                vqa_answer,
                pubmed_articles,
                memory,
                has_image
            )

    def _generate_casual(
        self,
        message: str,
        memory: Optional[InMemoryConversation]
    ) -> str:
        """Generate casual conversational response"""

        context = self._get_context(memory, num_turns=3) if memory else ""

        prompt = f"""You are a friendly Medical VQA Assistant.

{f"Previous conversation:{context}" if context else ""}

User: {message}

Respond naturally and warmly. Guidelines:
- Brief (1-3 sentences)
- Don't mention the user's name unless they ask "what's my name"
- NO citations or references
- If asked about their name, check conversation history
- If asked why you give references, explain it's for medical accuracy

Response:"""

        response = self.model.generate_content(prompt)
        text = response.text.strip()

        # Clean up excessive greetings
        import re
        text = re.sub(r'^(Hi|Hello|Hey),?\s+\w+[,!]?\s+', '', text, flags=re.IGNORECASE)

        return text

    def _generate_modification(
        self,
        message: str,
        previous_response: Optional[str]
    ) -> str:
        """Modify previous response based on instruction"""

        if not previous_response:
            return "I don't have a previous response to modify."

        # Check if it's a translation request
        translation_keywords = [
            'translate', 'explain in', 'in khmer', 'in spanish', 'in french',
            'in chinese', 'in korean', 'in japanese', 'in thai', 'in vietnamese',
            'របកប្រែ', 'ពន្យល់ជាភាសា'  # Khmer keywords
        ]

        is_translation_request = any(
            keyword in message.lower()
            for keyword in translation_keywords
        )

        if is_translation_request:
            # Extract target language from message
            # Examples: "translate to Spanish", "explain in Khmer", "in French"
            prompt = f"""The user wants the previous response translated or explained in another language.

PREVIOUS RESPONSE:
{previous_response}

USER REQUEST:
{message}

Task: Translate or rephrase the previous response into the requested language.
Keep the same meaning and medical accuracy.
If references were included, keep them in the same format.

TRANSLATED/REPHRASED RESPONSE:"""
        else:
            # Regular modification (remove refs, make shorter, etc.)
            prompt = f"""Modify this response based on the user's request.

PREVIOUS RESPONSE:
{previous_response}

USER REQUEST:
{message}

Common modifications:
- "remove references" → remove reference section
- "make shorter" → condense to key points
- "simplify" → use simpler language

MODIFIED RESPONSE:"""

        response = self.model.generate_content(prompt)
        return response.text.strip()

    def _generate_medical(
        self,
        message: str,
        vqa_answer: Optional[str],
        pubmed_articles: Optional[List[Article]],
        memory: Optional[InMemoryConversation],
        has_image: bool
    ) -> str:
        """Generate medical answer with literature"""

        context = self._get_context(memory, num_turns=3) if memory else ""

        # Build literature section
        lit_section = "No medical literature available."
        if pubmed_articles:
            lit_lines = ["Medical Literature:"]
            for i, article in enumerate(pubmed_articles[:5], 1):
                lit_lines.append(f"[{i}] {article.title}")
                lit_lines.append(f"    {article.abstract[:300]}...")
            lit_section = "\n".join(lit_lines)

        # Build prompt
        prompt = f"""You are a medical assistant.

{f"CONVERSATION HISTORY:{context}" if context else ""}

CURRENT QUESTION:
{message}

{f"IMAGE ANALYSIS:{vqa_answer}" if vqa_answer else ""}

{lit_section}

Generate a clear medical answer:
- Answer directly (1-2 sentences)
- Explain with key points
- Use literature to support [1][2][3]
- If image analysis provided, DON'T repeat it (shown separately)
- Be concise but thorough

Response:"""

        response = self.model.generate_content(prompt)
        answer = response.text.strip()

        # Add VQA detection line if image
        if has_image and vqa_answer:
            answer = f"VQA Detection: {vqa_answer}\n\n{answer}"

        # Add formatted references
        if pubmed_articles:
            answer += "\n\n---\n\n**References:**"
            for i, article in enumerate(pubmed_articles[:5], 1):
                score_text = ""
                if hasattr(article, 'relevance_score') and article.relevance_score:
                    score_pct = int(article.relevance_score * 100)
                    score_text = f" (relevance: {score_pct}%)"

                title = article.title[:80] + "..." if len(article.title) > 80 else article.title
                answer += f"\n{i}. [{title}]({article.url}){score_text}"

        return answer

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
                context_lines.append(f"\nUser: {msg.content}")
            elif msg.role == "assistant":
                context_lines.append(f"Assistant: {msg.content}")

        return "\n".join(context_lines) if context_lines else ""


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