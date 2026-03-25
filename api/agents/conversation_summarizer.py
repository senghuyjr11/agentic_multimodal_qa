"""
conversation_summarizer.py - Summarizes older conversation turns

ONE JOB: Compress older conversation turns into a rolling summary.
"""

from typing import Iterable, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class ConversationSummarizer:
    """
    Builds a rolling summary for long-running conversations.

    Falls back to a deterministic summary if the LLM call fails.
    """

    def __init__(self, google_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemma-3-4b-it",
            google_api_key=google_api_key
        )
        self.parser = StrOutputParser()

    def summarize(
        self,
        existing_summary: Optional[str],
        turns: Iterable[dict]
    ) -> str:
        turns = list(turns)
        if not turns:
            return existing_summary or ""

        transcript_lines = []
        for idx, turn in enumerate(turns, 1):
            user_text = turn.get("user", "").strip()
            assistant_text = turn.get("assistant", "").strip()

            if user_text:
                transcript_lines.append(f"Turn {idx} User: {user_text}")
            if assistant_text:
                transcript_lines.append(f"Turn {idx} Assistant: {assistant_text}")

        transcript = "\n".join(transcript_lines)

        prompt = ChatPromptTemplate.from_template(
            """You maintain compressed memory for a medical QA assistant.

EXISTING SUMMARY:
{existing_summary}

OLDER CONVERSATION TO COMPRESS:
{transcript}

Write an updated rolling summary that preserves:
- the user's goals and preferences
- important medical topics or images discussed
- prior explanations, translations, or follow-up requests
- unresolved questions or pending threads

Rules:
- keep it concise and factual
- do not invent details
- prefer compact bullets
- keep medically relevant details if they may matter later

UPDATED SUMMARY:"""
        )

        chain = prompt | self.llm | self.parser

        try:
            summary = chain.invoke({
                "existing_summary": existing_summary or "(none)",
                "transcript": transcript
            }).strip()

            return summary or self._fallback(existing_summary, turns)
        except Exception as e:
            print(f"[Summary] LLM summarization failed: {e}")
            return self._fallback(existing_summary, turns)

    def _fallback(
        self,
        existing_summary: Optional[str],
        turns: list[dict]
    ) -> str:
        snippets = []
        if existing_summary:
            snippets.append(existing_summary.strip())

        snippets.append("Older conversation summary:")

        for turn in turns[-6:]:
            user_text = turn.get("user", "").strip()
            assistant_text = turn.get("assistant", "").strip()

            if user_text:
                snippets.append(f"- User asked: {user_text[:180]}")
            if assistant_text:
                snippets.append(f"- Assistant answered: {assistant_text[:220]}")

        return "\n".join(snippets).strip()
