"""
text_only_agent.py - Handles text-only questions (no image)
Simplified version - English only, translation handled by TranslationAgent
For medical questions: uses PubMed + reasoning, NOT direct model answers
"""
import google.generativeai as genai


class TextOnlyAgent:
    """Routes text-only questions.
    - Casual questions: direct response
    - Medical questions: flag for PubMed search (no direct medical answers)
    """

    CASUAL_KEYWORDS = [
        "hi", "hello", "hey", "who are you", "what are you",
        "how are you", "thank you", "thanks", "bye", "goodbye",
        "help", "what can you do", "your name"
    ]

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemma-3-4b-it")

    def classify(self, question: str) -> str:
        """Classify question type: 'casual' or 'medical'."""
        question_lower = question.lower().strip()
        for keyword in self.CASUAL_KEYWORDS:
            if keyword in question_lower:
                return "casual"
        return "medical"

    def respond(self, question: str) -> dict:
        """Route question based on type.

        Args:
            question: Question in English (already translated by TranslationAgent)

        Returns:
            dict with question_type and response (only for casual questions)
            For medical questions, returns flag to use PubMed pipeline
        """
        question_type = self.classify(question)

        print(f"Question type: {question_type}")

        if question_type == "casual":
            # Only casual questions get direct responses
            prompt = f"""You are a friendly Medical VQA Assistant. Answer this simple question briefly and naturally.

Question: {question}

Keep your response short and friendly. If asked who you are, mention you help analyze medical images and answer health-related questions."""

            response = self.model.generate_content(prompt)

            return {
                "question": question,
                "question_type": "casual",
                "response": response.text,
                "needs_pubmed": False
            }

        else:  # medical
            # Medical questions MUST go through PubMed + Reasoning pipeline
            print("⚠️ Medical question detected - routing to PubMed pipeline")

            return {
                "question": question,
                "question_type": "medical",
                "response": None,  # No direct answer
                "needs_pubmed": True,  # Flag: MUST use PubMed + Reasoning
                "message": "Medical question - requires PubMed literature search"
            }