"""
text_only_agent.py - Handles text-only questions (no image)
"""
import google.generativeai as genai


class TextOnlyAgent:
    """Routes and handles text-only questions."""

    CASUAL_KEYWORDS = [
        "hi", "hello", "hey", "who are you", "what are you",
        "how are you", "thank you", "thanks", "bye", "goodbye",
        "help", "what can you do", "your name"
    ]

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model_fast = genai.GenerativeModel("gemma-3-4b-it")
        self.model_multilang = genai.GenerativeModel("gemma-3-12b-it")

    def _get_model(self, language: str):
        if language.lower() == "english":
            return self.model_fast, "gemma-3-4b-it"
        return self.model_multilang, "gemma-3-12b-it"

    def classify(self, question: str) -> str:
        """Classify question type: 'casual' or 'medical'."""
        question_lower = question.lower().strip()
        for keyword in self.CASUAL_KEYWORDS:
            if keyword in question_lower:
                return "casual"
        return "medical"

    def respond(self, question: str, language: str = "English") -> dict:
        """Generate response based on question type."""
        question_type = self.classify(question)
        model, model_name = self._get_model(language)

        print(f"Question type: {question_type}")
        print(f"Using model: {model_name}")

        if question_type == "casual":
            prompt = f"""You are a friendly Medical VQA Assistant. Answer this simple question briefly and naturally in {language}.

Question: {question}

Keep your response short and friendly. If asked who you are, mention you help analyze medical images and answer health-related questions."""

        else:  # medical
            prompt = f"""You are a medical AI assistant. Answer this health-related question in {language}.

Question: {question}

Provide a helpful, accurate response. If this requires professional medical advice, recommend consulting a healthcare provider."""

        response = model.generate_content(prompt)

        return {
            "question": question,
            "question_type": question_type,
            "response": response.text,
            "language": language,
            "model_used": model_name,
            "needs_pubmed": question_type == "medical"  # Flag for main.py
        }