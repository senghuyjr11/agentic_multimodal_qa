"""
reasoning_agent.py - Simplified (English only)
"""
import google.generativeai as genai


class ReasoningAgent:
    """Explains VQA answers in simple English.
    Translation is handled by TranslationAgent.
    """

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemma-3-4b-it")

    def generate_response(
            self,
            question: str,
            vqa_answer: str,
            pubmed_articles: str
    ) -> str:
        """Generate response in English only."""

        prompt = f"""You are a medical AI assistant. Based on the following information, provide a clear and helpful response.

QUESTION: {question}
VQA MODEL ANSWER: {vqa_answer}

RELATED MEDICAL LITERATURE:
{pubmed_articles}

Please provide your response with the following format:

1. ANSWER: State the answer clearly
2. SIMPLE EXPLANATION: Explain what this means in plain language that anyone can understand
3. CLINICAL CONTEXT: Provide relevant medical context based on the literature
4. REFERENCES: List the PubMed articles used with their links

Keep the explanation friendly and easy to understand. Avoid complex medical jargon unless necessary."""

        response = self.model.generate_content(prompt)
        return response.text