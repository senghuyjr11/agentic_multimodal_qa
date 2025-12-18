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

    def generate_response(self, question: str, vqa_answer: str, pubmed_articles: str,
                          article_objects: list = None) -> str:
        """
        Generates a structured medical response.

        Args:
            question: User's question
            vqa_answer: Visual answer from image model
            pubmed_articles: Formatted string (for context)
            article_objects: List of actual article dicts (for links)
        """

        # Use the pre-formatted string for context
        prompt = f"""
    You are an expert medical AI. Answer based strictly on the visual findings and research context.

    User Question: {question}
    Visual Findings: {vqa_answer}
    Research Context:
    {pubmed_articles}

    STRICT OUTPUT FORMAT (Follow exactly):
    Answer: (Direct 1-sentence answer)
    Explanation: (1-2 sentences with citations like [1], [2])
    Clinical Context: (1-2 sentences on implications, citing [1]-[3])

    Do NOT write a References section - it will be auto-generated.
    """

        # Call API
        response = self.model.generate_content(prompt)
        text_output = response.text.strip()

        # Remove any References section the model might add
        if "References:" in text_output or "Reference:" in text_output:
            text_output = text_output.split("Reference")[0].strip()

        # Append real links if article objects provided
        if article_objects:
            formatted_refs = "\n\nReferences:"
            for i, article in enumerate(article_objects[:3], 1):  # Top 3 only
                title = article.title[:60] + "..." if len(article.title) > 60 else article.title
                formatted_refs += f"\n{i}. [{title}]({article.url}) (PMID: {article.pmid})"

            return text_output + formatted_refs

        return text_output

