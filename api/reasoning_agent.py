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
            pubmed_articles: str,
            article_objects: list,
            conversation_context: str = ""  # ← ADD THIS
    ) -> str:
        """
        Generates a structured medical response with conversation awareness.

        Args:
            question: User's question
            vqa_answer: Visual answer from image model
            pubmed_articles: Formatted string (for context)
            article_objects: List of actual article dicts (for links)
            conversation_context: Previous conversation history
        """

        # Build context-aware prompt
        context_section = ""
        if conversation_context:
            context_section = f"""
    Previous Conversation:
    {conversation_context}

    IMPORTANT: Use the conversation history above to provide context-aware answers. If the current question refers to something discussed earlier (like "What are the symptoms?" after discussing a disease), reference that previous context.
    """

        prompt = f"""{context_section}
    You are an expert medical AI. Answer based strictly on the visual findings and research context.

    User Question: {question}
    Visual Findings: {vqa_answer}
    Research Context:
    {pubmed_articles}

    STRICT OUTPUT FORMAT (Follow exactly):
    Answer: (Direct 1-sentence answer that considers previous conversation if relevant)
    Location: (Describe WHERE in the image the findings are located - e.g., "right lower lobe", "left anterior region", "throughout the tissue")
    Explanation: (1-2 sentences explaining the significance, with citations like [1], [2])
    Clinical Context: (1-2 sentences on clinical implications, citing [1]-[3])

    Do NOT write a References section - it will be auto-generated.
    """

        # Call API
        response = self.model.generate_content(prompt)
        text_output = response.text.strip()

        # Remove any References section the model might add
        if "References:" in text_output or "Reference:" in text_output:
            text_output = text_output.split("Reference")[0].strip()

        # Append real links
        formatted_refs = "\n\nReferences:"
        for i, article in enumerate(article_objects[:3], 1):
            title = article.title[:60] + "..." if len(article.title) > 60 else article.title
            formatted_refs += f"\n{i}. [{title}]({article.url}) (PMID: {article.pmid})"

        return text_output + formatted_refs

