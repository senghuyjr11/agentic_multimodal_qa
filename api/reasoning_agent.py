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
        article_objects: list = None,
        conversation_context: str = "",
        is_image_question: bool = False
    ) -> str:
        article_objects = article_objects or []

        if is_image_question:
            # Structured format for image analysis
            prompt = f"""You are a medical expert analyzing a medical image.

    Question: {question}
    Image Analysis: {vqa_answer}

    Related Medical Literature:
    {pubmed_articles}

    Provide a comprehensive response in this format:

    Answer: [Concise 1-2 sentence answer]
    Location: [Where in the image this finding is located]
    Explanation: [Detailed explanation with citations]
    Clinical Context: [Clinical significance]

    References:
    [List numbered references with PMIDs]

    Guidelines:
    - Use markdown links: [Title](URL)
    - Cite using [1], [2], [3]
    - Location should describe the anatomical/spatial location in the image
    """
        else:
            # Natural format for text questions
            prompt = f"""You are a medical expert answering questions based on peer-reviewed research.

    Question: {question}

    Related Medical Literature:
    {pubmed_articles}

    Provide a comprehensive, evidence-based answer with these guidelines:

    1. Start with a direct answer (2-3 sentences)
    2. Provide detailed explanation with supporting evidence
    3. Include clinical context and practical implications
    4. Cite sources naturally throughout your response using [1], [2], [3]
    5. End with a "References" section listing all cited sources

    Style Guidelines:
    - Write naturally in prose (not structured sections)
    - Use markdown links: [Title](URL) (PMID: XXXXX)
    - Be concise but thorough
    - Focus on answering the question directly

    References format:
    1. [Article Title](URL) (PMID: XXXXX)
    2. [Article Title](URL) (PMID: XXXXX)
    """

        # Call API
        response = self.model.generate_content(prompt)
        text_output = response.text.strip()

        if "References:" in text_output or "Reference:" in text_output:
            text_output = text_output.split("Reference")[0].strip()

        if not article_objects:
            return text_output

        formatted_refs = "\n\nReferences:"
        for i, article in enumerate(article_objects[:3], 1):
            title = article.title[:60] + "..." if len(article.title) > 60 else article.title
            formatted_refs += f"\n{i}. [{title}]({article.url}) (PMID: {article.pmid})"

        return text_output + formatted_refs

    def rewrite_response(self, last_answer: str, instruction: str) -> str:
        """
        Rewrite the previous answer following the user's instruction
        (e.g., shorten, simplify, bullet points) WITHOUT adding new facts.
        """
        if not last_answer.strip():
            return ""

        prompt = f"""
        You are rewriting the assistant's previous answer.
        
        User instruction: "{instruction}"
        
        Rules:
        - DO NOT add new medical facts.
        - DO NOT invent citations.
        - Keep the meaning, remove redundancy.
        - If there is a References section, you may keep it but shorten to max 3 items.
        - Output ONLY the rewritten answer (no preamble).
        
        Previous answer:
        \"\"\"{last_answer}\"\"\"
        
        Rewritten answer:
        """
        response = self.model.generate_content(prompt)
        return response.text.strip()

