"""
reasoning_agent.py - Simplified (English only)
"""
import re

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
            Explanation: [Bullet points]
            Clinical Context: [Clinical significance]

            References:
            [List numbered references with PMIDs]

            Guidelines:
            - Keep "Answer" ≤ 2 sentences
            - Use bullet points in "Explanation" (3–5 key points max)
            - Avoid repeating the same idea
            - Be concise and clinically focused
            - Use markdown links: [Title](URL)
            - Cite using [1]–[5] only
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
            4. Cite sources naturally throughout your response using [1]–[5] only (no [6]+)
            5. End with a "References" section listing all cited sources
        
            Style Guidelines:
            - Write naturally in prose (not structured sections)
            - Use markdown links: [Title](URL) (PMID: XXXXX)
            - Be concise but thorough
            - Focus on answering the question directly
            - Use short paragraphs (2–4 lines)
            - Prefer bullet points for treatments or key takeaways
            - Avoid depth-for-the-sake-of-length
        
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

        formatted_refs = "\n\n---\n\n**Top 5 References Links (ranked by semantic match)**"
        for i, article in enumerate((article_objects or [])[:5], 1):
            title = article.title.strip() if article.title else "Untitled"
            title_short = title if len(title) <= 80 else title[:80].rstrip() + "…"

            badge, label = self._score_to_badge(getattr(article, "relevance_score", None))
            why = getattr(article, "relevance_why", "") or label

            formatted_refs += (
                f"\n{i}. [{title_short}]({article.url}) — **{badge}** — *{why}* (PMID: {article.pmid})"
            )

        return text_output + formatted_refs

    def rewrite_response(self, last_answer: str, instruction: str) -> str:
        """
        Rewrite ONLY the last answer according to the instruction.
        Strongly disallow adding any new references/links/citations.
        """
        prompt = f"""
You are editing an existing assistant answer.

INSTRUCTION:
{instruction}

TEXT TO EDIT (keep meaning the same unless instruction says otherwise):
---BEGIN---
{last_answer}
---END---

STRICT RULES:
- Do NOT add any new references, links, URLs, PMIDs, citations, or "References" section.
- If the text contains a "References" or "Reference" section, REMOVE it completely.
- If the text contains inline citations like [1], [2], remove them ONLY if needed to keep the text clean after removing references.
- Output ONLY the rewritten text. No preamble.
"""

        resp = self.model.generate_content(prompt)
        rewritten = (resp.text or "").strip()

        # Safety net: hard-remove references sections + trailing URLs
        rewritten = self._strip_references(rewritten)

        return rewritten

    def _strip_references(self, text: str) -> str:
        if not text:
            return text

        # Remove sections starting with "References:" or "Reference:"
        text = re.split(r"\n\s*References?\s*:\s*\n", text, flags=re.IGNORECASE)[0].strip()

        # Also remove common "References" header without colon
        text = re.split(r"\n\s*References?\s*\n", text, flags=re.IGNORECASE)[0].strip()

        # Remove trailing numbered URL-only lines (extra safety)
        text = re.sub(r"\n\s*\d+\.\s*https?://\S+\s*$", "", text, flags=re.MULTILINE).strip()

        return text

    def _score_to_badge(self, score: float | None) -> tuple[str, str]:
        """
        Convert cosine similarity (0..1-ish) into user-friendly badge.
        """
        if score is None:
            return ("Match: --", "Relevance not available")

        pct = max(0, min(100, int(round(score * 100))))
        if pct >= 80:
            label = "Strong match"
        elif pct >= 65:
            label = "Good match"
        elif pct >= 50:
            label = "Related"
        else:
            label = "Weak match"
        return (f"Match: {pct}%", label)


