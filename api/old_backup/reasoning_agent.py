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

            Guidelines:
            - Keep "Answer" ≤ 2 sentences
            - Use bullet points in "Explanation" (3–5 key points max)
            - Avoid repeating the same idea
            - Be concise and clinically focused
            - Cite using [1]–[5] inline
            - DO NOT include a References section at the end
            - References will be added automatically
            """
        else:
            prompt = f"""You are a medical expert answering questions based on peer-reviewed research.

            Question: {question}

            Related Medical Literature:
            {pubmed_articles}

            Provide a comprehensive, evidence-based answer with these guidelines:

            1. Start with a direct answer (2-3 sentences)
            2. Provide detailed explanation with supporting evidence
            3. Include clinical context and practical implications
            4. Cite sources naturally throughout using [1]–[5] only
            5. DO NOT include a References section - it will be added automatically

            Style Guidelines:
            - Write naturally in prose (not structured sections)
            - Be concise but thorough
            - Focus on answering the question directly
            - Use short paragraphs (2–4 lines)
            - Prefer bullet points for treatments or key takeaways
            - Avoid depth-for-the-sake-of-length
            """

        # Call API
        response = self.model.generate_content(prompt)
        text_output = response.text.strip()

        # Use the enhanced _strip_references method
        text_output = self._strip_references(text_output)

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
        """Enhanced method to remove all reference sections."""
        if not text:
            return text

        # Pattern 1: Remove "**References:**" or "**Reference:**" (with bold)
        text = re.split(r"\n\s*\*{0,2}References?\*{0,2}\s*:?\s*\n", text, flags=re.IGNORECASE)[0].strip()

        # Pattern 2: Line-by-line approach to catch numbered references
        lines = text.split('\n')
        cleaned_lines = []
        in_references = False

        for line in lines:
            # Check if we hit a references header
            if re.match(r'^\s*\*{0,2}References?\*{0,2}\s*:?\s*$', line, re.IGNORECASE):
                in_references = True
                break  # Stop processing, everything after is references

            # Skip lines that look like numbered references with URLs
            if re.match(r'^\s*\d+\.\s*\[.*\]\(https?://.*\)', line):
                in_references = True
                break

            # Skip lines that are just numbered URLs
            if re.match(r'^\s*\d+\.\s*https?://\S+', line):
                in_references = True
                break

            cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines).strip()

        # Pattern 3: Remove any trailing numbered list items (extra safety)
        text = re.sub(r'\n\s*\d+\.\s+\[.*?\]\(.*?\).*?$', '', text, flags=re.MULTILINE)

        return text.strip()

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