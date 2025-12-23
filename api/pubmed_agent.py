"""
pubmed_agent.py - Knowledge Augmentation from PubMed with LLM-based term extraction
"""
import json
import re
from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai
import requests


@dataclass
class Article:
    title: str
    abstract: str
    pmid: str
    url: str

    # NEW (optional fields)
    relevance_score: Optional[int] = None
    support_score: Optional[int] = None
    relevance_why: Optional[str] = None



class PubMedAgent:
    """Augments VQA answers with relevant medical literature from PubMed using LLM-based term extraction."""

    def __init__(self, email: str, api_key: str = None, google_api_key: str = None):
        self.email = email
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

        # Initialize Gemini for intelligent term extraction (REQUIRED)
        if not google_api_key:
            raise ValueError("google_api_key is required for LLM-based term extraction")

        genai.configure(api_key=google_api_key)
        self.llm = genai.GenerativeModel("gemma-3-4b-it")
        print("✓ PubMed Agent: Using LLM-based term extraction & scoring (gemma-3-4b-it)")

        self.rank_model = self.llm  # reuse same model

    def _params(self, extra: dict) -> dict:
        params = {"email": self.email, **extra}
        if self.api_key:
            params["api_key"] = self.api_key
        return params



    def _extract_json(self, text: str) -> dict | None:
        """
        Safely extract first JSON object from LLM output.
        """
        if not text:
            return None

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def score_articles(
            self,
            question: str,
            answer: str | None,
            articles: list[Article]
    ) -> list[Article]:
        """
        Score relevance and support of each article to the question/answer.
        """

        if not self.rank_model or not articles:
            return articles

        for article in articles:
            abstract = article.abstract or ""

            prompt = f"""
    You are evaluating how well a scientific article matches a medical question.

    User question:
    {question}

    Assistant answer:
    {answer or "(no answer provided)"}

    Article:
    Title: {article.title}
    Abstract: {abstract}

    Return ONLY JSON with:
    - relevance_score (0-100): how relevant this article is to the question
    - support_score (0-100): how strongly this article supports the answer
    - why: one short sentence explanation

    Rubric:
    90-100: directly answers the question
    70-89: strongly related, useful evidence
    40-69: same domain but indirect
    0-39: weak or irrelevant
    """

            response = self.rank_model.generate_content(prompt)
            data = self._extract_json(response.text)

            if not data:
                continue

            article.relevance_score = int(max(0, min(100, data.get("relevance_score", 0))))
            article.support_score = int(max(0, min(100, data.get("support_score", 0))))
            article.relevance_why = str(data.get("why", "")).strip()

        # Sort best first
        articles.sort(
            key=lambda a: ((a.relevance_score or 0), (a.support_score or 0)),
            reverse=True
        )

        return articles

    def search(self, query: str, max_results: int = 3) -> list[Article]:
        """Search PubMed for articles matching the query."""
        resp = requests.get(
            f"{self.base_url}/esearch.fcgi",
            params=self._params({"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"})
        )
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return []

        resp = requests.get(
            f"{self.base_url}/efetch.fcgi",
            params=self._params({"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
        )
        return self._parse_xml(resp.text)

    def _parse_xml(self, xml_text: str) -> list[Article]:
        """Parse PubMed XML response into Article objects."""
        import xml.etree.ElementTree as ET
        articles = []
        root = ET.fromstring(xml_text)

        for article in root.findall(".//PubmedArticle"):
            try:
                pmid = article.find(".//PMID").text
                title = article.find(".//ArticleTitle").text or ""
                abstract_elem = article.find(".//Abstract/AbstractText")
                abstract = abstract_elem.text[:500] if abstract_elem is not None and abstract_elem.text else "No abstract"
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                articles.append(Article(title=title, abstract=abstract, pmid=pmid, url=url))
            except Exception as e:
                # Skip articles with parsing errors
                continue
        return articles

    def _format_output(self, articles: list[Article]) -> str:
        """Format articles into a readable string."""
        if not articles:
            return "No related literature found."

        output = "Related Medical Literature:\n\n"
        for i, art in enumerate(articles, 1):
            output += f"[{i}] {art.title}\n"
            output += f"    {art.abstract}...\n"
            output += f"    Link: {art.url}\n\n"
        return output

    def _extract_medical_terms_with_llm(self, topic: str | None, question: str | None, answer: str | None) -> list[str]:
        """
        Use LLM to intelligently extract medical terms for PubMed search.

        Benefits over manual stopwords:
        - Understands medical context
        - Recognizes compound medical terms (e.g., "Type 2 diabetes", "lung cancer")
        - Filters out irrelevant words intelligently
        - Includes medical synonyms (e.g., "treatment" + "therapy")
        - No manual maintenance needed
        """

        # Build input text for LLM
        input_parts = []
        if topic:
            input_parts.append(f"Topic: {topic}")
        if question:
            input_parts.append(f"Question: {question}")
        if answer:
            input_parts.append(f"Answer: {answer}")

        input_text = "\n".join(input_parts)

        if not input_text.strip():
            return []

        prompt = f"""You are a medical information extraction expert. Your task is to extract the most important MEDICAL terms for a PubMed literature search.

Input:
{input_text}

Extract 2-4 important medical/scientific terms that would help find relevant research articles in PubMed. KEEP IT FOCUSED - fewer terms = better results!

RULES:
1. Focus on: diseases, symptoms, treatments, anatomical terms, medical procedures, drug names, pathologies
2. IGNORE: question words (what, how, why), common verbs (is, are, was), articles (the, a, an), time words (long, short, takes)
3. Keep compound medical terms together (e.g., "lung cancer" NOT "lung" + "cancer", "Type 2 diabetes" NOT "Type" + "diabetes")
4. Prioritize specific medical terminology over general words
5. DO NOT include both a general term and its subtypes (e.g., if you include "pneumonia", don't also add "bacterial pneumonia" and "viral pneumonia")
6. BE SELECTIVE - only include the most essential terms!

OUTPUT FORMAT (respond ONLY with comma-separated terms, no explanation or preamble):
term1, term2, term3

EXAMPLES:

Example 1:
Input: "Topic: Pneumonia\nQuestion: How is pneumonia treated?"
Output: Pneumonia, treatment

Example 2:
Input: "Question: What causes Type 2 diabetes?"
Output: Type 2 diabetes, causes

Example 3:
Input: "Topic: Heart disease\nQuestion: What are the symptoms?\nAnswer: Chest pain and shortness of breath"
Output: Heart disease, symptoms, chest pain

Example 4:
Input: "Question: How long does it take to heal from pneumonia?"
Output: Pneumonia, recovery

Example 5:
Input: "Question: What causes pneumonia?"
Output: Pneumonia, causes

Now extract terms from the input above:"""

        try:
            response = self.llm.generate_content(prompt)
            terms_text = response.text.strip()

            # Remove any markdown formatting if present
            terms_text = terms_text.replace('*', '').replace('`', '')

            # Parse comma-separated terms
            terms = [t.strip() for t in terms_text.split(',')]
            terms = [t for t in terms if t and len(t) > 0]  # Remove empty strings

            # Limit to 4 terms (fewer = better results)
            terms = terms[:4]

            print(f"[LLM] Extracted medical terms: {terms}")
            return terms

        except Exception as e:
            print(f"[ERROR] LLM extraction failed: {e}")
            # Simple fallback: extract words from input
            words = re.findall(r'\b[A-Za-z][A-Za-z\-]{2,}\b', input_text)
            # Take first few unique words as fallback
            seen = set()
            fallback_terms = []
            for word in words:
                if word.lower() not in seen and len(fallback_terms) < 6:
                    seen.add(word.lower())
                    fallback_terms.append(word)
            print(f"[FALLBACK] Using basic extraction: {fallback_terms}")
            return fallback_terms

    def build_pubmed_query(self, topic: str | None, question: str | None, answer: str | None) -> str:
        """
        Build a PubMed query using intelligent LLM-based term extraction.

        Args:
            topic: Main medical topic from conversation context
            question: User's current question
            answer: Answer from VQA or text agent

        Returns:
            PubMed query string with AND operators
        """

        # Use LLM to extract medical terms
        medical_terms = self._extract_medical_terms_with_llm(topic, question, answer)

        # Build query from extracted terms
        if medical_terms:
            query_parts = [f'("{t}"[Title/Abstract])' for t in medical_terms]
            query = " AND ".join(query_parts)
            return query

        # Fallback to original inputs if nothing extracted
        return topic or question or answer or ""

    def get_knowledge(self, vqa_answer: str, question: str, topic: str | None = None, max_results: int = 3) -> dict:
        """
        Get relevant medical literature based on VQA answer.

        Args:
            vqa_answer: Answer from VQA model
            question: Original question asked
            topic: Optional topic extracted from conversation context
            max_results: Number of articles to retrieve (default: 3)

        Returns:
            dict with 'query', 'articles', and 'formatted' keys
        """
        query = self.build_pubmed_query(
            topic=topic,
            question=question,
            answer=vqa_answer
        )

        # Bound query length for safety
        query = (query or "").strip()[:300]

        articles = self.search(query, max_results=max_results)

        return {
            "query": query,
            "articles": articles,
            "formatted": self._format_output(articles)
        }

    def search_topic(self, question: str, topic: str | None = None, max_results: int = 3) -> dict:
        """
        Search for medical literature based on a topic question (text-only, no VQA answer).

        Args:
            question: Medical question to search
            topic: Optional topic extracted from conversation context
            max_results: Number of articles to retrieve (default: 3)

        Returns:
            dict with 'query', 'articles', and 'formatted' keys
        """
        query = self.build_pubmed_query(
            topic=topic,
            question=question,
            answer=None
        )

        query = (query or "").strip()[:300]

        articles = self.search(query, max_results=max_results)

        return {
            "query": query,
            "articles": articles,
            "formatted": self._format_output(articles)
        }