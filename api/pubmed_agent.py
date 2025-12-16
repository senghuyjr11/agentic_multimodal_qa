"""
pubmed_agent.py - Knowledge Augmentation from PubMed
"""
import requests
from dataclasses import dataclass


@dataclass
class Article:
    title: str
    abstract: str
    pmid: str
    url: str


class PubMedAgent:
    """Augments VQA answers with relevant medical literature from PubMed."""

    def __init__(self, email: str, api_key: str = None):
        self.email = email
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def _params(self, extra: dict) -> dict:
        params = {"email": self.email, **extra}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search(self, query: str, max_results: int = 3) -> list[Article]:
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
            except:
                continue
        return articles

    def _format_output(self, articles: list[Article]) -> str:
        if not articles:
            return "No related literature found."

        output = "Related Medical Literature:\n\n"
        for i, art in enumerate(articles, 1):
            output += f"[{i}] {art.title}\n"
            output += f"    {art.abstract}...\n"
            output += f"    Link: {art.url}\n\n"
        return output

    def get_knowledge(self, vqa_answer: str, question: str) -> dict:
        """Get knowledge for image + question (combines answer and question)."""
        query = f"{vqa_answer} {question}"
        articles = self.search(query)

        return {
            "query": query,
            "articles": articles,
            "formatted": self._format_output(articles)
        }

    def search_topic(self, question: str) -> dict:
        """Get knowledge for text-only question."""
        articles = self.search(question)

        return {
            "query": question,
            "articles": articles,
            "formatted": self._format_output(articles)
        }