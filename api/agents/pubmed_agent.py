"""
pubmed_agent_simplified.py - Clean PubMed search and relevance scoring

Responsibilities:
1. Search PubMed given a query
2. Score articles by relevance using embeddings
3. That's it - no LLM extraction, no complex fallbacks
"""

import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from google import genai as genai_client
from google.genai import types as genai_types


@dataclass
class Article:
    """Simple article representation"""
    title: str
    abstract: str
    pmid: str
    url: str
    relevance_score: Optional[float] = None


class PubMedAgent:
    """
    Simplified PubMed agent - just search and score.

    Query building is now handled by the ConversationOrchestrator's LLM,
    which understands conversation context better.
    """

    def __init__(self, email: str, api_key: str, google_api_key: str):
        self.email = email
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

        # Validate Google API key
        if not google_api_key or google_api_key.strip() == "":
            raise ValueError(
                "Google API key is required for embeddings. "
                "Please set GOOGLE_API_KEY in your .env file."
            )

        # For embeddings
        self.embedding_client = genai_client.Client(api_key=google_api_key)

        print("✓ PubMed Agent initialized (simplified)")

    def search(self, query: str, max_results: int = 5) -> List[Article]:
        """
        Search PubMed with a query string.

        Args:
            query: PubMed search query (e.g., "diabetes treatment")
            max_results: Maximum number of articles to return

        Returns:
            List of Article objects
        """
        print(f"[PubMed] Searching: '{query}' (max {max_results} results)")

        # Search for PMIDs
        params = {
            "email": self.email,
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }

        if self.api_key:
            params["api_key"] = self.api_key

        try:
            resp = requests.get(f"{self.base_url}/esearch.fcgi", params=params, timeout=30)  # Changed from 10 to 30
            resp.raise_for_status()

            result = resp.json()
            pmids = result.get("esearchresult", {}).get("idlist", [])

            if not pmids:
                print(f"[PubMed] No articles found for query: '{query}'")
                return []

            print(f"[PubMed] Found {len(pmids)} PMIDs")

            # Fetch article details
            return self._fetch_articles(pmids)

        except Exception as e:
            print(f"[PubMed] Search error: {e}")
            return []

    def _fetch_articles(self, pmids: List[str]) -> List[Article]:
        """Fetch full article details for given PMIDs"""

        params = {
            "email": self.email,
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }

        if self.api_key:
            params["api_key"] = self.api_key

        # Add retry logic
        max_retries = 3

        for attempt in range(max_retries):
            try:
                print(f"[PubMed] Fetching articles (attempt {attempt + 1}/{max_retries})...")

                resp = requests.get(
                    f"{self.base_url}/efetch.fcgi",
                    params=params,
                    timeout=30  # Increased from 15 to 30 seconds
                )
                resp.raise_for_status()

                return self._parse_xml(resp.text)

            except requests.exceptions.Timeout:
                print(f"[PubMed] Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 2
                    print(f"[PubMed] Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"[PubMed] All retries failed due to timeout")
                    return []

            except requests.exceptions.ConnectionError as e:
                print(f"[PubMed] Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 2
                    print(f"[PubMed] Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"[PubMed] All retries failed due to connection error")
                    return []

            except Exception as e:
                print(f"[PubMed] Fetch error: {e}")
                return []

        return []

    def _parse_xml(self, xml_text: str) -> List[Article]:
        """Parse PubMed XML response"""
        articles = []

        try:
            root = ET.fromstring(xml_text)

            for article_elem in root.findall(".//PubmedArticle"):
                try:
                    pmid = article_elem.find(".//PMID").text

                    title_elem = article_elem.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None and title_elem.text else "Untitled"

                    abstract_elem = article_elem.find(".//Abstract/AbstractText")
                    abstract = "No abstract available"
                    if abstract_elem is not None and abstract_elem.text:
                        abstract = abstract_elem.text[:500]  # Truncate long abstracts

                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                    articles.append(Article(
                        title=title,
                        abstract=abstract,
                        pmid=pmid,
                        url=url
                    ))

                except Exception as e:
                    print(f"[PubMed] Error parsing article: {e}")
                    continue

            print(f"[PubMed] Successfully parsed {len(articles)} articles")
            return articles

        except Exception as e:
            print(f"[PubMed] XML parsing error: {e}")
            return []

    def score_articles(self, query: str, articles: List[Article]) -> List[Article]:
        """
        Score articles by relevance to query using embeddings.

        Args:
            query: Original user question
            articles: List of articles to score

        Returns:
            Articles sorted by relevance (highest first)
        """
        if not articles:
            return articles

        print(f"[PubMed] Scoring {len(articles)} articles for relevance...")

        try:
            # Get query embedding
            query_result = self.embedding_client.models.embed_content(
                model="models/text-embedding-004",
                contents=[query],
                config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            query_embedding = np.array(query_result.embeddings[0].values)

            # Get article embeddings
            article_texts = [f"{a.title}. {a.abstract}" for a in articles]
            article_result = self.embedding_client.models.embed_content(
                model="models/text-embedding-004",
                contents=article_texts,
                config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            article_embeddings = [np.array(e.values) for e in article_result.embeddings]

            # Compute similarities
            query_matrix = query_embedding.reshape(1, -1)
            article_matrix = np.array(article_embeddings)
            similarities = cosine_similarity(query_matrix, article_matrix)[0]

            # Assign scores
            for i, article in enumerate(articles):
                article.relevance_score = float(similarities[i])

            # Sort by relevance
            articles.sort(key=lambda a: a.relevance_score or 0, reverse=True)

            # Debug output
            print("[PubMed] Top 3 articles by relevance:")
            for i, article in enumerate(articles[:3], 1):
                print(f"  {i}. [{article.relevance_score:.3f}] {article.title[:60]}...")

            return articles

        except Exception as e:
            print(f"[PubMed] Scoring error: {e}, returning unscored articles")
            return articles