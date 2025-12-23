"""
main.py - Clean turn-based session storage (NO top-level agent redundancy)

Key changes vs your previous version:
- SessionManager.update(...) is NOT used anywhere.
- All agent outputs (translation/image/vqa/text/pubmed/reasoning) are stored per-turn in:
  session_data["conversation_history"][i]["meta"]
- run() returns (source_language, output_language) and keeps original_language alias.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple, Dict, Any

from image_agent import ImageAgent, ModelConfig, OODConfig
from pubmed_agent import PubMedAgent
from text_only_agent import TextOnlyAgent
from reasoning_agent import ReasoningAgent
from translation_agent import TranslationAgent
from session_manager import SessionManager

from transformers import Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
REQUIRED_KEYS = ["GOOGLE_API_KEY", "NCBI_EMAIL", "NCBI_API_KEY"]

missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Missing environment variables: {missing}")

class MedicalVQAPipeline:
    """Connects all agents with translation as front layer."""

    def __init__(
            self,
            ncbi_email: str,
            ncbi_api_key: str,
            google_api_key: str,
            pathvqa_config: ModelConfig,
            vqa_rad_config: ModelConfig,
            classifier_path: str,
            sessions_dir: str = "sessions",
            ood_config: OODConfig = None
    ):
        print("Initializing Medical VQA Pipeline...")

        # Translation Agent - FRONT LAYER
        self.translation_agent = TranslationAgent(api_key=google_api_key)

        # Clean turn-based session manager
        self.session_manager = SessionManager(sessions_dir)

        # Memory storage for active conversations
        self.active_conversations = {}  # {session_id: memory}

        self.image_agent = ImageAgent(
            pathvqa_config=pathvqa_config,
            vqa_rad_config=vqa_rad_config,
            classifier_path=classifier_path,
            ood_config=ood_config
        )

        self.text_only_agent = TextOnlyAgent(api_key=google_api_key)

        self.pubmed_agent = PubMedAgent(
            email=ncbi_email,
            api_key=ncbi_api_key,
            google_api_key=google_api_key
        )

        self.reasoning_agent = ReasoningAgent(api_key=google_api_key)

        print("✓ Pipeline ready\n")

    # -------------------------
    # Memory handling
    # -------------------------
    def get_or_create_memory(self, username: str, session_id: int) -> InMemoryChatMessageHistory:
        """
        Get memory from RAM if exists, otherwise load from JSON.
        This handles both: switching between chats + resuming old chats.
        """
        if session_id in self.active_conversations:
            print(f"✓ Using cached memory for session {session_id}")
            return self.active_conversations[session_id]

        print(f"⟳ Loading conversation history from JSON for session {session_id}")
        memory = self._restore_memory_from_json(username, session_id)

        self.active_conversations[session_id] = memory
        return memory

    def _restore_memory_from_json(self, username: str, session_id: int) -> InMemoryChatMessageHistory:
        """Reconstruct LangChain memory from saved JSON conversation history."""
        conversation_history = self.session_manager.get_conversation_history(username, session_id)

        memory = InMemoryChatMessageHistory()

        for turn in conversation_history:
            memory.add_user_message(turn["user"])
            memory.add_ai_message(turn["assistant"])

        if conversation_history:
            print(f"✓ Restored {len(conversation_history)} conversation turns")
        else:
            print("✓ New conversation started")

        return memory

    def get_conversation_context(self, memory: InMemoryChatMessageHistory) -> str:
        """Extract conversation history as formatted string for prompts."""
        messages = memory.messages
        if not messages:
            return ""

        context_lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                context_lines.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_lines.append(f"AI: {msg.content}")

        return "\n".join(context_lines)

    # -------------------------
    # Topic extraction / context enhancement
    # -------------------------
    def _extract_topic_from_context(self, conversation_context: str) -> Optional[str]:
        """
        Extract the main medical topic from conversation context.

        Strategy:
        1. Look at VQA answers (most reliable source)
        2. Look at first user question
        """
        if not conversation_context:
            return None

        lines = conversation_context.split('\n')

        vqa_answers = []
        user_questions = []

        for line in lines:
            if line.startswith("AI:"):
                ai_content = line.replace("AI:", "").strip()
                if "Original VQA Detection:" in ai_content:
                    match = re.search(r'Original VQA Detection:\*\*\s*([^\n*]+)', ai_content)
                    if match:
                        vqa_answer = match.group(1).strip()
                        vqa_answers.append(vqa_answer)
                        print(f"[DEBUG] Found VQA answer: '{vqa_answer}'")
            elif line.startswith("Human:"):
                user_questions.append(line.replace("Human:", "").strip())

        if vqa_answers:
            topic = vqa_answers[-1].replace("[Image uploaded]", "").strip()
            if topic and len(topic) > 2:
                print(f"[DEBUG] Extracted topic from VQA: '{topic}'")
                return topic

        if user_questions:
            first_question = user_questions[0]
            print(f"[DEBUG] Extracting topic from first question: '{first_question}'")

            question_words = r'\b(what|how|why|when|where|who|which|is|are|can|could|would|should|does|do|did|the|a|an|this|that|these|those)\b'
            cleaned = re.sub(question_words, ' ', first_question, flags=re.IGNORECASE)
            words = re.findall(r'\b[A-Za-z]{3,}\b', cleaned)

            stopwords = {'about', 'there', 'their', 'have', 'has', 'had', 'been', 'being', 'image', 'uploaded'}
            medical_terms = [w for w in words if w.lower() not in stopwords]

            if medical_terms:
                topic = medical_terms[0]
                print(f"[DEBUG] Extracted topic from first question: '{topic}'")
                return topic

        print("[DEBUG] No topic found")
        return None

    def _enhance_question_with_context(self, question: str, conversation_context: str) -> str:
        """Enhance a question with context for better PubMed searches."""
        if not conversation_context:
            return question

        topic = self._extract_topic_from_context(conversation_context)
        if not topic:
            print(f"[DEBUG] No topic found, using original question: '{question}'")
            return question

        pronouns = r'\b(it|this|that|they|them)\b'
        if re.search(pronouns, question, re.IGNORECASE):
            enhanced = re.sub(pronouns, topic.lower(), question, flags=re.IGNORECASE, count=1)
            print(f"✓ Enhanced question with context: '{question}' -> '{enhanced}'")
            return enhanced

        context_dependent_patterns = [
            r'^(what|how|why|when|where)\s+(is|are|causes?|treatments?|symptoms?|does|do)',
            r'^(can|could|should|would|will)',
        ]

        for pattern in context_dependent_patterns:
            if re.match(pattern, question, re.IGNORECASE):
                enhanced = f"{topic} {question}"
                print(f"✓ Enhanced question with context: '{question}' -> '{enhanced}'")
                return enhanced

        return question

    # -------------------------
    # PubMed fallback strategy
    # -------------------------
    def _search_pubmed_with_fallback(
            self,
            question: str,
            answer: Optional[str],
            topic: Optional[str],
            max_results: int = 3
    ) -> Optional[dict]:
        """
        Search PubMed with fallback:
        1) Specific (LLM extracted)
        2) Broader (topic + keyword)
        3) Broadest (topic only)
        """
        print("[PubMed] Attempt 1: Specific search with LLM-extracted terms")

        if answer:
            knowledge = self.pubmed_agent.get_knowledge(
                vqa_answer=answer,
                question=question,
                topic=topic,
                max_results=max_results
            )
        else:
            knowledge = self.pubmed_agent.search_topic(
                question=question,
                topic=topic,
                max_results=max_results
            )

        if knowledge and knowledge.get("articles"):
            print(f"[PubMed] ✓ Found {len(knowledge['articles'])} articles (specific search)")
            return knowledge

        if topic:
            print("[PubMed] Attempt 2: Broader search (topic + main keyword)")

            keywords = ["treatment", "causes", "symptoms", "diagnosis", "prevention", "recovery", "pathogenesis"]
            main_keyword = next((k for k in keywords if k in question.lower()), None)

            if main_keyword:
                query = f'("{topic}"[Title/Abstract]) AND ("{main_keyword}"[Title/Abstract])'
            else:
                query = f'("{topic}"[Title/Abstract])'

            articles = self.pubmed_agent.search(query, max_results=max_results * 2)
            if articles:
                print(f"[PubMed] ✓ Found {len(articles)} articles (broader search)")
                return {
                    "query": query,
                    "articles": articles,
                    "formatted": self.pubmed_agent._format_output(articles)
                }

        if topic:
            print("[PubMed] Attempt 3: Broadest search (topic only)")
            query = f'("{topic}"[Title/Abstract])'
            articles = self.pubmed_agent.search(query, max_results=max_results * 3)
            if articles:
                print(f"[PubMed] ✓ Found {len(articles)} articles (broadest search)")
                return {
                    "query": query,
                    "articles": articles,
                    "formatted": self.pubmed_agent._format_output(articles)
                }

        print("[PubMed] ✗ No articles found after all attempts")
        return None

    def _handle_no_articles_found(self, topic: Optional[str]) -> Tuple[str, dict]:
        """Return an honest message and meta to store in the turn."""
        topic_text = f" about '{topic}'" if topic else ""

        honest_response = (
            f"I apologize, but I couldn't find peer-reviewed research articles "
            f"to answer your question{topic_text} with proper citations.\n\n"
            f"This could mean:\n"
            f"• The search terms were too specific\n"
            f"• Limited published research exists on this exact aspect\n"
            f"• The topic might need to be rephrased\n\n"
            f"Suggestion: Try asking a more general question{topic_text}, "
            f"or rephrase your question differently.\n\n"
            f"For medical information, I can only provide answers backed by "
            f"published research to ensure accuracy and trustworthiness."
        )

        meta = {
            "pubmed_agent": {
                "status": "no_articles_found",
                "query": "multiple_attempts_failed",
                "articles": [],
                "attempts": 3,
                "topic": topic
            },
            "reasoning_agent": {
                "skipped": True,
                "reason": "no_pubmed_articles",
                "response": honest_response
            }
        }

        return honest_response, meta

    # -------------------------
    # MAIN ENTRYPOINT
    # -------------------------
    def run(
            self,
            username: str,
            question: str = None,
            image_path: str = None,
            session_id: int = None
    ) -> dict:
        """
        Clean turn-based run():
        - No top-level agent keys in session JSON
        - All per-turn meta stored in conversation_history[*].meta
        """
        if not image_path and not question:
            raise ValueError("Must provide at least image_path or question")

        print("=" * 60)
        print("MEDICAL VQA PIPELINE")
        print("=" * 60)

        # --- Translation layer ---
        source_language = "English"
        output_language = "English"
        english_question = question if question else ""

        if question:
            translation_result = self.translation_agent.process_input(question)
            source_language = translation_result.get("source_language", "English")
            output_language = translation_result.get(
                "output_language",
                translation_result.get("detected_language", "English")
            )
            english_question = translation_result["english_question"]

        # Determine input type
        input_type = "image" if image_path else "text_only"
        print(f"Input type: {input_type}")

        # --- Session / memory ---
        if session_id is None:
            session_id = self.session_manager.create_session(
                username,
                question or "",
                image_path
            )
            memory = self.get_or_create_memory(username, session_id)
            print(f"✓ Started new session: {username}/{session_id}")
        else:
            if not self.session_manager.session_exists(username, session_id):
                raise ValueError(f"Session {username}/{session_id} not found")
            memory = self.get_or_create_memory(username, session_id)
            print(f"✓ Continuing session: {username}/{session_id}")

        conversation_context = self.get_conversation_context(memory)
        if conversation_context:
            print(f"[DEBUG] Conversation context:\n{conversation_context[:200]}...")

        # Build per-turn meta root (translation always stored per turn)
        turn_meta: Dict[str, Any] = {
            "translation": {
                "source_language": source_language,
                "output_language": output_language,
                "original_question": question,
                "english_question": english_question
            }
        }

        # --- Run agents (English internal) ---
        if input_type == "text_only":
            english_response, meta_updates = self._run_text_only(
                english_question=english_question,
                output_language=output_language,
                conversation_context=conversation_context
            )
        else:
            english_response, meta_updates = self._run_with_image(
                image_path=image_path,
                english_question=english_question,
                output_language=output_language,
                conversation_context=conversation_context
            )

        if meta_updates:
            turn_meta.update(meta_updates)

        # Translate back to output language
        final_response = self.translation_agent.process_output(english_response, output_language)

        # Save to RAM memory
        memory.add_user_message(english_question if question else "[Image uploaded]")
        memory.add_ai_message(english_response)

        # Save to disk as a new turn with meta (clean schema)
        self.session_manager.add_conversation_turn(
            username=username,
            session_id=session_id,
            user_message=question if question else "[Image uploaded]",
            assistant_message=final_response,
            image_path=image_path,
            meta=turn_meta
        )

        self._print_final_output(final_response, username, session_id)

        session_data = self.session_manager.load(username, session_id)
        current_turn = len(session_data.get("conversation_history", []))

        return {
            "username": username,
            "session_id": session_id,
            "input_type": input_type,

            # Correct semantics
            "source_language": source_language,
            "output_language": output_language,

            # Backward compatibility alias
            "original_language": output_language,

            "enhanced_response": final_response,
            "turn": current_turn
        }

    # -------------------------
    # IMAGE PIPELINE
    # -------------------------
    def _run_with_image(
            self,
            image_path: str,
            english_question: str,
            output_language: str,
            conversation_context: str = ""
    ) -> Tuple[str, dict]:
        """Run full pipeline with image (internal processing in English). Returns (english_response, meta_updates)."""

        meta_updates: Dict[str, Any] = {}

        # 1) Image Agent
        print("\n[1/3] Image Agent - Routing & Prediction")
        vqa_result = self.image_agent.predict(image_path, english_question)

        meta_updates["image_agent"] = {
            "routed_to": vqa_result.get("model"),
            "confidence": vqa_result.get("confidence"),
            "ood": vqa_result.get("ood", False),
            "ood_rule": vqa_result.get("ood_rule", None),
        }

        meta_updates["vqa_agent"] = {
            "question": vqa_result.get("question"),
            "answer": vqa_result.get("answer"),
            "ood": vqa_result.get("ood", False),
            "original_output": vqa_result.get("answer")
        }

        print(f"VQA Answer: {vqa_result.get('answer')}")

        # OOD short-circuit
        if vqa_result.get("ood", False) or vqa_result.get("model") == "unknown":
            meta_updates["pubmed_agent"] = {"skipped": True, "reason": "ood_rejection"}
            meta_updates["reasoning_agent"] = {
                "skipped": True,
                "reason": "ood_rejection",
                "response": vqa_result.get("answer", "")
            }
            return vqa_result.get("answer", ""), meta_updates

        # 2) PubMed Agent with fallback
        print("\n[2/3] PubMed Agent - Fetching Knowledge")

        topic = self._extract_topic_from_context(conversation_context)

        knowledge = self._search_pubmed_with_fallback(
            question=vqa_result.get("question", english_question),
            answer=vqa_result.get("answer", ""),
            topic=topic,
            max_results=3
        )

        if knowledge is None or not knowledge.get("articles"):
            print("[PubMed] No articles found - returning honest response")
            honest_response, no_article_meta = self._handle_no_articles_found(topic)
            meta_updates.update(no_article_meta)
            return honest_response, meta_updates

        meta_updates["pubmed_agent"] = {
            "query": knowledge.get("query"),
            "articles": [
                {"title": a.title, "abstract": a.abstract, "pmid": a.pmid, "url": a.url}
                for a in knowledge["articles"]
            ]
        }

        print(f"Found {len(knowledge['articles'])} articles")

        # 3) Reasoning Agent
        print("\n[3/3] Reasoning Agent - Generating Explanation")

        english_response = self.reasoning_agent.generate_response(
            question=vqa_result.get("question", english_question),
            vqa_answer=vqa_result.get("answer", ""),
            pubmed_articles=knowledge.get("formatted", ""),
            article_objects=knowledge["articles"]
        )

        original_vqa_section = f"**Original VQA Detection:** {vqa_result.get('answer','')}\n\n---\n\n"
        english_response = original_vqa_section + english_response

        meta_updates["reasoning_agent"] = {
            "original_vqa_output": vqa_result.get("answer"),
            "enhanced_response": english_response
        }

        return english_response, meta_updates

    # -------------------------
    # TEXT-ONLY PIPELINE
    # -------------------------
    def _run_text_only(
            self,
            english_question: str,
            output_language: str,
            conversation_context: str = ""
    ) -> Tuple[str, dict]:
        """Run pipeline for text-only input. Returns (english_response, meta_updates)."""

        meta_updates: Dict[str, Any] = {}

        # 1) Text-only agent
        print("\n[1] Text-Only Agent - Classifying Question")
        text_only_result = self.text_only_agent.respond(english_question, conversation_context)

        meta_updates["text_agent"] = {
            "question": text_only_result.get("question"),
            "question_type": text_only_result.get("question_type")
        }

        # Casual -> skip PubMed
        if text_only_result.get("question_type") == "casual":
            meta_updates["pubmed_agent"] = {"skipped": True, "reason": "casual_question"}
            meta_updates["reasoning_agent"] = {
                "skipped": True,
                "reason": "casual_question",
                "response": text_only_result.get("response", "")
            }
            return text_only_result.get("response", ""), meta_updates

        # 2) PubMed
        print("\n[2] PubMed Agent - Fetching Knowledge")

        enhanced_question = self._enhance_question_with_context(english_question, conversation_context)
        topic = self._extract_topic_from_context(conversation_context)

        print(f"[DEBUG] Using enhanced question: '{enhanced_question}' with topic: '{topic}'")

        knowledge = self._search_pubmed_with_fallback(
            question=enhanced_question,
            answer=None,
            topic=topic,
            max_results=3
        )

        if knowledge is None or not knowledge.get("articles"):
            print("[PubMed] No articles found - returning honest response")
            honest_response, no_article_meta = self._handle_no_articles_found(topic)
            meta_updates.update(no_article_meta)
            return honest_response, meta_updates

        meta_updates["pubmed_agent"] = {
            "query": knowledge.get("query"),
            "articles": [
                {"title": a.title, "abstract": a.abstract, "pmid": a.pmid, "url": a.url}
                for a in knowledge["articles"]
            ]
        }

        print(f"Found {len(knowledge['articles'])} articles")

        # 3) Reasoning
        print("\n[3] Reasoning Agent - Generating Explanation")

        english_response = self.reasoning_agent.generate_response(
            question=english_question,
            vqa_answer="(Text-only question - no image analysis)",
            pubmed_articles=knowledge.get("formatted", ""),
            article_objects=knowledge["articles"],
            conversation_context=conversation_context
        )

        meta_updates["reasoning_agent"] = {"response": english_response}
        return english_response, meta_updates

    # -------------------------
    # Debug output
    # -------------------------
    def _print_final_output(self, response: str, username: str, session_id: int):
        print("\n" + "=" * 60)
        print("FINAL OUTPUT")
        print("=" * 60)
        print(response)
        print("=" * 60)
        print(f"Session saved: {username}/{session_id}")


if __name__ == "__main__":
    pathvqa_config = ModelConfig(
        base_model_id="Qwen/Qwen2-VL-7B-Instruct",
        adapter_path="../qwen2vl_7b_pathvqa_adapters",
        model_class=Qwen2VLForConditionalGeneration
    )

    vqa_rad_config = ModelConfig(
        base_model_id="Qwen/Qwen3-VL-2B-Instruct",
        adapter_path="../qwen3vl_2b_vqa_rad_adapters",
        model_class=Qwen3VLForConditionalGeneration
    )

    pipeline = MedicalVQAPipeline(
        ncbi_email=os.getenv("NCBI_EMAIL"),
        ncbi_api_key=os.getenv("NCBI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        pathvqa_config=pathvqa_config,
        vqa_rad_config=vqa_rad_config,
        classifier_path="../modality_classifier"
    )

    print("\n### TEST: Context-Aware Conversation ###")

    result1 = pipeline.run(
        username="kyojuro",
        question="What is pneumonia?"
    )
    sid = result1["session_id"]

    result2 = pipeline.run(
        username="kyojuro",
        question="What causes it?",
        session_id=sid
    )

    result3 = pipeline.run(
        username="kyojuro",
        question="What are the treatment options?",
        session_id=sid
    )
