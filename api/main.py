"""
main.py - Simplified with TranslationAgent as front layer
"""
from image_agent import ImageAgent, ModelConfig, OODConfig
from pubmed_agent import PubMedAgent
from text_only_agent import TextOnlyAgent
from reasoning_agent import ReasoningAgent
from translation_agent import TranslationAgent
from session_manager import SessionManager
from transformers import Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
import re

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

        self.session_manager = SessionManager(sessions_dir)

        # ADD THIS LINE - Memory storage for active conversations
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
            google_api_key=google_api_key  # â† ADD: Enable LLM-based term extraction
        )

        self.reasoning_agent = ReasoningAgent(api_key=google_api_key)

        print("âœ“ Pipeline ready\n")

    def get_or_create_memory(self, username: str, session_id: int) -> InMemoryChatMessageHistory:
        """
        Get memory from RAM if exists, otherwise load from JSON.
        This handles both: switching between chats + resuming old chats.
        """
        # Already in RAM?
        if session_id in self.active_conversations:
            print(f"âœ“ Using cached memory for session {session_id}")
            return self.active_conversations[session_id]

        # Not in RAM - need to restore from JSON
        print(f"âŸ³ Loading conversation history from JSON for session {session_id}")
        memory = self._restore_memory_from_json(username, session_id)

        # Cache it in RAM
        self.active_conversations[session_id] = memory

        return memory

    def _restore_memory_from_json(self, username: str, session_id: int) -> InMemoryChatMessageHistory:
        """Reconstruct LangChain memory from saved JSON conversation history."""

        # Get conversation history from JSON
        conversation_history = self.session_manager.get_conversation_history(username, session_id)

        # Create new memory
        memory = InMemoryChatMessageHistory()

        # Rebuild all previous turns
        for turn in conversation_history:
            memory.add_user_message(turn["user"])
            memory.add_ai_message(turn["assistant"])

        if conversation_history:
            print(f"âœ“ Restored {len(conversation_history)} conversation turns")
        else:
            print(f"âœ“ New conversation started")

        return memory

    def get_conversation_context(self, memory: InMemoryChatMessageHistory) -> str:
        """Extract conversation history as formatted string for prompts."""

        messages = memory.messages

        if not messages:
            return ""

        # Format as "Human: ...\nAI: ..."
        context_lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                context_lines.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_lines.append(f"AI: {msg.content}")

        return "\n".join(context_lines)

    def _extract_topic_from_context(self, conversation_context: str) -> str:
        """
        Extract the main medical topic from conversation context.
        Now looks at ALL previous turns, not just the first question.

        Strategy:
        1. Look at VQA answers (most reliable source)
        2. Look at all user questions
        3. Prioritize medical terms over general words
        """
        if not conversation_context:
            return None

        # Get all lines
        lines = conversation_context.split('\n')

        # Collect potential topics from different sources
        vqa_answers = []
        user_questions = []

        for line in lines:
            if line.startswith("AI:"):
                ai_content = line.replace("AI:", "").strip()
                # Check if this is a VQA answer (contains "Original VQA Detection:")
                if "Original VQA Detection:" in ai_content:
                    # Extract the VQA answer
                    match = re.search(r'Original VQA Detection:\*\*\s*([^\n*]+)', ai_content)
                    if match:
                        vqa_answer = match.group(1).strip()
                        vqa_answers.append(vqa_answer)
                        print(f"[DEBUG] Found VQA answer: '{vqa_answer}'")
            elif line.startswith("Human:"):
                question = line.replace("Human:", "").strip()
                user_questions.append(question)

        # Priority 1: Use VQA answer (most reliable)
        if vqa_answers:
            # Get the most recent VQA answer
            topic = vqa_answers[-1]
            # Clean up common VQA outputs
            topic = topic.replace("[Image uploaded]", "").strip()
            if topic and len(topic) > 2:
                print(f"[DEBUG] Extracted topic from VQA: '{topic}'")
                return topic

        # Priority 2: Extract from first user question (most likely contains topic)
        if user_questions:
            first_question = user_questions[0]
            print(f"[DEBUG] Extracting topic from first question: '{first_question}'")

            # Remove question words
            question_words = r'\b(what|how|why|when|where|who|which|is|are|can|could|would|should|does|do|did|the|a|an|this|that|these|those)\b'
            cleaned = re.sub(question_words, ' ', first_question, flags=re.IGNORECASE)

            # Extract remaining words (potential medical terms)
            words = re.findall(r'\b[A-Za-z]{3,}\b', cleaned)

            # Filter out remaining common words
            stopwords = {'about', 'there', 'their', 'have', 'has', 'had', 'been', 'being', 'image', 'uploaded'}
            medical_terms = [w for w in words if w.lower() not in stopwords]

            if medical_terms:
                topic = medical_terms[0]
                print(f"[DEBUG] Extracted topic from first question: '{topic}'")
                return topic

        print(f"[DEBUG] No topic found")
        return None

    def _enhance_question_with_context(self, question: str, conversation_context: str) -> str:
        """
        Enhance a question with context for better PubMed searches.
        Example: "How is it treated?" + context about pneumonia -> "How is pneumonia treated?"
        """
        if not conversation_context:
            return question

        # Extract topic from context
        topic = self._extract_topic_from_context(conversation_context)

        if not topic:
            print(f"[DEBUG] No topic found, using original question: '{question}'")
            return question

        # If question has pronouns like "it", "this", "that", replace with topic
        pronouns = r'\b(it|this|that|they|them)\b'
        if re.search(pronouns, question, re.IGNORECASE):
            # Replace pronoun with topic
            enhanced = re.sub(pronouns, topic.lower(), question, flags=re.IGNORECASE, count=1)
            print(f"âœ“ Enhanced question with context: '{question}' -> '{enhanced}'")
            return enhanced

        # If question is context-dependent but doesn't have pronouns, prepend topic
        context_dependent_patterns = [
            r'^(what|how|why|when|where)\s+(is|are|causes?|treatments?|symptoms?|does|do)',
            r'^(can|could|should|would|will)',
        ]

        for pattern in context_dependent_patterns:
            if re.match(pattern, question, re.IGNORECASE):
                enhanced = f"{topic} {question}"
                print(f"âœ“ Enhanced question with context: '{question}' -> '{enhanced}'")
                return enhanced

        return question

    def _get_state(self, session_id: int) -> dict:
        """
        Get session state for topic tracking.
        Returns a dict with 'topic' key for PubMed query enhancement.
        """
        # This is a simple implementation - you can enhance with actual topic extraction
        return {
            "topic": None  # Can be enhanced to extract topic from conversation history
        }

    def _search_pubmed_with_fallback(self, question: str, answer: str, topic: str | None, max_results: int = 3) -> dict | None:
        """
        Search PubMed with automatic fallback to broader searches.

        Strategy:
        1. Try specific search (with LLM-extracted terms)
        2. If fails, try broader search (topic + main keyword)
        3. If fails, try broadest search (topic only)
        4. If still fails, return None

        Args:
            question: User's question
            answer: VQA answer or None for text-only
            topic: Main topic extracted from context
            max_results: Number of articles to retrieve

        Returns:
            dict with 'query', 'articles', 'formatted' keys, or None if no articles found
        """

        # ATTEMPT 1: Specific search with LLM extraction
        print("[PubMed] Attempt 1: Specific search with LLM-extracted terms")

        if answer:
            # Image + text: use get_knowledge
            knowledge = self.pubmed_agent.get_knowledge(
                vqa_answer=answer,
                question=question,
                topic=topic,
                max_results=max_results
            )
        else:
            # Text-only: use search_topic
            knowledge = self.pubmed_agent.search_topic(
                question=question,
                topic=topic,
                max_results=max_results
            )

        if knowledge["articles"]:
            print(f"[PubMed] âœ“ Found {len(knowledge['articles'])} articles (specific search)")
            return knowledge

        # ATTEMPT 2: Broader search (topic + main keyword from question)
        if topic:
            print("[PubMed] Attempt 2: Broader search (topic + main keyword)")

            # Extract main keyword from question (simple approach)
            keywords = ["treatment", "causes", "symptoms", "diagnosis", "prevention", "recovery"]
            main_keyword = None
            for keyword in keywords:
                if keyword in question.lower():
                    main_keyword = keyword
                    break

            if main_keyword:
                query = f'("{topic}"[Title/Abstract]) AND ("{main_keyword}"[Title/Abstract])'
            else:
                # If no keyword found, just use topic
                query = f'("{topic}"[Title/Abstract])'

            articles = self.pubmed_agent.search(query, max_results=max_results * 2)

            if articles:
                print(f"[PubMed] âœ“ Found {len(articles)} articles (broader search)")
                return {
                    "query": query,
                    "articles": articles,
                    "formatted": self.pubmed_agent._format_output(articles)
                }

        # ATTEMPT 3: Broadest search (topic only)
        if topic:
            print("[PubMed] Attempt 3: Broadest search (topic only)")

            query = f'("{topic}"[Title/Abstract])'
            articles = self.pubmed_agent.search(query, max_results=max_results * 3)

            if articles:
                print(f"[PubMed] âœ“ Found {len(articles)} articles (broadest search)")
                return {
                    "query": query,
                    "articles": articles,
                    "formatted": self.pubmed_agent._format_output(articles)
                }

        # ALL ATTEMPTS FAILED
        print("[PubMed] âœ— No articles found after all attempts")
        return None

    def _handle_no_articles_found(self, username: str, session_id: int, topic: str | None) -> str:
        """
        Handle the case when PubMed returns no articles after all attempts.
        Returns an honest message to the user.
        """
        topic_text = f" about '{topic}'" if topic else ""

        honest_response = (
            f"I apologize, but I couldn't find peer-reviewed research articles "
            f"to answer your question{topic_text} with proper citations.\n\n"
            f"This could mean:\n"
            f"â€¢ The search terms were too specific\n"
            f"â€¢ Limited published research exists on this exact aspect\n"
            f"â€¢ The topic might need to be rephrased\n\n"
            f"ðŸ’¡ Suggestion: Try asking a more general question{topic_text}, "
            f"or rephrase your question differently.\n\n"
            f"For medical information, I can only provide answers backed by "
            f"published research to ensure accuracy and trustworthiness."
        )

        self.session_manager.update(username, session_id, "pubmed_agent", {
            "query": "multiple_attempts_failed",
            "articles": [],
            "attempts": 3,
            "status": "no_articles_found"
        })

        self.session_manager.update(username, session_id, "reasoning_agent", {
            "skipped": True,
            "reason": "no_pubmed_articles",
            "response": honest_response
        })

        return honest_response

    def run(
            self,
            username: str,
            question: str = None,
            image_path: str = None,
            session_id: int = None  # â† ADD THIS: for continuing conversations
    ) -> dict:
        """
        Main entry point - handles any language input.
        Now supports multi-turn conversations!

        Args:
            username: User identifier
            question: User's question (any language)
            image_path: Optional image file path
            session_id: Optional - if provided, continues existing conversation
        """

        if not image_path and not question:
            raise ValueError("Must provide at least image_path or question")

        print("=" * 60)
        print("MEDICAL VQA PIPELINE")
        print("=" * 60)

        # ===== TRANSLATION LAYER (FRONT) =====
        original_language = "English"
        english_question = question

        if question:
            translation_result = self.translation_agent.process_input(question)
            original_language = translation_result["detected_language"]
            english_question = translation_result["english_question"]

        # Determine input type
        if image_path:
            input_type = "image"
        else:
            input_type = "text_only"

        print(f"Input type: {input_type}")

        # ===== MEMORY MANAGEMENT =====
        # Create or continue session
        if session_id is None:
            # New conversation
            session_id = self.session_manager.create_session(
                username,
                question or "",
                image_path
            )
            memory = self.get_or_create_memory(username, session_id)
            print(f"âœ“ Started new session: {username}/{session_id}")
        else:
            # Continue existing conversation
            if not self.session_manager.session_exists(username, session_id):
                raise ValueError(f"Session {username}/{session_id} not found")
            memory = self.get_or_create_memory(username, session_id)
            print(f"âœ“ Continuing session: {username}/{session_id}")

        # Get conversation context for agents
        conversation_context = self.get_conversation_context(memory)

        # Debug: Print conversation context
        if conversation_context:
            print(f"[DEBUG] Conversation context:\n{conversation_context[:200]}...")

        # Save translation info
        self.session_manager.update(username, session_id, "translation", {
            "original_language": original_language,
            "original_question": question,
            "english_question": english_question
        })

        # Route based on input type
        if input_type == "text_only":
            english_response = self._run_text_only(
                username, session_id, english_question, original_language,
                conversation_context  # â† ADD THIS
            )
        else:
            english_response = self._run_with_image(
                username, session_id, image_path, english_question, original_language,
                conversation_context  # â† ADD THIS
            )

        # ===== TRANSLATE BACK (if needed) =====
        final_response = self.translation_agent.process_output(
            english_response,
            original_language
        )

        # ===== SAVE TO MEMORY =====
        # Add to LangChain memory (RAM)
        memory.add_user_message(question if question else "[Image uploaded]")
        memory.add_ai_message(final_response)

        # Add to JSON (Disk) - Include image path for this turn
        self.session_manager.add_conversation_turn(
            username,
            session_id,
            question if question else "[Image uploaded]",
            final_response,
            image_path=image_path  # â† ADD: Pass image path for this turn
        )

        self._print_final_output(final_response, username, session_id)

        # FIX: Get turn count from JSON (source of truth)
        session_data = self.session_manager.load(username, session_id)
        current_turn = len(session_data.get("conversation_history", []))

        return {
            "username": username,
            "session_id": session_id,
            "input_type": input_type,
            "original_language": original_language,
            "enhanced_response": final_response,
            "turn": current_turn  # â† FIXED: Use JSON as source of truth
        }

    def _run_with_image(
            self,
            username: str,
            session_id: int,
            image_path: str,
            english_question: str,
            original_language: str,
            conversation_context: str = ""  # â† ADD THIS
    ) -> str:
        """Run full pipeline with image (all in English)."""

        # 1. Image Agent
        print("\n[1/3] Image Agent - Routing & Prediction")
        vqa_result = self.image_agent.predict(image_path, english_question)

        # Save routing info (include OOD fields)
        self.session_manager.update(username, session_id, "image_agent", {
            "routed_to": vqa_result.get("model"),
            "confidence": vqa_result.get("confidence"),
            "ood": vqa_result.get("ood", False),
            "ood_rule": vqa_result.get("ood_rule", None),
        })

        self.session_manager.update(username, session_id, "vqa_agent", {
            "question": vqa_result.get("question"),
            "answer": vqa_result.get("answer"),
            "ood": vqa_result.get("ood", False),
            "original_output": vqa_result.get("answer")  # Save original answer
        })

        print(f"VQA Answer: {vqa_result['answer']}")  # Show in console

        # ===== OOD SHORT-CIRCUIT =====
        if vqa_result.get("ood", False) or vqa_result.get("model") == "unknown":
            # Skip PubMed + Reasoning
            self.session_manager.update(username, session_id, "pubmed_agent", {
                "skipped": True,
                "reason": "ood_rejection"
            })
            self.session_manager.update(username, session_id, "reasoning_agent", {
                "skipped": True,
                "reason": "ood_rejection",
                # Save translated response for user convenience
                "response": self.translation_agent.process_output(vqa_result["answer"], original_language)
            })

            # Return English for translation layer (run() will translate back)
            return vqa_result["answer"]

        # 2. PubMed Agent - With fallback search strategy
        print("\n[2/3] PubMed Agent - Fetching Knowledge")

        # Extract topic from conversation context
        topic = self._extract_topic_from_context(conversation_context)

        # Use fallback search strategy (tries specific â†’ broader â†’ broadest)
        knowledge = self._search_pubmed_with_fallback(
            question=vqa_result["question"],
            answer=vqa_result["answer"],
            topic=topic,
            max_results=3
        )

        # Check if articles were found after all attempts
        if knowledge is None or not knowledge["articles"]:
            print("[PubMed] No articles found - returning honest response")
            return self._handle_no_articles_found(username, session_id, topic)

        self.session_manager.update(username, session_id, "pubmed_agent", {
            "query": knowledge["query"],
            "articles": [
                {
                    "title": a.title,
                    "abstract": a.abstract,
                    "pmid": a.pmid,
                    "url": a.url
                } for a in knowledge["articles"]
            ]
        })

        print(f"Found {len(knowledge['articles'])} articles")

        # 3. Reasoning Agent
        print(f"\n[3/3] Reasoning Agent - Generating Explanation")

        english_response = self.reasoning_agent.generate_response(
            question=vqa_result["question"],
            vqa_answer=vqa_result["answer"],
            pubmed_articles=knowledge["formatted"],  # String for context
            article_objects=knowledge["articles"]  # List for links
        )

        # Prepend original VQA output for transparency
        original_vqa_section = f"**Original VQA Detection:** {vqa_result['answer']}\n\n---\n\n"
        english_response = original_vqa_section + english_response

        # Translate to user's language BEFORE saving
        final_response = self.translation_agent.process_output(
            english_response,
            original_language
        )

        self.session_manager.update(username, session_id, "reasoning_agent", {
            "original_vqa_output": vqa_result["answer"],  # Save original
            "enhanced_response": final_response
        })

        return english_response

    def _run_text_only(
            self,
            username: str,
            session_id: int,
            english_question: str,
            original_language: str,
            conversation_context: str = ""
    ) -> str:
        """Run pipeline for text-only input (all in English)."""

        # 1. Text-Only Agent (classify and respond)
        print("\n[1] Text-Only Agent - Classifying Question")

        # Pass conversation context to text-only agent
        text_only_result = self.text_only_agent.respond(
            english_question,
            conversation_context
        )

        # Save text-only agent result (only save keys that exist)
        self.session_manager.update(username, session_id, "text_agent", {
            "question": text_only_result.get("question"),
            "question_type": text_only_result.get("question_type")
        })

        self.session_manager.update(username, session_id, "image_agent", {
            "skipped": True,
            "reason": "text_only_input"
        })

        self.session_manager.update(username, session_id, "vqa_agent", {
            "skipped": True,
            "reason": "text_only_input"
        })

        # If casual question, skip PubMed and Reasoning
        if text_only_result["question_type"] == "casual":
            final_response = self.translation_agent.process_output(
                text_only_result["response"],
                original_language
            )

            self.session_manager.update(username, session_id, "pubmed_agent", {
                "skipped": True,
                "reason": "casual_question"
            })

            self.session_manager.update(username, session_id, "reasoning_agent", {
                "skipped": True,
                "reason": "casual_question",
                "response": final_response
            })

            return text_only_result["response"]

        # Medical question: continue with PubMed + Reasoning
        print("\n[2] PubMed Agent - Fetching Knowledge")

        # ===== FIX: Enhance question with conversation context =====
        enhanced_question = self._enhance_question_with_context(
            english_question,
            conversation_context
        )

        # Extract topic from context
        topic = self._extract_topic_from_context(conversation_context)

        print(f"[DEBUG] Using enhanced question: '{enhanced_question}' with topic: '{topic}'")

        # Use fallback search strategy (tries specific â†’ broader â†’ broadest)
        knowledge = self._search_pubmed_with_fallback(
            question=enhanced_question,
            answer=None,  # No VQA answer for text-only
            topic=topic,
            max_results=3
        )

        # Check if articles were found after all attempts
        if knowledge is None or not knowledge["articles"]:
            print("[PubMed] No articles found - returning honest response")
            return self._handle_no_articles_found(username, session_id, topic)

        self.session_manager.update(username, session_id, "pubmed_agent", {
            "query": knowledge["query"],
            "articles": [
                {
                    "title": a.title,
                    "abstract": a.abstract,
                    "pmid": a.pmid,
                    "url": a.url
                } for a in knowledge["articles"]
            ]
        })

        print(f"Found {len(knowledge['articles'])} articles")

        print(f"\n[3] Reasoning Agent - Generating Explanation")

        # Pass conversation context to reasoning agent
        english_response = self.reasoning_agent.generate_response(
            question=english_question,
            vqa_answer="(Text-only question - no image analysis)",
            pubmed_articles=knowledge["formatted"],
            article_objects=knowledge["articles"],
            conversation_context=conversation_context  # â† ADD THIS
        )

        final_response = self.translation_agent.process_output(
            english_response,
            original_language
        )

        self.session_manager.update(username, session_id, "reasoning_agent", {
            "response": final_response
        })

        return english_response

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
        ncbi_email="senghuymit007@gmail.com",
        ncbi_api_key="92da6b3a9eb8f5916e252e7fbc9d9aed3609",
        google_api_key="GOOGLE_API_KEY_REMOVED",
        pathvqa_config=pathvqa_config,
        vqa_rad_config=vqa_rad_config,
        classifier_path="../modality_classifier"
    )

    # Test conversation with context awareness
    print("\n### TEST: Context-Aware Conversation ###")

    # Turn 1
    result1 = pipeline.run(
        username="kyojuro",
        question="What is pneumonia?"
    )
    session_id = result1["session_id"]

    # Turn 2 - Should understand "it" refers to pneumonia
    result2 = pipeline.run(
        username="kyojuro",
        question="What causes it?",
        session_id=session_id
    )

    # Turn 3 - Should understand context from previous turns
    result3 = pipeline.run(
        username="kyojuro",
        question="What are the treatment options?",
        session_id=session_id
    )