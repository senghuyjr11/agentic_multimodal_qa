"""
main_simple.py - Clean pipeline coordinator

ONE JOB: Coordinate agents (don't do the work yourself!)

This file is like a project manager:
- Calls the right agents
- Passes data between them
- Returns the final result

Each agent does ONE thing well.
"""

import os
from typing import Optional

from runtime_config import (
    CLASSIFIER_MODEL_DIR,
    PATHVQA_ADAPTER_DIR,
    VQA_RAD_ADAPTER_DIR,
    configure_runtime_environment,
)

configure_runtime_environment()

# Import our simple agents
from agents.image_agent import ModelConfig, ImageAgent
from agents.conversation_summarizer import ConversationSummarizer
from agents.memory_manager import MemoryManager
from agents.pubmed_agent import PubMedAgent
from agents.response_generator import ResponseGenerator
from agents.router_agent import RouterAgent
from agents.session_manager import SessionManager
from agents.translation_agent import TranslationAgent


class MedicalVQAPipeline:
    """
    Coordinates all agents to answer medical questions.

    Does NOT do the actual work - just coordinates!
    """

    def __init__(
        self,
        ncbi_email: str,
        ncbi_api_key: str,
        google_api_key: str,
        pathvqa_config: ModelConfig,
        vqa_rad_config: ModelConfig,
        classifier_path: str
    ):
        print("Initializing Medical VQA Pipeline...")

        # Initialize all agents (each does ONE job)
        self.router = RouterAgent(google_api_key)
        self.response_gen = ResponseGenerator(google_api_key)
        self.summarizer = ConversationSummarizer(google_api_key)
        self.memory = MemoryManager()
        self.translator = TranslationAgent()
        self.session_mgr = SessionManager()

        self.image_agent = ImageAgent(
            pathvqa_config=pathvqa_config,
            vqa_rad_config=vqa_rad_config,
            classifier_path=classifier_path,
        )

        self.pubmed_agent = PubMedAgent(
            email=ncbi_email,
            api_key=ncbi_api_key,
            google_api_key=google_api_key
        )

        print("✓ All agents initialized\n")

    def run(
        self,
        username: str,
        question: Optional[str],
        image_path: Optional[str] = None,
        session_id: Optional[int] = None
    ) -> dict:
        """
        ONE JOB: Coordinate agents to answer the question.

        Flow:
        1. Translate input
        2. Get/create memory
        3. Router decides what to do
        4. Execute (Image? PubMed?)
        5. Generate response
        6. Translate output
        7. Save to memory + disk

        Returns:
            {
                "response": "answer text",
                "session_id": 123,
                "metadata": {...}
            }
        """

        print(f"\n{'='*60}")
        print(f"Processing question for {username}")
        print(f"{'='*60}")

        # Track if this was an image-only upload (no user question)
        is_image_only = (not question or not question.strip()) and image_path is not None
        effective_question = question.strip() if question and question.strip() else None
        display_question = "[Image Uploaded]" if is_image_only else (effective_question or "")

        # ==========================================
        # STEP 1: TRANSLATE INPUT
        # ==========================================
        print("\n[Step 1] Translation...")
        if effective_question is None:
            translation_result = {
                "english_question": None,
                "source_language": "en",
                "output_language": "en",
            }
            english_question = None
            source_lang = "en"
            output_lang = "en"
            print("  Image-only upload → skipping text translation")
        else:
            translation_result = self.translator.process_input(effective_question)
            english_question = translation_result["english_question"]
            source_lang = translation_result["source_language"]
            output_lang = translation_result["output_language"]

        print(f"  Source: {source_lang}")
        print(f"  English: {english_question if english_question is not None else '[Image-only]'}")
        print(f"  Display: {display_question}")

        # ==========================================
        # STEP 2: GET/CREATE MEMORY & SESSION
        # ==========================================
        print("\n[Step 2] Memory & Session...")

        if session_id is None:
            # New session
            session_id = self.session_mgr.create_session(
                username=username,
                question=display_question,
                image_path=image_path
            )
            conversation_history = []
            memory_state = self.session_mgr.get_memory_state(username, session_id)
            print(f"  Created new session: {session_id}")
        else:
            # Continuing session
            conversation_history = self.session_mgr.get_conversation_history(
                username, session_id
            )
            memory_state = self.session_mgr.get_memory_state(username, session_id)
            print(f"  Continuing session: {session_id}")

        # **CHECK THAI BLOCKING AFTER SESSION EXISTS**
        if english_question == "THAI_LANGUAGE_BLOCKED":
            print("  Thai language blocked - saving message to session")

            blocked_message = "Sorry, this system does not support Thai language."

            self.session_mgr.add_conversation_turn(
                username=username,
                session_id=session_id,
                user_message=effective_question or display_question,
                assistant_message=blocked_message,
                image_path=image_path,
                meta={
                    "translation": {"source_language": "th-BLOCKED", "output_language": "en"},
                    "blocked": True,
                    "reason": "thai_language_blocked"
                }
            )

            # Keep in-memory conversation in sync
            if session_id in self.memory.active_sessions:
                self.memory.add_turn(session_id, display_question, blocked_message)

            return {
                "response": blocked_message,
                "session_id": session_id,
                "metadata": {"blocked": True, "reason": "thai_language_blocked"}
            }

        # **CHECK COMING-SOON LANGUAGES (e.g. Khmer input)**
        if english_question and english_question.startswith("COMING_SOON_LANGUAGE:"):
            lang_code = english_question.split(":")[1]
            lang_name = self.translator.get_language_name(lang_code)
            print(f"  {lang_name} coming soon - saving message to session")

            coming_soon_message = (
                f"Thank you for using {lang_name}! "
                f"We are currently working on adding {lang_name} language support. "
                f"It will be available soon. Please use English or another supported language for now."
            )

            self.session_mgr.add_conversation_turn(
                username=username,
                session_id=session_id,
                user_message=effective_question or display_question,
                assistant_message=coming_soon_message,
                image_path=image_path,
                meta={
                    "translation": {"source_language": lang_code, "output_language": "en"},
                    "coming_soon": True,
                    "reason": f"{lang_code}_coming_soon"
                }
            )

            # Keep in-memory conversation in sync
            if session_id in self.memory.active_sessions:
                self.memory.add_turn(session_id, display_question, coming_soon_message)

            return {
                "response": coming_soon_message,
                "session_id": session_id,
                "metadata": {"coming_soon": True, "reason": f"{lang_code}_coming_soon"}
            }

        # **CHECK UNSUPPORTED LANGUAGE**
        if english_question and english_question.startswith("UNSUPPORTED_LANGUAGE:"):
            detected_code = english_question.split(":")[1]
            supported_names = translation_result.get("supported_languages", "")
            print(f"  Unsupported language '{detected_code}' - saving message to session")

            unsupported_message = (
                f"Sorry, your language is not supported yet. "
                f"Please use one of the following supported languages: "
                f"English, {supported_names}."
            )

            self.session_mgr.add_conversation_turn(
                username=username,
                session_id=session_id,
                user_message=effective_question or display_question,
                assistant_message=unsupported_message,
                image_path=image_path,
                meta={
                    "translation": {"source_language": detected_code, "output_language": "en"},
                    "unsupported": True,
                    "reason": f"{detected_code}_not_supported"
                }
            )

            return {
                "response": unsupported_message,
                "session_id": session_id,
                "metadata": {"unsupported": True, "reason": f"{detected_code}_not_supported"}
            }

        # **CHECK IF USER ASKS IN ENGLISH TO TRANSLATE TO A COMING-SOON LANGUAGE**
        if english_question:
            question_lower = english_question.lower()
            for lang_code in self.translator.COMING_SOON_LANGUAGES:
                lang_name = self.translator.get_language_name(lang_code).lower()
                if lang_name in question_lower and any(
                    w in question_lower for w in ['translate', 'in ' + lang_name, 'to ' + lang_name]
                ):
                    print(f"  User requested {lang_name} translation → coming soon")

                    coming_soon_message = (
                        f"We are currently working on adding {self.translator.get_language_name(lang_code)} "
                        f"language support. It will be available soon!"
                    )

                    self.session_mgr.add_conversation_turn(
                        username=username,
                        session_id=session_id,
                        user_message=effective_question or display_question,
                        assistant_message=coming_soon_message,
                        image_path=image_path,
                        meta={
                            "translation": {"source_language": "en", "output_language": lang_code},
                            "coming_soon": True,
                            "reason": f"{lang_code}_coming_soon"
                        }
                    )

                    return {
                        "response": coming_soon_message,
                        "session_id": session_id,
                        "metadata": {"coming_soon": True, "reason": f"{lang_code}_coming_soon"}
                    }

        # Get memory for this session
        memory = self.memory.get_or_create(session_id, conversation_history, memory_state)

        compaction_result = self.memory.compact_if_needed(
            session_id=session_id,
            summarizer=self.summarizer,
            full_conversation_history=conversation_history,
            persist_callback=lambda state: self.session_mgr.update_memory_state(
                username, session_id, state
            )
        )

        if compaction_result.get("compacted"):
            memory = self.memory.get_or_create(
                session_id,
                conversation_history,
                self.session_mgr.get_memory_state(username, session_id)
            )

        # ==========================================
        # STEP 3: ROUTER DECIDES
        # ==========================================
        print("\n[Step 3] Router deciding...")

        decision = self.router.decide(
            message=english_question or "[Image Uploaded]",
            has_image=bool(image_path),
            memory=memory
        )

        # NEW: Hard override - if image present, force VQA
        if bool(image_path) and not decision.needs_vqa:
            print(f"  ⚠️  Override: Image detected, forcing VQA")
            decision.needs_vqa = True
            if decision.response_mode == "casual_chat":
                decision.response_mode = "medical_answer"

        # Check 1: User asking to elaborate on existing references (MOVED UP - HIGHER PRIORITY)
        use_cached_articles = False
        if english_question and self.router.is_asking_about_previous_references(english_question):
            cached_articles = self.memory.get_pubmed_articles(session_id)

            if cached_articles:
                print(f"\n[REUSING ARTICLES] User asking to elaborate on previous response")
                print(f"  Found {len(cached_articles)} cached articles")
                print(f"  Overriding router decision - will NOT search PubMed")

                # Override router's decision
                decision.needs_pubmed = True  # We need PubMed, but using cache
                decision.search_query = None  # Clear search query to prevent new search
                decision.response_mode = "medical_answer"
                use_cached_articles = True

        # Check 2: Follow-up asking for explanation of previous short answer
        elif not decision.needs_pubmed and not decision.needs_vqa:
            needs_pubmed_followup, search_query = self.router.detect_followup_needs_pubmed(
                message=english_question or "",
                memory=memory
            )

            if needs_pubmed_followup:
                decision.needs_pubmed = True
                decision.search_query = search_query
                decision.response_mode = "medical_answer"
                print(f"  → Follow-up detected, will search PubMed")

        # ==========================================
        # STEP 4: EXECUTE BASED ON DECISION
        # ==========================================
        vqa_answer = None
        pubmed_articles = []

        # 4a. Image analysis?
        if decision.needs_vqa and image_path:
            print("\n[Step 4a] Image analysis...")
            vqa_result = self.image_agent.predict(image_path, english_question)

            if vqa_result.get("ood", False):
                print("  Image rejected (OOD)")
                # Don't return early — set vqa_answer to rejection message
                # and let it fall through to Step 5, 6, 7 to save normally
                vqa_answer = vqa_result["answer"]
                decision.needs_pubmed = False
                decision.search_query = None
            else:
                vqa_answer = vqa_result["answer"]
                print(f"  VQA: {vqa_answer}")
                decision.needs_pubmed = False
                decision.search_query = None

            vqa_answer = vqa_result["answer"]
            print(f"  VQA: {vqa_answer}")

            # NEW SIMPLE LOGIC: Just show VQA answer, don't search PubMed
            # User will ask follow-up if confused
            print("  → Image answered by VQA model")
            print("  → Skipping PubMed (user can ask follow-up for explanation)")
            decision.needs_pubmed = False
            decision.search_query = None

        # 4b. PubMed search?
        if decision.needs_pubmed:
            # Case 1: Reuse cached articles
            if use_cached_articles:
                pubmed_articles = self.memory.get_pubmed_articles(session_id)
                print(f"  Using {len(pubmed_articles)} cached articles (no new search)")

            # Case 2: Search PubMed (only for text-only questions, not images)
            elif decision.search_query and not vqa_answer:
                print(f"\n[Step 4b] PubMed search: '{decision.search_query}'")
                articles = self.pubmed_agent.search(
                    decision.search_query,
                    max_results=5
                )

                if articles:
                    pubmed_articles = self.pubmed_agent.score_articles(
                    query=english_question or "medical image finding",
                    articles=articles
                )
                    print(f"  Found {len(pubmed_articles)} articles")

                    # Store articles in memory for future reference
                    self.memory.store_pubmed_articles(session_id, pubmed_articles)
                else:
                    print("  No articles found")

                    # Try using cached articles from previous turn
                    cached_articles = self.memory.get_pubmed_articles(session_id)
                    if cached_articles:
                        print(f"  → Using {len(cached_articles)} cached articles from previous turn")
                        pubmed_articles = cached_articles

        # ==========================================
        # STEP 5: GENERATE RESPONSE
        # ==========================================
        print("\n[Step 5] Generating response...")

        # Get previous response if needed for modification
        previous_response = None
        if decision.response_mode == "modify_previous":
            previous_response = self.memory.get_last_ai_message(session_id)

        # Generate response
        english_response = self.response_gen.generate(
            message=english_question or "[Image Uploaded]",
            response_mode=decision.response_mode,
            vqa_answer=vqa_answer,
            pubmed_articles=pubmed_articles,
            memory=memory,
            previous_response=previous_response,
            has_image=bool(image_path)
        )

        # ==========================================
        # STEP 6: TRANSLATE OUTPUT
        # ==========================================
        print("\n[Step 6] Translation...")

        # Skip translation for casual chat - keep it in English
        if decision.response_mode == "casual_chat":
            final_response = english_response
            output_lang = "en"
            print("  Casual chat - keeping in English")
        else:
            final_response = self.translator.process_output(
                english_response,
                output_lang
            )

        # ==========================================
        # STEP 7: SAVE TO MEMORY & DISK
        # ==========================================
        print("\n[Step 7] Saving...")

        # Add to memory - use display_question so history makes sense
        self.memory.add_turn(
            session_id=session_id,
            user_message=display_question,  # ← was english_question, causes "[Image Uploaded]" to be stored correctly
            ai_message=english_response
        )

        metadata = {
            "translation": {
                "source_language": source_lang,
                "output_language": output_lang
            },
            "decision": {
                "response_mode": decision.response_mode,
                "needs_pubmed": decision.needs_pubmed,
                "search_query": decision.search_query,
                "reasoning": decision.reasoning
            },
            "vqa_answer": vqa_answer,
            "num_articles": len(pubmed_articles),
            "memory": {
                **self.memory.get_context_status(session_id),
                "summary_updated": compaction_result.get("compacted", False)
            },
            "articles": [
                {
                    "title": a.title,
                    "pmid": a.pmid,
                    "url": a.url,
                    "relevance": getattr(a, 'relevance_score', None)
                }
                for a in pubmed_articles[:5]
            ] if pubmed_articles else []
        }

        self.session_mgr.add_conversation_turn(
            username=username,
            session_id=session_id,
            user_message=display_question,  # Show [Image Uploaded] to user
            assistant_message=final_response,
            image_path=image_path,
            meta=metadata
        )

        self.session_mgr.update_memory_state(
            username=username,
            session_id=session_id,
            memory_state=self.memory.active_sessions[session_id].memory_state()
        )

        print("✓ Saved to disk")

        # ==========================================
        # DONE!
        # ==========================================
        print(f"\n{'='*60}")
        print("✓ Complete!")
        print(f"{'='*60}\n")

        return {
            "response": final_response,
            "session_id": session_id,
            "metadata": metadata
        }


if __name__ == "__main__":
    # Example usage
    from transformers import Qwen3VLForConditionalGeneration

    pathvqa_config = ModelConfig(
        base_model_id="Qwen/Qwen3-VL-8B-Instruct",
        adapter_path=str(PATHVQA_ADAPTER_DIR),
        model_class=Qwen3VLForConditionalGeneration,
    )

    vqa_rad_config = ModelConfig(
        base_model_id="Qwen/Qwen3-VL-8B-Instruct",
        adapter_path=str(VQA_RAD_ADAPTER_DIR),
        model_class=Qwen3VLForConditionalGeneration,
    )

    pipeline = MedicalVQAPipeline(
        ncbi_email=os.getenv("NCBI_EMAIL"),
        ncbi_api_key=os.getenv("NCBI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        pathvqa_config=pathvqa_config,
        vqa_rad_config=vqa_rad_config,
        classifier_path=str(CLASSIFIER_MODEL_DIR)
    )

    # Test
    result = pipeline.run(
        username="test",
        question="What is diabetes?"
    )

    print(f"\nFinal response: {result['response'][:200]}...")
