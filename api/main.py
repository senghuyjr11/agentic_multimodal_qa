"""
main.py - Simplified with TranslationAgent as front layer
"""
from image_agent import ImageAgent, ModelConfig
from pubmed_agent import PubMedAgent
from text_only_agent import TextOnlyAgent
from reasoning_agent import ReasoningAgent
from translation_agent import TranslationAgent
from session_manager import SessionManager
from transformers import Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration


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
            sessions_dir: str = "sessions"
    ):
        print("Initializing Medical VQA Pipeline...")

        # Translation Agent - FRONT LAYER
        self.translation_agent = TranslationAgent(api_key=google_api_key)

        self.session_manager = SessionManager(sessions_dir)

        self.image_agent = ImageAgent(
            pathvqa_config=pathvqa_config,
            vqa_rad_config=vqa_rad_config,
            classifier_path=classifier_path
        )

        self.text_only_agent = TextOnlyAgent(api_key=google_api_key)

        self.pubmed_agent = PubMedAgent(
            email=ncbi_email,
            api_key=ncbi_api_key
        )

        self.reasoning_agent = ReasoningAgent(api_key=google_api_key)

        print("âœ“ Pipeline ready\n")

    def run(
            self,
            username: str,
            question: str = None,
            image_path: str = None,
            session_id: int = None
    ) -> dict:
        """
        Main entry point - handles any language input.
        No need to specify language manually!
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

        # Create or load session
        if session_id is None:
            session_id = self.session_manager.create_session(
                username,
                question or "",
                image_path
            )
        else:
            if not self.session_manager.session_exists(username, session_id):
                raise ValueError(f"Session {username}/{session_id} not found")
            print(f"âœ“ Continuing session: {username}/{session_id}")

        # Save translation info
        self.session_manager.update(username, session_id, "translation", {
            "original_language": original_language,
            "original_question": question,
            "english_question": english_question
        })

        # Route based on input type
        # Route based on input type
        if input_type == "text_only":
            english_response = self._run_text_only(
                username, session_id, english_question, original_language  # ADD original_language
            )
        else:
            english_response = self._run_with_image(
                username, session_id, image_path, english_question, original_language  # ADD original_language
            )

        # ===== TRANSLATE BACK (if needed) =====
        final_response = self.translation_agent.process_output(
            english_response,
            original_language
        )

        self._print_final_output(final_response, username, session_id)

        return {
            "username": username,
            "session_id": session_id,
            "input_type": input_type,
            "original_language": original_language,
            "enhanced_response": final_response
        }

    def _run_with_image(
            self,
            username: str,
            session_id: int,
            image_path: str,
            english_question: str,
            original_language: str
    ) -> str:
        """Run full pipeline with image (all in English)."""

        # 1. Image Agent
        print("\n[1/3] Image Agent - Routing & Prediction")
        vqa_result = self.image_agent.predict(image_path, english_question)

        self.session_manager.update(username, session_id, "image_agent", {
            "routed_to": vqa_result["model"],
            "confidence": vqa_result["confidence"]
        })

        self.session_manager.update(username, session_id, "vqa_agent", {
            "question": vqa_result["question"],
            "answer": vqa_result["answer"]
        })

        print(f"Answer: {vqa_result['answer']}")

        # 2. PubMed Agent
        print("\n[2/3] PubMed Agent - Fetching Knowledge")
        knowledge = self.pubmed_agent.get_knowledge(
            vqa_answer=vqa_result["answer"],
            question=vqa_result["question"]
        )

        self.session_manager.update(username, session_id, "pubmed_agent", {
            "query": knowledge["query"],
            "articles_count": len(knowledge["articles"]),  # ADD THIS LINE
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
            pubmed_articles=knowledge["formatted"]
        )

        # Translate to user's language BEFORE saving
        final_response = self.translation_agent.process_output(
            english_response,
            original_language
        )

        self.session_manager.update(username, session_id, "reasoning_agent", {
            "response": final_response
        })

        return english_response

    def _run_text_only(
            self,
            username: str,
            session_id: int,
            english_question: str,
            original_language: str
    ) -> str:
        """Run pipeline for text-only input (all in English)."""

        # 1. Text-Only Agent (classify and respond)
        print("\n[1] Text-Only Agent - Classifying Question")
        text_only_result = self.text_only_agent.respond(english_question)

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
            # Translate response before saving
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
                "response": final_response  # Save translated version
            })

            return text_only_result["response"]  # Return English for processing

        # Medical question: continue with PubMed + Reasoning
        print("\n[2] PubMed Agent - Fetching Knowledge")
        knowledge = self.pubmed_agent.search_topic(english_question)

        self.session_manager.update(username, session_id, "pubmed_agent", {
            "query": knowledge["query"],
            "articles_count": len(knowledge["articles"]),  # ADD THIS LINE
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
        english_response = self.reasoning_agent.generate_response(
            question=english_question,
            vqa_answer="(Text-only question - no image analysis)",
            pubmed_articles=knowledge["formatted"]
        )

        # Translate to user's language BEFORE saving
        final_response = self.translation_agent.process_output(
            english_response,
            original_language
        )

        self.session_manager.update(username, session_id, "reasoning_agent", {
            "response": final_response  # Save translated version
        })

        return english_response  # Return English for processing

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

    # Test with different languages - NO NEED TO SPECIFY LANGUAGE!

    # Test 1: English (no translation needed)
    result = pipeline.run(
        username="kyojuro",
        question="What is adenocarcinoma?"
    )

    # Test 2: Korean (auto-detected and translated)
    # result = pipeline.run(
    #     username="kyojuro",
    #     question="ì„ ì•”ì´ ë­ì˜ˆìš”?"
    # )

    # Test 3: Image + Korean question
    # result = pipeline.run(
    #     username="kyojuro",
    #     image_path="../dataset_pathvqa/test/images/test_00001.jpg",
    #     question="ížˆìŠ¤í†¤ í•˜ìœ„ ë‹¨ìœ„ëŠ” ì–´ë–»ê²Œ ì¶©ì „ë˜ë‚˜ìš”?"
    # )