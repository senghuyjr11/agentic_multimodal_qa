"""
main.py - Full pipeline connecting all agents
"""
from image_agent import ImageAgent, ModelConfig
from text_agent import TextAgent
from reasoning_agent import ReasoningAgent
from session_manager import SessionManager
from transformers import Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration


class MedicalVQAPipeline:
    """Connects all agents and manages session storage."""

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

        self.session_manager = SessionManager(sessions_dir)

        self.image_agent = ImageAgent(
            pathvqa_config=pathvqa_config,
            vqa_rad_config=vqa_rad_config,
            classifier_path=classifier_path
        )

        self.text_agent = TextAgent(
            email=ncbi_email,
            api_key=ncbi_api_key
        )

        self.reasoning_agent = ReasoningAgent(
            api_key=google_api_key
        )

        print("âœ“ Pipeline ready\n")

    def run(
            self,
            username: str,
            image_path: str,
            question: str,
            language: str = "English",
            session_id: int = None  # None = new chat, number = continue chat
    ) -> dict:
        """Run full pipeline and store results."""

        print("=" * 60)
        print("MEDICAL VQA PIPELINE")
        print("=" * 60)

        # Create or load session
        if session_id is None:
            session_id = self.session_manager.create_session(username, image_path, question)
        else:
            if not self.session_manager.session_exists(username, session_id):
                raise ValueError(f"Session {username}/{session_id} not found")
            print(f"âœ“ Continuing session: {username}/{session_id}")

        # 1. Image Agent
        print("\n[1/3] Image Agent - Routing & Prediction")
        vqa_result = self.image_agent.predict(image_path, question)

        self.session_manager.update(username, session_id, "image_agent", {
            "routed_to": vqa_result["model"],
            "confidence": vqa_result["confidence"]
        })

        self.session_manager.update(username, session_id, "vqa_agent", {
            "question": vqa_result["question"],
            "answer": vqa_result["answer"]
        })

        print(f"Answer: {vqa_result['answer']}")

        # 2. Text Agent
        print("\n[2/3] Text Agent - Fetching PubMed Knowledge")
        knowledge = self.text_agent.get_knowledge(
            vqa_answer=vqa_result["answer"],
            question=vqa_result["question"]
        )

        self.session_manager.update(username, session_id, "text_agent", {
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
        print(f"\n[3/3] Reasoning Agent - Generating Explanation ({language})")
        reasoning_result = self.reasoning_agent.generate_response(
            question=vqa_result["question"],
            vqa_answer=vqa_result["answer"],
            pubmed_articles=knowledge["formatted"],
            language=language
        )

        self.session_manager.update(username, session_id, "reasoning_agent", {
            "language": language,
            "model_used": reasoning_result["model_used"],
            "response": reasoning_result["enhanced_response"]
        })

        # Final output
        print("\n" + "=" * 60)
        print("FINAL OUTPUT")
        print("=" * 60)
        print(reasoning_result["enhanced_response"])
        print("=" * 60)
        print(f"Session saved: {username}/{session_id}")

        return {
            "username": username,
            "session_id": session_id,
            "vqa_answer": vqa_result["answer"],
            "enhanced_response": reasoning_result["enhanced_response"]
        }

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

    # Initialize pipeline
    pipeline = MedicalVQAPipeline(
        ncbi_email="senghuymit007@gmail.com",
        ncbi_api_key="92da6b3a9eb8f5916e252e7fbc9d9aed3609",
        google_api_key="GOOGLE_API_KEY_REMOVED",
        pathvqa_config=pathvqa_config,
        vqa_rad_config=vqa_rad_config,
        classifier_path="../modality_classifier"
    )

    # Run
    result = pipeline.run(
        username="kyojuro",
        image_path="../dataset_pathvqa/test/images/test_00001.jpg",
        question="how are the histone subunits charged?",
        language="English"
    )