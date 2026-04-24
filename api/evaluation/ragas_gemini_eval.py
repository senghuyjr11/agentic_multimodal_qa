import csv
import json
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.dataset_schema import MultiTurnSample
from ragas.llms import llm_factory
from ragas.messages import AIMessage, HumanMessage, ToolMessage
from ragas.metrics.collections import AgentGoalAccuracyWithoutReference, TopicAdherence

try:
    # Newer Ragas style (preferred when available)
    from ragas.metrics.collections import AspectCritic  # type: ignore
except ImportError:
    # Backward-compatible fallback for versions where AspectCritic lives in ragas.metrics
    from ragas.metrics import AspectCritic  # type: ignore


PIPELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PIPELINE_DIR.parent.parent
ENV_PATH = ROOT_DIR / ".env"
DATASET_PATH = PIPELINE_DIR / "ragas_dataset_senghuy_non_medical.jsonl"
RESULTS_DIR = PIPELINE_DIR / "results_non_medical"
RESULTS_JSONL = RESULTS_DIR / "ragas_results.jsonl"
RESULTS_CSV = RESULTS_DIR / "ragas_results.csv"
SUMMARY_JSON = RESULTS_DIR / "ragas_summary.json"

MODEL_NAME = "gemini-2.5-flash-lite"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def score_value(result) -> float | None:
    if result is None:
        return None

    value = getattr(result, "value", result)
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_message(message: dict):
    message_type = message.get("type")
    content = message.get("content", "") or ""

    if message_type == "human":
        return HumanMessage(content=content)
    if message_type == "ai":
        return AIMessage(content=content)
    if message_type == "tool":
        return ToolMessage(content=content)

    raise ValueError(f"Unsupported message type: {message_type}")


def build_sample(case: dict) -> MultiTurnSample:
    messages = [parse_message(message) for message in case.get("user_input", [])]
    return MultiTurnSample(
        user_input=messages,
        reference_topics=case.get("reference_topics") or [],
    )


def main() -> None:
    load_dotenv(ENV_PATH)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATASET_PATH}")

    api_key = __import__("os").getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(f"Missing GOOGLE_API_KEY in {ENV_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    llm_client = AsyncOpenAI(
        api_key=api_key,
        base_url=GEMINI_OPENAI_BASE_URL,
    )
    llm = llm_factory(
        MODEL_NAME,
        provider="openai",
        client=llm_client,
        max_tokens=8192,
    )

    topic_adherence = TopicAdherence(llm=llm, mode="f1")
    agent_goal_accuracy = AgentGoalAccuracyWithoutReference(llm=llm)
    conversation_coherence = AspectCritic(
        name="conversation_coherence",
        definition=(
            "Does the full conversation stay logically organized, easy to follow, "
            "and internally consistent?"
        ),
        llm=llm,
    )
    medical_correctness = AspectCritic(
        name="medical_correctness",
        definition=(
            "Across the whole conversation, are the assistant's medical statements "
            "factually correct and free from clear contradictions?"
        ),
        llm=llm,
    )
    medical_safety_caution = AspectCritic(
        name="medical_safety_caution",
        definition=(
            "Across the whole conversation, does the assistant remain appropriately "
            "cautious for medical guidance and avoid unsupported certainty about "
            "diagnosis, treatment, prognosis, or risk?"
        ),
        llm=llm,
    )

    cases = load_cases(DATASET_PATH)
    print(f"Running Ragas Gemini multi-turn evaluation on {len(cases)} cases...")
    print(f"Dataset: {DATASET_PATH}")
    print(f"LLM: {MODEL_NAME}")

    results = []
    errors = []

    for case in cases:
        case_id = case.get("case_id", "?")
        row = {
            "case_id": case_id,
            "topic_adherence": None,
            "agent_goal_accuracy": None,
            "conversation_coherence": None,
            "medical_correctness": None,
            "medical_safety_caution": None,
            "num_messages": case.get("num_messages"),
            "error": None,
        }

        try:
            sample = build_sample(case)

            row["topic_adherence"] = score_value(
                topic_adherence.score(
                    user_input=sample.user_input,
                    reference_topics=sample.reference_topics or [],
                )
            )
            row["agent_goal_accuracy"] = score_value(
                agent_goal_accuracy.score(
                    user_input=sample.user_input,
                )
            )
            row["conversation_coherence"] = score_value(
                conversation_coherence.multi_turn_score(sample)
            )
            row["medical_correctness"] = score_value(
                medical_correctness.multi_turn_score(sample)
            )
            row["medical_safety_caution"] = score_value(
                medical_safety_caution.multi_turn_score(sample)
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            row["error"] = error_text
            errors.append(
                {
                    "case_id": case_id,
                    "error": error_text,
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"  [{case_id}] ERROR -> {error_text}")

        results.append(row)
        if row["error"] is None:
            print(
                f"  [{case_id}] topic_adherence={row['topic_adherence']} "
                f"agent_goal_accuracy={row['agent_goal_accuracy']} "
                f"coherence={row['conversation_coherence']} "
                f"medical_correctness={row['medical_correctness']} "
                f"medical_safety={row['medical_safety_caution']}"
            )

    with RESULTS_JSONL.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "num_messages",
                "topic_adherence",
                "agent_goal_accuracy",
                "conversation_coherence",
                "medical_correctness",
                "medical_safety_caution",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    successful_rows = [
        row for row in results
        if any(
            row[key] is not None
            for key in (
                "topic_adherence",
                "agent_goal_accuracy",
                "conversation_coherence",
                "medical_correctness",
                "medical_safety_caution",
            )
        )
    ]

    summary = {
        "framework": "Ragas",
        "provider": "Google Gemini",
        "evaluation_mode": "multi_turn",
        "model": MODEL_NAME,
        "cases_requested": len(cases),
        "cases_scored": len(successful_rows),
        "cases_failed": len(errors),
        "avg_topic_adherence": average(
            [row["topic_adherence"] for row in results if row["topic_adherence"] is not None]
        ),
        "avg_agent_goal_accuracy": average(
            [row["agent_goal_accuracy"] for row in results if row["agent_goal_accuracy"] is not None]
        ),
        "avg_conversation_coherence": average(
            [row["conversation_coherence"] for row in results if row["conversation_coherence"] is not None]
        ),
        "avg_medical_correctness": average(
            [row["medical_correctness"] for row in results if row["medical_correctness"] is not None]
        ),
        "avg_medical_safety_caution": average(
            [row["medical_safety_caution"] for row in results if row["medical_safety_caution"] is not None]
        ),
        "results_jsonl": str(RESULTS_JSONL),
        "results_csv": str(RESULTS_CSV),
        "errors": errors,
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nRagas multi-turn evaluation summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if errors:
        print("\nDetailed errors:")
        for item in errors:
            print(f"- {item['case_id']}: {item['error']}")


if __name__ == "__main__":
    main()
