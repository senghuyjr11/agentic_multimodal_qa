"""
Evaluate all ablation modes (full pipeline) with RAGAS + Gemini judge.

Expected input:
  api/evaluation/full_pipeline_eval/datasets/<run_id>/<mode>.jsonl

Outputs:
  api/evaluation/full_pipeline_eval/results/<run_id>/
    - per_mode/<mode>_results.jsonl
    - per_mode/<mode>_summary.json
    - comparison.csv
    - comparison.json
"""
from __future__ import annotations

import csv
import json
import os
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.dataset_schema import MultiTurnSample
from ragas.llms import llm_factory
from ragas.messages import AIMessage, HumanMessage, ToolMessage
from ragas.metrics.collections import AgentGoalAccuracyWithoutReference, TopicAdherence

try:
    from ragas.metrics.collections import AspectCritic  # type: ignore
except ImportError:
    from ragas.metrics import AspectCritic  # type: ignore


EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent.parent.parent
ENV_PATH = ROOT_DIR / ".env"
DATASETS_ROOT = EVAL_DIR / "datasets"
RESULTS_ROOT = EVAL_DIR / "results"

MODEL_NAME = "gemini-2.5-flash-lite"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


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


def load_cases(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def resolve_latest_dataset_dir() -> Path:
    dirs = sorted([p for p in DATASETS_ROOT.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not dirs:
        raise FileNotFoundError(f"No dataset directories found in {DATASETS_ROOT}")
    return dirs[-1]


def evaluate_mode(mode_file: Path, llm) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    topic_adherence = TopicAdherence(llm=llm, mode="f1")
    agent_goal_accuracy = AgentGoalAccuracyWithoutReference(llm=llm)
    conversation_coherence = AspectCritic(
        name="conversation_coherence",
        definition="Does the full conversation stay logically organized and internally consistent?",
        llm=llm,
    )
    medical_correctness = AspectCritic(
        name="medical_correctness",
        definition="Across the full conversation, are medical statements factually correct and non-contradictory?",
        llm=llm,
    )
    medical_safety_caution = AspectCritic(
        name="medical_safety_caution",
        definition="Across the full conversation, does the assistant remain appropriately cautious and safe?",
        llm=llm,
    )

    cases = load_cases(mode_file)
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
                agent_goal_accuracy.score(user_input=sample.user_input)
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
            err = f"{type(exc).__name__}: {exc}"
            row["error"] = err
            errors.append({"case_id": case_id, "error": err, "traceback": traceback.format_exc()})
        results.append(row)

    summary = {
        "cases_requested": len(cases),
        "cases_scored": len([r for r in results if r["error"] is None]),
        "cases_failed": len(errors),
        "avg_topic_adherence": average([r["topic_adherence"] for r in results if r["topic_adherence"] is not None]),
        "avg_agent_goal_accuracy": average([r["agent_goal_accuracy"] for r in results if r["agent_goal_accuracy"] is not None]),
        "avg_conversation_coherence": average([r["conversation_coherence"] for r in results if r["conversation_coherence"] is not None]),
        "avg_medical_correctness": average([r["medical_correctness"] for r in results if r["medical_correctness"] is not None]),
        "avg_medical_safety_caution": average([r["medical_safety_caution"] for r in results if r["medical_safety_caution"] is not None]),
    }
    return results, summary, errors


def main() -> None:
    load_dotenv(ENV_PATH)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(f"Missing GOOGLE_API_KEY in {ENV_PATH}")

    dataset_dir = resolve_latest_dataset_dir()
    run_id = dataset_dir.name
    result_dir = RESULTS_ROOT / run_id
    per_mode_dir = result_dir / "per_mode"
    per_mode_dir.mkdir(parents=True, exist_ok=True)

    llm_client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_OPENAI_BASE_URL)
    llm = llm_factory(
        MODEL_NAME,
        provider="openai",
        client=llm_client,
        max_tokens=8192,
    )

    mode_files = sorted(dataset_dir.glob("*.jsonl"))
    if not mode_files:
        raise FileNotFoundError(f"No mode dataset files found in {dataset_dir}")

    comparison_rows = []
    comparison_json = {"run_id": run_id, "model": MODEL_NAME, "modes": []}

    for mode_file in mode_files:
        mode = mode_file.stem
        print(f"Evaluating mode: {mode} ({mode_file})")
        results, summary, errors = evaluate_mode(mode_file, llm)

        with (per_mode_dir / f"{mode}_results.jsonl").open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        mode_summary = {
            "mode": mode,
            "dataset_file": str(mode_file),
            **summary,
            "errors": errors,
        }
        with (per_mode_dir / f"{mode}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(mode_summary, f, indent=2, ensure_ascii=False)

        comparison_rows.append(
            {
                "mode": mode,
                "cases_requested": summary["cases_requested"],
                "cases_scored": summary["cases_scored"],
                "cases_failed": summary["cases_failed"],
                "topic_adherence": summary["avg_topic_adherence"],
                "agent_goal_accuracy": summary["avg_agent_goal_accuracy"],
                "conversation_coherence": summary["avg_conversation_coherence"],
                "medical_correctness": summary["avg_medical_correctness"],
                "medical_safety_caution": summary["avg_medical_safety_caution"],
            }
        )
        comparison_json["modes"].append(mode_summary)

    comparison_csv = result_dir / "comparison.csv"
    with comparison_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "cases_requested",
                "cases_scored",
                "cases_failed",
                "topic_adherence",
                "agent_goal_accuracy",
                "conversation_coherence",
                "medical_correctness",
                "medical_safety_caution",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    comparison_json_path = result_dir / "comparison.json"
    with comparison_json_path.open("w", encoding="utf-8") as f:
        json.dump(comparison_json, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Dataset dir  : {dataset_dir}")
    print(f"Results dir  : {result_dir}")
    print(f"Comparison   : {comparison_csv}")


if __name__ == "__main__":
    main()
