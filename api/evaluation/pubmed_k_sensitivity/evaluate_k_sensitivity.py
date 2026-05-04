"""
Evaluate PubMed top-k sensitivity runs with RAGAS + Gemini judge.

Expected input:
  api/evaluation/pubmed_k_sensitivity/runs/<timestamp>/k_*/run_results.json

Outputs:
  api/evaluation/pubmed_k_sensitivity/results/<timestamp>/
    - per_k/k_<K>_results.jsonl
    - per_k/k_<K>_summary.json
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
from ragas.messages import AIMessage, HumanMessage
from ragas.metrics.collections import AgentGoalAccuracyWithoutReference, TopicAdherence

try:
    from ragas.metrics.collections import AspectCritic  # type: ignore
except ImportError:
    from ragas.metrics import AspectCritic  # type: ignore


EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent.parent.parent
ENV_PATH = ROOT_DIR / ".env"
RUNS_DIR = EVAL_DIR / "runs"
RESULTS_DIR = EVAL_DIR / "results"

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


def resolve_latest_run_dir() -> Path:
    runs = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No PubMed K-sensitivity runs found in {RUNS_DIR}")
    return runs[-1]


def build_sample(case: dict[str, Any]) -> MultiTurnSample:
    messages = []
    for turn in case.get("turn_logs", []):
        question = (turn.get("question") or "").strip()
        answer = (turn.get("response") or "").strip()
        if question:
            messages.append(HumanMessage(content=question))
        if answer:
            messages.append(AIMessage(content=answer))

    return MultiTurnSample(
        user_input=messages,
        reference_topics=[
            "medical treatment",
            "medical explanation",
            "biomedical evidence",
            "clinical safety",
            "patient safety",
        ],
    )


def evaluate_run(run_file: Path, llm) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    topic_adherence = TopicAdherence(llm=llm, mode="f1")
    agent_goal_accuracy = AgentGoalAccuracyWithoutReference(llm=llm)
    medical_correctness = AspectCritic(
        name="medical_correctness",
        definition="Across the conversation, are the medical statements factually correct and non-contradictory?",
        llm=llm,
    )
    evidence_grounding = AspectCritic(
        name="evidence_grounding",
        definition="Does the answer use the provided biomedical evidence appropriately without unsupported overclaiming?",
        llm=llm,
    )
    medical_safety_caution = AspectCritic(
        name="medical_safety_caution",
        definition="Does the assistant remain appropriately cautious and safe for clinical information?",
        llm=llm,
    )

    payload = json.loads(run_file.read_text(encoding="utf-8"))
    top_k = payload.get("pubmed_top_k")
    results = []
    errors = []

    for case in payload.get("cases", []):
        row = {
            "case_id": case.get("name", "?"),
            "pubmed_top_k": top_k,
            "topic_adherence": None,
            "agent_goal_accuracy": None,
            "medical_correctness": None,
            "evidence_grounding": None,
            "medical_safety_caution": None,
            "avg_articles_per_case": None,
            "error": None,
        }
        article_counts = [
            int(((turn.get("raw") or {}).get("metadata", {}) or {}).get("num_articles") or 0)
            for turn in case.get("turn_logs", [])
        ]
        if article_counts:
            row["avg_articles_per_case"] = round(sum(article_counts) / len(article_counts), 4)

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
            row["medical_correctness"] = score_value(
                medical_correctness.multi_turn_score(sample)
            )
            row["evidence_grounding"] = score_value(
                evidence_grounding.multi_turn_score(sample)
            )
            row["medical_safety_caution"] = score_value(
                medical_safety_caution.multi_turn_score(sample)
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            row["error"] = err
            errors.append({"case_id": row["case_id"], "error": err, "traceback": traceback.format_exc()})

        results.append(row)

    summary = {
        "pubmed_top_k": top_k,
        "cases_requested": len(payload.get("cases", [])),
        "cases_scored": len([r for r in results if r["error"] is None]),
        "cases_failed": len(errors),
        "avg_topic_adherence": average([r["topic_adherence"] for r in results if r["topic_adherence"] is not None]),
        "avg_agent_goal_accuracy": average([r["agent_goal_accuracy"] for r in results if r["agent_goal_accuracy"] is not None]),
        "avg_medical_correctness": average([r["medical_correctness"] for r in results if r["medical_correctness"] is not None]),
        "avg_evidence_grounding": average([r["evidence_grounding"] for r in results if r["evidence_grounding"] is not None]),
        "avg_medical_safety_caution": average([r["medical_safety_caution"] for r in results if r["medical_safety_caution"] is not None]),
        "avg_articles_per_case": average([r["avg_articles_per_case"] for r in results if r["avg_articles_per_case"] is not None]),
    }
    return results, summary, errors


def main() -> None:
    load_dotenv(ENV_PATH)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(f"Missing GOOGLE_API_KEY in {ENV_PATH}")

    run_dir = resolve_latest_run_dir()
    run_id = run_dir.name
    result_dir = RESULTS_DIR / run_id
    per_k_dir = result_dir / "per_k"
    per_k_dir.mkdir(parents=True, exist_ok=True)

    llm_client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_OPENAI_BASE_URL)
    llm = llm_factory(
        MODEL_NAME,
        provider="openai",
        client=llm_client,
        max_tokens=8192,
    )

    run_files = sorted(run_dir.glob("k_*/run_results.json"))
    if not run_files:
        raise FileNotFoundError(f"No run_results.json files found under {run_dir}")

    comparison_rows = []
    comparison_json = {"run_id": run_id, "model": MODEL_NAME, "k_values": []}

    for run_file in run_files:
        mode_name = run_file.parent.name
        print(f"Evaluating {mode_name}: {run_file}")
        results, summary, errors = evaluate_run(run_file, llm)

        with (per_k_dir / f"{mode_name}_results.jsonl").open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        mode_summary = {
            "mode": mode_name,
            "run_file": str(run_file),
            **summary,
            "errors": errors,
        }
        with (per_k_dir / f"{mode_name}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(mode_summary, f, indent=2, ensure_ascii=False)

        comparison_rows.append(
            {
                "k": summary["pubmed_top_k"],
                "cases_requested": summary["cases_requested"],
                "cases_scored": summary["cases_scored"],
                "cases_failed": summary["cases_failed"],
                "topic_adherence": summary["avg_topic_adherence"],
                "agent_goal_accuracy": summary["avg_agent_goal_accuracy"],
                "medical_correctness": summary["avg_medical_correctness"],
                "evidence_grounding": summary["avg_evidence_grounding"],
                "medical_safety_caution": summary["avg_medical_safety_caution"],
                "avg_articles_per_case": summary["avg_articles_per_case"],
            }
        )
        comparison_json["k_values"].append(mode_summary)

    comparison_rows.sort(key=lambda row: row["k"])
    comparison_csv = result_dir / "comparison.csv"
    with comparison_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "k",
                "cases_requested",
                "cases_scored",
                "cases_failed",
                "topic_adherence",
                "agent_goal_accuracy",
                "medical_correctness",
                "evidence_grounding",
                "medical_safety_caution",
                "avg_articles_per_case",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    comparison_json_path = result_dir / "comparison.json"
    with comparison_json_path.open("w", encoding="utf-8") as f:
        json.dump(comparison_json, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Run dir    : {run_dir}")
    print(f"Result dir : {result_dir}")
    print(f"Comparison : {comparison_csv}")


if __name__ == "__main__":
    main()
