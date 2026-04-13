import csv
import json
import os
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'python-dotenv'. Install it in your evaluation environment with "
        "'pip install python-dotenv'."
    ) from exc

try:
    from google import genai
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'google-genai'. Install it in your evaluation environment with "
        "'pip install google-genai'."
    ) from exc


EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent.parent
ENV_PATH = ROOT_DIR / ".env"
CASES_PATH = EVAL_DIR / "eval_cases.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_JSONL = RESULTS_DIR / "llm_judge_results.jsonl"
RESULTS_CSV = RESULTS_DIR / "llm_judge_results.csv"
SUMMARY_JSON = RESULTS_DIR / "llm_judge_summary.json"


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def load_runtime_config() -> tuple[str, float, int | None]:
    model_name = os.getenv("LLM_JUDGE_MODEL", "models/gemini-flash-latest")
    sleep_seconds = float(os.getenv("LLM_JUDGE_SLEEP_SECONDS", "8"))
    max_cases_env = os.getenv("LLM_JUDGE_MAX_CASES")
    max_cases = int(max_cases_env) if max_cases_env else None
    return model_name, sleep_seconds, max_cases


def build_prompt(case: dict) -> str:
    return f"""You are an evaluator for a medical QA assistant.

Score the answer using only the rubric below. Be fair and concise.

Return valid JSON only with this exact schema:
{{
  "answer_relevance_score": 1-5,
  "groundedness_score": 1-5,
  "context_relevance_score": 1-5,
  "coherence_score": 1-5,
  "overall_score": 1-5,
  "strength": "one short sentence",
  "weakness": "one short sentence"
}}

Scoring rubric:
- answer_relevance_score: Does the answer directly address the user's question and intent?
- groundedness_score: Does the answer stay supported by its own cited evidence or clearly stated information, without obvious unsupported leaps?
- context_relevance_score: If the answer uses references or supporting context, are they relevant to the user's question? If no context is used and none seems needed, score based on whether the answer still remains appropriately focused.
- coherence_score: Is the response logically organized, readable, and easy to follow?
- overall_score: Overall usefulness of the answer.

Case:
- question: {json.dumps(case.get("question", ""), ensure_ascii=False)}
- answer: {json.dumps(case.get("expected_answer", ""), ensure_ascii=False)}
- expected_needs_pubmed: {json.dumps(case.get("expected_needs_pubmed"))}
- expected_has_references: {json.dumps(case.get("expected_has_references"))}
- pubmed_attempted: {json.dumps(case.get("pubmed_attempted"))}
"""


def extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("Model response did not contain valid JSON.")


def clamp_score(value: object) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, min(5, score))


def judge_case(client: "genai.Client", case: dict, model_name: str) -> dict:
    prompt = build_prompt(case)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    payload = extract_json_object(getattr(response, "text", ""))

    return {
        "case_id": case.get("case_id"),
        "question": case.get("question"),
        "answer_relevance_score": clamp_score(payload.get("answer_relevance_score")),
        "groundedness_score": clamp_score(payload.get("groundedness_score")),
        "context_relevance_score": clamp_score(payload.get("context_relevance_score")),
        "coherence_score": clamp_score(payload.get("coherence_score")),
        "overall_score": clamp_score(payload.get("overall_score")),
        "strength": str(payload.get("strength", "")).strip(),
        "weakness": str(payload.get("weakness", "")).strip(),
    }


def main() -> None:
    load_dotenv(ENV_PATH)
    model_name, sleep_seconds, max_cases = load_runtime_config()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            f"Missing GOOGLE_API_KEY environment variable. Checked repo .env at {ENV_PATH}"
        )
    if not CASES_PATH.exists():
        raise FileNotFoundError(f"Missing eval cases file: {CASES_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=api_key)
    cases = load_cases(CASES_PATH)
    if max_cases is not None:
        cases = cases[:max_cases]

    print(f"Running LLM-as-judge evaluation on {len(cases)} cases with {model_name}...")
    print(f"Results JSONL: {RESULTS_JSONL}")
    print(f"Results CSV:   {RESULTS_CSV}")
    print(f"Summary JSON:  {SUMMARY_JSON}")
    rows = []
    errors = []

    for idx, case in enumerate(cases, start=1):
        print(f"  [{idx}/{len(cases)}] {case.get('case_id')} - {str(case.get('question', ''))[:70]}")
        try:
            row = judge_case(client, case, model_name)
            rows.append(row)
        except Exception as exc:
            errors.append({
                "case_id": case.get("case_id"),
                "error": str(exc),
            })
            print(f"    Skipped: {exc}")

        if idx < len(cases) and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    with RESULTS_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "case_id",
        "question",
        "answer_relevance_score",
        "groundedness_score",
        "context_relevance_score",
        "coherence_score",
        "overall_score",
        "strength",
        "weakness",
    ]
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": model_name,
        "cases_requested": len(cases),
        "cases_scored": len(rows),
        "cases_failed": len(errors),
        "avg_answer_relevance_score": average([r["answer_relevance_score"] for r in rows]),
        "avg_groundedness_score": average([r["groundedness_score"] for r in rows]),
        "avg_context_relevance_score": average([r["context_relevance_score"] for r in rows]),
        "avg_coherence_score": average([r["coherence_score"] for r in rows]),
        "avg_overall_score": average([r["overall_score"] for r in rows]),
        "results_jsonl": str(RESULTS_JSONL),
        "results_csv": str(RESULTS_CSV),
        "errors": errors,
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nLLM judge summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
