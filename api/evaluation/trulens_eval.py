import json
import csv
from pathlib import Path

from trulens.apps.app import TruApp, instrument
from trulens.core import Metric
from trulens.core import Selector
from trulens.core import TruSession


EVAL_DIR = Path(__file__).resolve().parent
CASES_PATH = EVAL_DIR / "eval_cases.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
DB_PATH = RESULTS_DIR / "trulens_eval.sqlite"
RESULTS_JSONL = RESULTS_DIR / "eval_results.jsonl"
RESULTS_CSV = RESULTS_DIR / "eval_results.csv"
SUMMARY_JSON = RESULTS_DIR / "eval_summary.json"


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def contains_references(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return "**references:**" in lowered or "\nreferences:" in lowered


def answer_presence(output: dict) -> float:
    answer = output.get("answer", "") if isinstance(output, dict) else ""
    return 1.0 if answer and answer.strip() else 0.0


def structure_score(output: dict) -> float:
    answer = output.get("answer", "") if isinstance(output, dict) else ""
    question = output.get("question", "") if isinstance(output, dict) else ""
    if not answer:
        return 0.0

    lowered_q = (question or "").lower()
    lowered_a = answer.lower()
    bullet_count = sum(1 for line in answer.splitlines() if line.strip().startswith("- "))

    if any(k in lowered_q for k in ["explain", "detail", "summarize", "break it down"]):
        headings = ["summary:", "key findings:", "plain explanation:", "why it matters:", "key evidence:"]
        heading_hits = sum(1 for h in headings if h in lowered_a)
        if heading_hits >= 2 and bullet_count >= 3:
            return 1.0
        if heading_hits >= 1 and bullet_count >= 2:
            return 0.7
        return 0.3

    if len(answer.split()) <= 120:
        return 1.0
    if len(answer.split()) <= 220:
        return 0.7
    return 0.4


def reference_behavior_score(output: dict) -> float:
    if not isinstance(output, dict):
        return 0.0

    answer = output.get("answer", "")
    actual_has_references = contains_references(answer)
    expected_has_references = bool(output.get("expected_has_references"))

    return 1.0 if actual_has_references == expected_has_references else 0.0


def pubmed_decision_score(output: dict) -> float:
    if not isinstance(output, dict):
        return 0.0

    expected = output.get("expected_needs_pubmed")
    pubmed_attempted = output.get("pubmed_attempted")

    if expected is None:
        return 0.0

    if pubmed_attempted is None:
        return 1.0 if expected is False else 0.0

    return 1.0 if bool(pubmed_attempted) == bool(expected) else 0.0


def average(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


class SessionReplayApp:
    @instrument
    def pipeline(self, case: dict) -> dict:
        return {
            "case_id": case.get("case_id"),
            "question": case.get("question"),
            "answer": case.get("expected_answer"),
            "expected_needs_pubmed": case.get("expected_needs_pubmed"),
            "expected_has_references": case.get("expected_has_references"),
            "pubmed_attempted": case.get("pubmed_attempted"),
        }


app = SessionReplayApp()

m_answer_presence = Metric(
    implementation=answer_presence,
    name="Answer Presence",
    selectors={"output": Selector.select_record_output()},
)

m_structure = Metric(
    implementation=structure_score,
    name="Structure Score",
    selectors={"output": Selector.select_record_output()},
)

m_reference_behavior = Metric(
    implementation=reference_behavior_score,
    name="Reference Behavior",
    selectors={"output": Selector.select_record_output()},
)

m_pubmed_decision = Metric(
    implementation=pubmed_decision_score,
    name="PubMed Decision Match",
    selectors={"output": Selector.select_record_output()},
)

metrics = [
    m_answer_presence,
    m_structure,
    m_reference_behavior,
    m_pubmed_decision,
]


def main() -> None:
    if not CASES_PATH.exists():
        raise FileNotFoundError(f"Missing eval cases file: {CASES_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    session = TruSession(database_url=f"sqlite:///{DB_PATH}")
    session.reset_database()

    tru_app = TruApp(
        app,
        app_name="MedicalVQAReplay",
        app_version="v1.0",
        feedbacks=metrics,
    )

    cases = load_cases(CASES_PATH)
    print(f"Running replay evaluation on {len(cases)} cases...")
    rows = []

    with tru_app as recording:
        for case in cases:
            print(f"  [{case['case_id']}] {str(case.get('question', ''))[:80]}")
            output = app.pipeline(case)
            rows.append({
                "case_id": output.get("case_id"),
                "question": output.get("question"),
                "answer_presence": answer_presence(output),
                "structure_score": structure_score(output),
                "reference_behavior_score": reference_behavior_score(output),
                "pubmed_decision_score": pubmed_decision_score(output),
                "pubmed_attempted": output.get("pubmed_attempted"),
                "has_references": output.get("expected_has_references"),
            })

    leaderboard = session.get_leaderboard()
    print("\nEvaluation complete.")
    print(leaderboard)

    with RESULTS_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "case_id",
        "question",
        "answer_presence",
        "structure_score",
        "reference_behavior_score",
        "pubmed_decision_score",
        "pubmed_attempted",
        "has_references",
    ]
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "cases_evaluated": len(rows),
        "avg_answer_presence": average([r["answer_presence"] for r in rows]),
        "avg_structure_score": average([r["structure_score"] for r in rows]),
        "avg_reference_behavior_score": average([r["reference_behavior_score"] for r in rows]),
        "avg_pubmed_decision_score": average([r["pubmed_decision_score"] for r in rows]),
        "results_jsonl": str(RESULTS_JSONL),
        "results_csv": str(RESULTS_CSV),
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nManual summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    leaderboard_path = RESULTS_DIR / "leaderboard.csv"
    try:
        leaderboard.to_csv(leaderboard_path, index=False)
        print(f"\nSaved leaderboard to {leaderboard_path}")
    except Exception as e:
        print(f"\nCould not save leaderboard CSV: {e}")


if __name__ == "__main__":
    main()
