import json
from pathlib import Path
from typing import Iterable


EVAL_DIR = Path(__file__).resolve().parent
API_DIR = EVAL_DIR.parent
SESSIONS_DIR = API_DIR / "sessions"
DEFAULT_OUTPUT = EVAL_DIR / "eval_cases.jsonl"
DEFAULT_USERNAME = "kyojuro"


def iter_session_files(sessions_dir: Path, username: str | None = None) -> Iterable[Path]:
    if username:
        user_dir = sessions_dir / username
        if not user_dir.exists():
            return []
        return sorted(
            user_dir.glob("*/session_data.json"),
            key=lambda p: int(p.parent.name)
        )

    return sorted(
        sessions_dir.glob("*/*/session_data.json"),
        key=lambda p: (p.parent.parent.name, int(p.parent.name))
    )


def make_case(session_data: dict, turn: dict, turn_index: int) -> dict:
    meta = turn.get("meta", {}) or {}
    decision = meta.get("decision", {}) or {}

    return {
        "case_id": f"{session_data['username']}_s{session_data['session_id']}_t{turn.get('turn', turn_index + 1)}",
        "question": turn.get("user"),
        "expected_answer": turn.get("assistant"),
        "expected_needs_pubmed": decision.get("needs_pubmed"),
        "expected_has_references": bool(meta.get("articles")),
        "pubmed_attempted": meta.get("pubmed_attempted"),
    }


def main() -> None:
    sessions_dir = SESSIONS_DIR.resolve()
    output_path = DEFAULT_OUTPUT.resolve()
    username = DEFAULT_USERNAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session_files = list(iter_session_files(sessions_dir, username))
    exported = 0

    with output_path.open("w", encoding="utf-8") as f:
        for session_file in session_files:
            with session_file.open("r", encoding="utf-8") as session_handle:
                session_data = json.load(session_handle)

            for idx, turn in enumerate(session_data.get("conversation_history", [])):
                case = make_case(session_data, turn, idx)
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
                exported += 1

    print(
        f"Exported {exported} cases from {len(session_files)} session files "
        f"for user '{username}' to {output_path}"
    )


if __name__ == "__main__":
    main()
