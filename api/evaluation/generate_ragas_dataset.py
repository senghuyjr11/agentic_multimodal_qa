import json
from pathlib import Path
from typing import Iterable


PIPELINE_DIR = Path(__file__).resolve().parent
API_DIR = PIPELINE_DIR.parent.parent
SESSIONS_DIR = API_DIR / "sessions"
OUTPUT_PATH = PIPELINE_DIR / "ragas_dataset_senghuy.jsonl"

# Simple manual config
DEFAULT_USERNAME = "senghuy"
SESSION_IDS = [1]

# Multi-turn evaluation config
MAX_TURNS_PER_SESSION: int | None = None
REFERENCE_TOPICS = [
    "medical diagnosis",
    "medical treatment",
    "medical explanation",
    "medical imaging",
    "infectious disease",
    "pathology",
]


def iter_session_files(
    sessions_dir: Path,
    username: str,
    session_ids: list[int] | None,
) -> Iterable[Path]:
    user_dir = sessions_dir / username
    if not user_dir.exists():
        return []

    if session_ids:
        candidates = [
            user_dir / str(session_id) / "session_data.json"
            for session_id in session_ids
        ]
        return [path for path in candidates if path.exists()]

    return sorted(user_dir.glob("*/session_data.json"), key=lambda p: int(p.parent.name))


def should_include_turn(turn: dict) -> bool:
    meta = turn.get("meta", {}) or {}
    decision = meta.get("decision", {}) or {}
    response_mode = decision.get("response_mode")

    # For session-level memory evaluation, keep substantive turns and skip only
    # trivial greetings/thanks that add little signal.
    if response_mode == "medical_answer":
        return True

    user_text = (turn.get("user") or "").strip().lower()
    assistant_text = (turn.get("assistant") or "").strip().lower()

    trivial_patterns = [
        "hi",
        "hello",
        "thank you",
        "thanks",
        "good memory",
    ]

    if user_text in trivial_patterns:
        return False

    if assistant_text.startswith("you're welcome") or assistant_text.startswith("you're most welcome"):
        return False

    return True


def build_messages(history: list[dict]) -> list[dict]:
    messages = []

    selected_turns = [turn for turn in history if should_include_turn(turn)]
    if MAX_TURNS_PER_SESSION is not None:
        selected_turns = selected_turns[:MAX_TURNS_PER_SESSION]

    for turn in selected_turns:
        user_message = (turn.get("user") or "").strip()
        assistant_message = (turn.get("assistant") or "").strip()

        if user_message:
            messages.append(
                {
                    "type": "human",
                    "content": user_message,
                }
            )

        if assistant_message:
            messages.append(
                {
                    "type": "ai",
                    "content": assistant_message,
                }
            )

    return messages


def make_case(session_data: dict) -> dict | None:
    history = session_data.get("conversation_history", []) or []
    messages = build_messages(history)

    if len(messages) < 2:
        return None

    return {
        "case_id": f"{session_data['username']}_s{session_data['session_id']}",
        "user_input": messages,
        "reference_topics": REFERENCE_TOPICS,
        "source_session_id": session_data["session_id"],
        "num_messages": len(messages),
    }


def main() -> None:
    output_path = OUTPUT_PATH.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session_files = list(iter_session_files(SESSIONS_DIR.resolve(), DEFAULT_USERNAME, SESSION_IDS))
    exported = 0

    with output_path.open("w", encoding="utf-8") as output:
        for session_file in session_files:
            with session_file.open("r", encoding="utf-8") as source:
                session_data = json.load(source)

            case = make_case(session_data)
            if case is None:
                continue

            output.write(json.dumps(case, ensure_ascii=False) + "\n")
            exported += 1

    print(
        f"Exported {exported} multi-turn Ragas cases from {len(session_files)} session files "
        f"for user '{DEFAULT_USERNAME}'"
        f"{f' with sessions {SESSION_IDS}' if SESSION_IDS else ''} "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
