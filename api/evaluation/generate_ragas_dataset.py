import json
from pathlib import Path
from typing import Iterable


PIPELINE_DIR = Path(__file__).resolve().parent
API_DIR = PIPELINE_DIR.parent
SESSIONS_DIR = API_DIR / "sessions"
OUTPUT_PATH = PIPELINE_DIR / "ragas_dataset_senghuy_non_medical.jsonl"

# Simple manual config
DEFAULT_USERNAME = "senghuy"
SESSION_IDS = [7]
TURN_FILTER_MODE = "medical_only"  # "medical_only" | "all"

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


def should_include_turn(turn: dict, mode: str = "medical_only") -> bool:
    meta = turn.get("meta", {}) or {}
    decision = meta.get("decision", {}) or {}
    response_mode = decision.get("response_mode")
    num_articles = meta.get("num_articles", 0) or 0
    assistant_text = (turn.get("assistant") or "").strip().lower()

    if mode == "all":
        return True

    # Keep medically meaningful turns for conversation-level medical evaluation.
    # Exclude non-medical/casual turns and invalid-image guardrail messages.
    if response_mode != "medical_answer":
        return False

    invalid_image_markers = [
        "does not appear to be a medical scan",
        "please upload a valid medical image",
        "model can only analyze medical images",
    ]
    if any(marker in assistant_text for marker in invalid_image_markers):
        return False

    # Keep turns that are grounded in retrieval or an actual medical explanation.
    if num_articles > 0:
        return True

    image_response_indicators = [
        "image shows",
        "this image",
        "consistent with",
        "diagnosis",
        "findings",
    ]
    return any(marker in assistant_text for marker in image_response_indicators)


def build_messages(history: list[dict], mode: str = "medical_only") -> list[dict]:
    messages = []

    selected_turns = [turn for turn in history if should_include_turn(turn, mode=mode)]
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
    messages = build_messages(history, mode=TURN_FILTER_MODE)

    if len(messages) < 2:
        return None

    return {
        "case_id": f"{session_data['username']}_s{session_data['session_id']}",
        "user_input": messages,
        "reference_topics": REFERENCE_TOPICS,
        "source_session_id": session_data["session_id"],
        "num_messages": len(messages),
        "turn_filter_mode": TURN_FILTER_MODE,
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
    print(f"Turn filter mode: {TURN_FILTER_MODE}")
    if exported == 0:
        print(
            "No cases exported. If your session is mostly casual chat, set "
            "TURN_FILTER_MODE = 'all' to include all turns."
        )
    print(f"Sessions directory: {SESSIONS_DIR.resolve()}")


if __name__ == "__main__":
    main()
