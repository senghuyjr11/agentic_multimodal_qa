import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai


ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"


def load_environment() -> None:
    load_dotenv(ENV_PATH)
    required_keys = ["GOOGLE_API_KEY"]
    missing = [key for key in required_keys if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {missing}")


def list_gemini_models(client: genai.Client) -> list[str]:
    names = []
    for model in client.models.list():
        name = getattr(model, "name", "")
        if "gemini" in name.lower():
            names.append(name)
    return sorted(names)


def is_useful_judge_model(model_name: str) -> bool:
    lowered = model_name.lower()

    disallowed_terms = [
        "embedding",
        "image",
        "tts",
        "audio",
        "live",
        "robotics",
        "computer-use",
        "native-audio",
        "preview",
        "thinking",
    ]

    if any(term in lowered for term in disallowed_terms):
        return False

    allowed_terms = [
        "flash",
        "pro",
    ]

    return any(term in lowered for term in allowed_terms)


def extract_error_payload(exc: Exception) -> dict:
    text = str(exc)
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def parse_retry_delay(payload: dict) -> str | None:
    details = payload.get("error", {}).get("details", [])
    for detail in details:
        retry_delay = detail.get("retryDelay")
        if retry_delay:
            return retry_delay
    return None


def parse_quota_value(payload: dict) -> str | None:
    details = payload.get("error", {}).get("details", [])
    for detail in details:
        violations = detail.get("violations", [])
        for violation in violations:
            quota_value = violation.get("quotaValue")
            if quota_value:
                return quota_value
    return None


def probe_model(client: genai.Client, model_name: str) -> dict:
    try:
        client.models.generate_content(
            model=model_name,
            contents="Reply with exactly: OK",
        )
        return {
            "model": model_name,
            "status": "available",
            "retry_after": None,
            "quota_limit": None,
            "message": None,
        }
    except Exception as exc:
        payload = extract_error_payload(exc)
        message = payload.get("error", {}).get("message", str(exc))
        retry_after = parse_retry_delay(payload)
        quota_limit = parse_quota_value(payload)

        status = "error"
        if "RESOURCE_EXHAUSTED" in str(exc) or payload.get("error", {}).get("status") == "RESOURCE_EXHAUSTED":
            status = "rate_limited"

        return {
            "model": model_name,
            "status": status,
            "retry_after": retry_after,
            "quota_limit": quota_limit,
            "message": message,
        }


def main() -> None:
    load_environment()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    all_model_names = list_gemini_models(client)
    model_names = [name for name in all_model_names if is_useful_judge_model(name)]

    print(f"Loaded .env from: {ENV_PATH}")
    print(f"Total Gemini models: {len(all_model_names)}")
    print(f"Judge-capable models checked: {len(model_names)}")

    available = []
    rate_limited = []
    errored = []

    for model_name in model_names:
        result = probe_model(client, model_name)
        if result["status"] == "available":
            available.append(result)
        elif result["status"] == "rate_limited":
            rate_limited.append(result)
        else:
            errored.append(result)

    print("\nAVAILABLE")
    if available:
        for result in available:
            print(f"- {result['model']}")
    else:
        print("- None")

    print("\nRATE LIMITED")
    if rate_limited:
        for result in rate_limited:
            line = f"- {result['model']}"
            if result["retry_after"]:
                line += f" | retry_after={result['retry_after']}"
            if result["quota_limit"]:
                line += f" | quota_limit={result['quota_limit']}/min"
            print(line)
    else:
        print("- None")

    print("\nERROR")
    if errored:
        for result in errored:
            line = f"- {result['model']}"
            if result["message"]:
                short_message = result["message"].splitlines()[0]
                line += f" | {short_message}"
            print(line)
    else:
        print("- None")

    print("\nNote: Google does not expose an exact remaining-request counter in this view.")


if __name__ == "__main__":
    main()
