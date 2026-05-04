"""
Run PubMed top-k sensitivity experiments.

This script starts the existing API four times with PUBMED_TOP_K set to
1, 3, 5, and 10. It uses only evidence-seeking text cases, so the experiment
isolates the effect of the number of retrieved PubMed articles.

Usage:
  python run_k_sensitivity.py
  python run_k_sensitivity.py --port 8010 --k-values 1 3 5 10
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


EVAL_DIR = Path(__file__).resolve().parent
API_DIR = EVAL_DIR.parent.parent
RUNS_DIR = EVAL_DIR / "runs"

DEFAULT_K_VALUES = [1, 3, 5, 10]
DEFAULT_SUITE = [
    {
        "name": "pneumococcal_treatment",
        "turns": [
            "What is the usual treatment for Streptococcus pneumoniae infection?",
            "Give evidence-based reasoning with references.",
        ],
    },
    {
        "name": "pneumococcal_vaccine",
        "turns": [
            "How does pneumococcal vaccination reduce disease risk?",
            "Support the answer with biomedical evidence.",
        ],
    },
    {
        "name": "covid_pneumonia",
        "turns": [
            "Can COVID-19 cause pneumonia?",
            "What evidence supports that?",
        ],
    },
    {
        "name": "tuberculosis_treatment",
        "turns": [
            "How is tuberculosis usually treated?",
            "Use references and mention why treatment takes time.",
        ],
    },
    {
        "name": "diabetes_complications",
        "turns": [
            "What complications can diabetes cause?",
            "Give evidence-grounded explanation.",
        ],
    },
    {
        "name": "meningitis_urgency",
        "turns": [
            "Why is bacterial meningitis considered urgent?",
            "Cite evidence about severity or outcomes.",
        ],
    },
    {
        "name": "antibiotic_resistance",
        "turns": [
            "Why is antibiotic resistance a concern in bacterial infections?",
            "Support with medical literature.",
        ],
    },
    {
        "name": "pneumonia_recovery",
        "turns": [
            "How long does pneumonia recovery usually take?",
            "Use evidence and explain uncertainty.",
        ],
    },
    {
        "name": "asthma_inhaled_steroids",
        "turns": [
            "Why are inhaled corticosteroids used in asthma?",
            "Give a referenced explanation.",
        ],
    },
    {
        "name": "stroke_time_window",
        "turns": [
            "Why does stroke treatment depend on time?",
            "Use evidence-grounded reasoning.",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--k-values", nargs="*", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--startup-timeout", type=int, default=900)
    return parser.parse_args()


def wait_for_health(base_url: str, timeout_sec: int, proc: subprocess.Popen | None = None) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"Server process exited early with code {proc.returncode} before health check passed."
            )
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"Server did not become healthy within {timeout_sec}s: {base_url}")


def start_server(top_k: int, host: str, port: int, log_path: Path) -> tuple[subprocess.Popen, Any]:
    env = os.environ.copy()
    env["ABLATION_MODE"] = "full"
    env["PUBMED_TOP_K"] = str(top_k)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api_refactored:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(API_DIR),
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, log_fp


def stop_server(proc: subprocess.Popen, log_fp: Any | None = None) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    if log_fp is not None:
        log_fp.flush()
        log_fp.close()


def register_and_login(base_url: str, username: str, password: str) -> str:
    try:
        requests.post(
            f"{base_url}/auth/register",
            json={"username": username, "password": password},
            timeout=20,
        )
    except requests.RequestException:
        pass

    response = requests.post(
        f"{base_url}/auth/login",
        json={"username": username, "password": password},
        timeout=20,
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("Login succeeded but no access_token returned.")
    return token


def run_suite(base_url: str, token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    output: dict[str, Any] = {"cases": []}

    for case in DEFAULT_SUITE:
        turns = case["turns"]
        first_question = turns[0]

        response = requests.post(
            f"{base_url}/chat/new",
            headers=headers,
            data={"question": first_question},
            timeout=180,
        )
        response.raise_for_status()
        first_payload = response.json()
        session_id = first_payload["session_id"]
        logs = [
            {
                "turn": 1,
                "question": first_question,
                "response": first_payload.get("response"),
                "raw": first_payload,
            }
        ]

        for turn_index, question in enumerate(turns[1:], start=2):
            followup = requests.post(
                f"{base_url}/chat/{session_id}/message",
                headers=headers,
                data={"question": question},
                timeout=180,
            )
            followup.raise_for_status()
            payload = followup.json()
            logs.append(
                {
                    "turn": turn_index,
                    "question": question,
                    "response": payload.get("response"),
                    "raw": payload,
                }
            )

        output["cases"].append(
            {
                "name": case["name"],
                "session_id": session_id,
                "turn_count": len(turns),
                "turn_logs": logs,
            }
        )

    return output


def summarize(run_data: dict[str, Any]) -> dict[str, Any]:
    total_turns = 0
    pubmed_attempted = 0
    article_counts: list[int] = []

    for case in run_data.get("cases", []):
        for turn in case.get("turn_logs", []):
            total_turns += 1
            metadata = (turn.get("raw") or {}).get("metadata", {}) or {}
            if metadata.get("pubmed_attempted"):
                pubmed_attempted += 1
            article_counts.append(int(metadata.get("num_articles") or 0))

    avg_articles = round(sum(article_counts) / len(article_counts), 4) if article_counts else 0.0
    return {
        "total_cases": len(run_data.get("cases", [])),
        "total_turns": total_turns,
        "pubmed_attempted_turns": pubmed_attempted,
        "total_articles_returned": sum(article_counts),
        "avg_articles_per_turn": avg_articles,
    }


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = RUNS_DIR / timestamp
    root.mkdir(parents=True, exist_ok=True)

    base_url = f"http://{args.host}:{args.port}"
    overall = {"timestamp": timestamp, "base_url": base_url, "k_values": []}

    for top_k in args.k_values:
        if top_k < 1:
            raise ValueError(f"Invalid K value: {top_k}. K must be >= 1.")

        mode_name = f"k_{top_k}"
        mode_dir = root / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Running PubMed top-k: {top_k} ===")

        server_log_path = mode_dir / "server.log"
        proc, log_fp = start_server(top_k=top_k, host=args.host, port=args.port, log_path=server_log_path)
        try:
            wait_for_health(base_url, args.startup_timeout, proc=proc)
            username = f"pubmed_k_{timestamp}_{top_k}"
            password = f"eval_{secrets.token_urlsafe(18)}"
            token = register_and_login(base_url, username=username, password=password)
            run_data = run_suite(base_url, token)
            run_data["pubmed_top_k"] = top_k
            run_data["summary"] = summarize(run_data)

            result_file = mode_dir / "run_results.json"
            with result_file.open("w", encoding="utf-8") as f:
                json.dump(run_data, f, indent=2, ensure_ascii=False)

            overall["k_values"].append(
                {"k": top_k, "summary": run_data["summary"], "result_file": str(result_file)}
            )
            print(f"Saved: {result_file}")
        finally:
            stop_server(proc, log_fp=log_fp)

    summary_file = root / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\n=== PubMed K-sensitivity runs complete ===")
    print(f"Output root: {root}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
