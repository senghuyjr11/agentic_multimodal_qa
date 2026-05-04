"""
Run end-to-end ablation modes automatically and save outputs per mode.

What this script does:
1) Starts API server with ABLATION_MODE for each mode
2) Registers/logins a test user
3) Replays a fixed multi-turn prompt suite via /chat endpoints
4) Saves full responses + metadata under evaluation/ablation_runs/<timestamp>/<mode>/

Usage:
  python run_ablation_modes.py
  python run_ablation_modes.py --port 8010 --modes full no_pubmed
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
API_DIR = EVAL_DIR.parent

DEFAULT_MODES = ["full", "no_memory", "no_pubmed", "no_router", "no_translation"]
DEFAULT_SUITE = [
    # Memory/coreference cases: these should expose degradation in no_memory.
    {
        "name": "memory_patient_condition",
        "turns": [
            "A patient has pneumonia.",
            "What complications can it cause?",
            "Explain that condition more simply.",
        ],
    },
    {
        "name": "memory_named_disease",
        "turns": [
            "I am asking about Streptococcus pneumoniae.",
            "Is it usually mild or serious?",
            "What treatment is commonly used for it?",
        ],
    },
    {
        "name": "memory_symptom_context",
        "turns": [
            "The patient has fever, cough, and chest pain.",
            "What diagnosis could these symptoms suggest?",
            "Which warning signs would make it urgent?",
        ],
    },
    {
        "name": "memory_followup_reference",
        "turns": [
            "What is bacterial meningitis?",
            "Can it become life-threatening?",
            "What did you say was dangerous about it?",
        ],
    },
    {
        "name": "memory_pronoun_resolution",
        "turns": [
            "Tell me about tuberculosis.",
            "How is it transmitted?",
            "Can it be prevented?",
        ],
    },
    {
        "name": "memory_user_name",
        "turns": [
            "My name is Alex and I am asking about diabetes.",
            "What is diabetes?",
            "What is my name?",
        ],
    },
    {
        "name": "memory_medication_context",
        "turns": [
            "The patient is taking antibiotics for pneumonia.",
            "Why are antibiotics used?",
            "When should the patient seek urgent care?",
        ],
    },
    {
        "name": "memory_summary_request",
        "turns": [
            "Explain chronic obstructive pulmonary disease.",
            "What symptoms are common?",
            "Summarize everything so far in three bullets.",
        ],
    },
    {
        "name": "memory_comparison_followup",
        "turns": [
            "What is asthma?",
            "How is it different from pneumonia?",
            "Which one is usually infectious?",
        ],
    },
    {
        "name": "memory_risk_factor_followup",
        "turns": [
            "Tell me about stroke.",
            "What are common risk factors?",
            "Which of those can be modified?",
        ],
    },

    # Retrieval/evidence-needed cases: these should expose degradation in no_pubmed.
    {
        "name": "retrieval_guideline_antibiotic",
        "turns": [
            "What is the usual treatment for Streptococcus pneumoniae infection?",
            "Give evidence-based reasoning with references.",
        ],
    },
    {
        "name": "retrieval_vaccine_evidence",
        "turns": [
            "How does pneumococcal vaccination reduce disease risk?",
            "Support the answer with biomedical evidence.",
        ],
    },
    {
        "name": "retrieval_covid_pneumonia",
        "turns": [
            "Can COVID-19 cause pneumonia?",
            "What evidence supports that?",
        ],
    },
    {
        "name": "retrieval_tb_treatment",
        "turns": [
            "How is tuberculosis usually treated?",
            "Use references and mention why treatment takes time.",
        ],
    },
    {
        "name": "retrieval_diabetes_complications",
        "turns": [
            "What complications can diabetes cause?",
            "Give evidence-grounded explanation.",
        ],
    },
    {
        "name": "retrieval_meningitis_urgency",
        "turns": [
            "Why is bacterial meningitis considered urgent?",
            "Cite evidence about severity or outcomes.",
        ],
    },
    {
        "name": "retrieval_antibiotic_resistance",
        "turns": [
            "Why is antibiotic resistance a concern in bacterial infections?",
            "Support with medical literature.",
        ],
    },
    {
        "name": "retrieval_pneumonia_recovery",
        "turns": [
            "How long does pneumonia recovery usually take?",
            "Use evidence and explain uncertainty.",
        ],
    },
    {
        "name": "retrieval_asthma_inhaled_steroids",
        "turns": [
            "Why are inhaled corticosteroids used in asthma?",
            "Give a referenced explanation.",
        ],
    },
    {
        "name": "retrieval_stroke_time_window",
        "turns": [
            "Why does stroke treatment depend on time?",
            "Use evidence-grounded reasoning.",
        ],
    },

    # Router edge cases: these should expose degradation in no_router.
    {
        "name": "router_casual_to_medical",
        "turns": [
            "Hi there.",
            "Now explain pneumonia warning signs.",
        ],
    },
    {
        "name": "router_nonmedical_chat",
        "turns": [
            "I just need someone to talk to.",
            "Can you keep the response non-medical?",
        ],
    },
    {
        "name": "router_modify_previous",
        "turns": [
            "What is diabetes?",
            "Make your previous answer shorter.",
        ],
    },
    {
        "name": "router_reference_request",
        "turns": [
            "What is tuberculosis?",
            "Show the references from the previous answer.",
        ],
    },
    {
        "name": "router_topic_boundary",
        "turns": [
            "Tell me a joke.",
            "Actually, what are symptoms of meningitis?",
        ],
    },

    # Multilingual cases: these should expose degradation in no_translation.
    {
        "name": "multilingual_spanish",
        "turns": [
            "Que es la neumonia?",
            "Es peligrosa?",
        ],
    },
    {
        "name": "multilingual_french",
        "turns": [
            "Qu'est-ce que le diabete?",
            "Quels sont les signes d'urgence?",
        ],
    },
    {
        "name": "multilingual_korean",
        "turns": [
            "폐렴은 무엇인가요?",
            "언제 병원에 가야 하나요?",
        ],
    },
    {
        "name": "multilingual_japanese",
        "turns": [
            "結核とは何ですか?",
            "どのように治療しますか?",
        ],
    },
    {
        "name": "multilingual_chinese",
        "turns": [
            "糖尿病是什么?",
            "它会导致哪些并发症?",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--modes", nargs="*", default=DEFAULT_MODES)
    p.add_argument("--startup-timeout", type=int, default=900)
    return p.parse_args()


def wait_for_health(base_url: str, timeout_sec: int, proc: subprocess.Popen | None = None) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"Server process exited early with code {proc.returncode} before health check passed."
            )
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"Server did not become healthy within {timeout_sec}s: {base_url}")


def start_server(mode: str, host: str, port: int, log_path: Path) -> tuple[subprocess.Popen, Any]:
    env = os.environ.copy()
    env["ABLATION_MODE"] = mode
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

    r = requests.post(
        f"{base_url}/auth/login",
        json={"username": username, "password": password},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Login succeeded but no access_token returned.")
    return token


def run_suite(base_url: str, token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    out: dict[str, Any] = {"cases": []}

    for case in DEFAULT_SUITE:
        turns = case["turns"]
        first = turns[0]

        r = requests.post(
            f"{base_url}/chat/new",
            headers=headers,
            data={"question": first},
            timeout=120,
        )
        r.raise_for_status()
        first_payload = r.json()
        session_id = first_payload["session_id"]
        logs = [{"turn": 1, "question": first, "response": first_payload.get("response"), "raw": first_payload}]

        for idx, question in enumerate(turns[1:], start=2):
            rr = requests.post(
                f"{base_url}/chat/{session_id}/message",
                headers=headers,
                data={"question": question},
                timeout=120,
            )
            rr.raise_for_status()
            payload = rr.json()
            logs.append({"turn": idx, "question": question, "response": payload.get("response"), "raw": payload})

        out["cases"].append(
            {
                "name": case["name"],
                "session_id": session_id,
                "turn_count": len(turns),
                "turn_logs": logs,
            }
        )

    return out


def summarize(run_data: dict[str, Any]) -> dict[str, Any]:
    total_turns = 0
    pubmed_attempted = 0
    articles_total = 0

    for case in run_data.get("cases", []):
        for t in case.get("turn_logs", []):
            total_turns += 1
            meta = (t.get("raw") or {}).get("metadata", {}) or {}
            if meta.get("pubmed_attempted"):
                pubmed_attempted += 1
            articles_total += int(meta.get("num_articles") or 0)

    return {
        "total_cases": len(run_data.get("cases", [])),
        "total_turns": total_turns,
        "pubmed_attempted_turns": pubmed_attempted,
        "total_articles_returned": articles_total,
    }


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = EVAL_DIR / "ablation_runs" / ts
    root.mkdir(parents=True, exist_ok=True)

    base_url = f"http://{args.host}:{args.port}"
    username = f"ablation_{ts}"
    password = f"eval_{secrets.token_urlsafe(18)}"

    overall = {"timestamp": ts, "base_url": base_url, "modes": []}

    for mode in args.modes:
        mode_dir = root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Running mode: {mode} ===")

        server_log_path = mode_dir / "server.log"
        proc, log_fp = start_server(mode=mode, host=args.host, port=args.port, log_path=server_log_path)
        try:
            wait_for_health(base_url, args.startup_timeout, proc=proc)
            token = register_and_login(base_url, username=username, password=password)
            run_data = run_suite(base_url, token)
            run_data["mode"] = mode
            run_data["summary"] = summarize(run_data)

            with (mode_dir / "run_results.json").open("w", encoding="utf-8") as f:
                json.dump(run_data, f, indent=2, ensure_ascii=False)

            overall["modes"].append(
                {"mode": mode, "summary": run_data["summary"], "result_file": str(mode_dir / "run_results.json")}
            )
            print(f"Saved: {mode_dir / 'run_results.json'}")
        finally:
            stop_server(proc, log_fp=log_fp)

    with (root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\n=== Ablation runs complete ===")
    print(f"Output root: {root}")
    print(f"Summary: {root / 'summary.json'}")


if __name__ == "__main__":
    main()
