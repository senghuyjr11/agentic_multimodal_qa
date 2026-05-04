"""
Build per-mode multi-turn RAGAS datasets from ablation run outputs.

Input:
  api/evaluation/ablation_runs/<timestamp>/<mode>/run_results.json

Output:
  api/evaluation/full_pipeline_eval/datasets/<timestamp>/<mode>.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
ABLATION_RUNS_DIR = EVAL_DIR.parent / "ablation_runs"
DATASETS_DIR = EVAL_DIR / "datasets"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-id",
        type=str,
        default="latest",
        help="Ablation run folder name under evaluation/ablation_runs (default: latest).",
    )
    return p.parse_args()


def resolve_run_dir(run_id: str) -> Path:
    if run_id != "latest":
        path = ABLATION_RUNS_DIR / run_id
        if not path.exists():
            raise FileNotFoundError(f"Run id not found: {path}")
        return path

    all_runs = sorted([p for p in ABLATION_RUNS_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not all_runs:
        raise FileNotFoundError(f"No ablation runs found under {ABLATION_RUNS_DIR}")
    return all_runs[-1]


def to_case(mode: str, case: dict[str, Any], run_id: str) -> dict[str, Any]:
    messages = []
    for turn in case.get("turn_logs", []):
        q = (turn.get("question") or "").strip()
        a = (turn.get("response") or "").strip()
        if q:
            messages.append({"type": "human", "content": q})
        if a:
            messages.append({"type": "ai", "content": a})

    return {
        "case_id": f"{mode}_{run_id}_{case.get('name','case')}_{case.get('session_id','na')}",
        "mode": mode,
        "session_id": case.get("session_id"),
        "num_messages": len(messages),
        "user_input": messages,
        "reference_topics": [
            "medical diagnosis",
            "medical treatment",
            "medical explanation",
            "medical imaging",
            "patient safety",
        ],
    }


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_id)
    run_id = run_dir.name

    out_dir = DATASETS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir()])
    if not mode_dirs:
        raise RuntimeError(f"No mode directories found in {run_dir}")

    total_cases = 0
    for mode_dir in mode_dirs:
        mode = mode_dir.name
        run_file = mode_dir / "run_results.json"
        if not run_file.exists():
            continue

        payload = json.loads(run_file.read_text(encoding="utf-8"))
        cases = []
        for case in payload.get("cases", []):
            converted = to_case(mode=mode, case=case, run_id=run_id)
            if converted["num_messages"] >= 2:
                cases.append(converted)

        out_file = out_dir / f"{mode}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        total_cases += len(cases)
        print(f"[{mode}] wrote {len(cases)} cases -> {out_file}")

    print(f"\nDone. Total exported cases: {total_cases}")
    print(f"Datasets dir: {out_dir}")


if __name__ == "__main__":
    main()
