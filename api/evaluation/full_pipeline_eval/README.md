# Full Pipeline Ablation Evaluation

This folder evaluates **full pipeline ablation runs** mode-by-mode using RAGAS + Gemini judge.

## Folder Outputs

- `datasets/<run_id>/<mode>.jsonl`
- `results/<run_id>/per_mode/<mode>_results.jsonl`
- `results/<run_id>/per_mode/<mode>_summary.json`
- `results/<run_id>/comparison.csv`
- `results/<run_id>/comparison.json`

## Step 1: Build Mode Datasets From Ablation Runs

From `api/evaluation`:

```bash
python full_pipeline_eval/build_mode_datasets.py
```

Optional specific run id:

```bash
python full_pipeline_eval/build_mode_datasets.py --run-id 20260501_180311
```

## Step 2: Evaluate All Modes With RAGAS

From `api/evaluation`:

```bash
python full_pipeline_eval/evaluate_modes_ragas.py
```

This automatically reads the latest `datasets/<run_id>` and writes the comparison table under `results/<run_id>/comparison.csv`.
