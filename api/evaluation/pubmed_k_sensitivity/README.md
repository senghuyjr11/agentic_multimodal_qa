# PubMed K-Sensitivity Evaluation

This folder evaluates whether the default PubMed `top_k=5` setting is reasonable.

The normal app behavior is unchanged because `PUBMED_TOP_K` defaults to `5`.

## Run the sensitivity experiment

From `api/evaluation/pubmed_k_sensitivity`:

```powershell
python run_k_sensitivity.py
```

Optional custom port:

```powershell
python run_k_sensitivity.py --port 8010
```

The script starts the API four times:

- `PUBMED_TOP_K=1`
- `PUBMED_TOP_K=3`
- `PUBMED_TOP_K=5`
- `PUBMED_TOP_K=10`

Raw outputs are saved to:

```text
api/evaluation/pubmed_k_sensitivity/runs/<timestamp>/k_<K>/run_results.json
```

## Evaluate with RAGAS

After the run finishes:

```powershell
python evaluate_k_sensitivity.py
```

The comparison table is saved to:

```text
api/evaluation/pubmed_k_sensitivity/results/<timestamp>/comparison.csv
```
