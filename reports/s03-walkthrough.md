# S03 walkthrough

1. Define the supported-intent versus rejection decision.
2. Follow `main.py`: training/validation split → matching → threshold → evaluation.
3. Inspect `model.py`: regex tokens, stems, n-grams, and Jaccard overlap.
4. Run `uv run python main.py`; inspect metrics and errors in `outputs/results.json`.
5. Change only `NGRAM_MAX` or `STEMMING`, rerun, and explain the difference. Save the previous results first.

W01–W02 is required; W03–W04 is bonus. Keep final-test data untouched. Each two-week milestone reports a decision, evidence, and interpretation.
