# Data

Source: [CLINC150 / oos-eval](https://github.com/clinc/oos-eval), accompanying Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction* (EMNLP 2019). Dataset terms: [CC BY 3.0](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE).

- File: `data/data_full.json` at revision `828f8093932c8fe6ca7936c3d2e52903b1c523de`.
- SHA-256: `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`.
- Run `uv run python main.py`; the unchanged file is stored under ignored `data/raw/`.
- Fit only on `train` (15,000 utterances, 150 intents). Tune on `val` + `oos_val` (3,000 + 100).
- `oos_train` is unused in this baseline. The loader maps the source label `oos` to `out_of_scope` for validation.
- Reserve `test` and `oos_test` for final evaluation. The downloaded JSON contains them, but experiments never select them.

Runs count exact overlaps after lowercasing and whitespace normalization between training and validation. This is a basic audit, not a paraphrase-leakage check. Preserve the official splits and report any overlaps rather than silently changing them.
