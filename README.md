# HIAST NLP: intent routing

Minimal W01–W02 teacher example: route a request to one of CLINC150’s 150 intents or reject it as `out_of_scope`.

```sh
uv sync --locked
uv run python main.py
uv run python main.py "How much money is in my account?"
uv run python -m unittest discover -s tests -v
```

Python 3.11+. The first run downloads the pinned dataset. Each run fits the baseline, chooses a threshold on validation, saves `outputs/results.json`, and routes one example. No GPU or NLTK corpus downloads are needed.

- `main.py`: settings and the complete experiment.
- `model.py`: regex tokens, optional stemming, n-grams, and nearest-example Jaccard matching.
- `dataset.py`: verified download and training/validation splits.
- `evaluation.py`: metrics and threshold selection.

Change `STEMMING` or `NGRAM_MAX` in `main.py` for a controlled comparison; save results before rerunning. Final-test data is never used. Similarity is not a probability; validation scores are also used for threshold selection.

[S03 walkthrough](reports/s03-walkthrough.md) · [Baseline report](reports/baseline.md) · [Data source](data/README.md)

S03 requires W01–W02; W03–W04 is optional bonus work. Five milestones each span two actual weeks. Private instructor material for now.
