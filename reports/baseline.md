# W01–W02 baseline

**Decision:** match each request to a training example using Jaccard overlap of stemmed unigrams and bigrams. Reject low-overlap requests.

**Reproduce:** `uv run python main.py`. Use all 150 intents and the original training/validation splits. Choose the threshold by the mean of in-scope macro-F1 and OOS F1; ties prefer the lower threshold.

| System | Threshold | In-scope macro-F1 | OOS F1 | OOS AUROC |
| --- | ---: | ---: | ---: | ---: |
| Majority | — | 0.0001 | 0.0000 | 0.5000 |
| Lexical baseline | 0.20 | 0.7687 | 0.3062 | 0.8407 |

**Interpretation:** the baseline accepts 68% of OOS requests. Errors include translation requests routed to spelling and transfers routed to routing-number queries. Inspect `outputs/results.json` before changing the pipeline.

**Limits:** these scores use the same validation set as threshold selection. Three normalized training/validation overlaps were retained. No paraphrase audit or final-test evaluation has been performed.

**Next experiment:** set `NGRAM_MAX = 1` in `main.py`, rerun, and compare. This snapshot is not a completed two-week milestone.
