# Initial W01–W02 baseline

**Decision:** compare nearest-training-example Jaccard matching over regex tokens and optional Porter stems. Retain negation; compare unigram/bigram features against unigrams alone. No learned semantic representations are used.

**Reproduce:** `uv run intent-routing run`. See `configs/baseline.json` and the [pinned data](../data/README.md). All 150 intents are included. Each variant selects its threshold on validation by maximizing the mean of in-scope macro-F1 and OOS F1; ties prefer the lower threshold.

| Variant | Threshold | In-scope macro-F1 | OOS F1 | OOS AUROC |
| --- | ---: | ---: | ---: | ---: |
| Majority | — | 0.0001 | 0.0000 | 0.5000 |
| Stemmed unigrams + bigrams | 0.20 | 0.7687 | 0.3062 | 0.8407 |
| No stemming | 0.20 | 0.7575 | 0.2909 | 0.8316 |
| Stemmed unigrams | 0.30 | 0.8023 | 0.3143 | 0.8654 |
| Drop negation | 0.20 | 0.7642 | 0.3010 | 0.8413 |

**Interpretation:** unigrams lead this validation comparison. Bigrams did not help; stemming modestly helped. Even the best variant accepts 67% of OOS requests, so routing accuracy alone is misleading. Errors include translation requests routed to spelling and transfer requests routed to routing-number queries.

**Limits:** these are threshold-selection scores, not held-out estimates. Three normalized training/validation overlaps were found; official splits were retained. No paraphrase audit or final-test evaluation has been performed. The default prediction command demonstrates the configured unigram/bigram baseline, not an automatically selected winner.

**Next:** inspect rejection errors and threshold trade-offs before introducing later methods. This initial snapshot is not a completed two-week milestone.
