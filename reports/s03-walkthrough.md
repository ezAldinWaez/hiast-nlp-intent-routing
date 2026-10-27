# S03 walkthrough

1. Frame the decision: supported intent or rejection. Explain false acceptance versus rejecting a valid request.
2. Inspect training examples and the fixed training/validation split. Keep final-test examples out of class exploration.
3. Trace regex tokens, Porter stems, and unigram/bigram sets for one request.
4. Match against training examples using Jaccard overlap; inspect the nearest example. The similarity is not a probability.
5. Run the comparisons: majority, baseline, no stemming, unigrams only, and dropped negation. Each lexical variant tunes its own rejection threshold on validation.
6. Read metrics and several errors from `outputs/baseline/`. Explain one failure and propose the next experiment.

**Required:** W01–W02 techniques. **Bonus:** a justified W03–W04 extension once the student can explain its tools; not needed for the core submission. Evaluation metrics and threshold selection are supporting practical material.

Each milestone report states the decision, evidence, and interpretation. Dates follow actual two-week periods, not THEO file pairs.
