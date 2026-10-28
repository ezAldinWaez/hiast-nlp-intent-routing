"""Run the baseline, inspect validation errors, and route one example."""

from collections import Counter
import json
from pathlib import Path
import sys

from dataset import SHA256, development, download, load
from evaluation import metrics, select_threshold
from model import Router, reject

STEMMING = True
NGRAM_MAX = 2
THRESHOLDS = [i / 20 for i in range(21)]
ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "data/raw/data_full.json"
    download(path)  # Uses the local file when present; verifies its checksum.
    train, validation, labels = development(load(path))
    truth = [label for _, label in validation]
    router = Router(train, STEMMING, NGRAM_MAX)
    matches = [router.match(text) for text, _ in validation]
    best, trials = select_threshold(truth, matches, labels, THRESHOLDS)
    counts = Counter(label for _, label in train)
    majority = min(counts, key=lambda label: (-counts[label], label))
    reference = metrics(truth, [majority] * len(truth), labels, [0] * len(truth))
    errors = []
    for (text, expected), (label, score, _) in zip(validation, matches, strict=True):
        predicted = reject(label, score, best["threshold"])
        if predicted != expected:
            errors.append({"text": text, "expected": expected, "predicted": predicted})
    normalize = lambda text: " ".join(text.lower().split())
    training_texts = {normalize(text) for text, _ in train}
    result = {
        "dataset_sha256": SHA256, "split": "val + oos_val",
        "settings": {"stemming": STEMMING, "ngram_max": NGRAM_MAX},
        "majority": reference, "baseline": best, "thresholds": trials,
        "train_validation_overlap_rows": sum(normalize(t) in training_texts for t, _ in validation),
        "errors": errors,
    }
    output = ROOT / "outputs/results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Validation results (also used to choose the threshold):")
    for name, scores in [("Majority", reference), ("Baseline", best)]:
        print(f"{name}: in-scope F1={scores['in_scope_macro_f1']:.4f}, OOS F1={scores['oos_f1']:.4f}")
    print(f"Threshold: {best['threshold']:.2f}. Metrics and errors: {output}")
    text = " ".join(sys.argv[1:]) or "How much money is in my account?"
    label, score, index = router.match(text)
    print(f"{text} -> {reject(label, score, best['threshold'])} (similarity={score:.3f})")
    if index is not None:
        print(f"Nearest training example: {train[index][0]}")


if __name__ == "__main__":
    main()
