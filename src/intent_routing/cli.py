"""Download, run validation comparisons, and route individual requests."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from .baseline import Features, Router, reject
from .data import REVISION, SHA256, development, download, load
from .evaluation import metrics, select_threshold


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args):
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train, validation, labels = development(load(args.data))
    truth = [y for _, y in validation]
    majority = sorted(Counter(y for _, y in train).items(), key=lambda p: (-p[1], p[0]))[0][0]
    results = {"majority": metrics(truth, [majority] * len(truth), labels, [0] * len(truth))}
    # Each variant selects its own threshold using validation only.
    variants = {
        "baseline": Features(config["ngram_max"], config["stemming"], config["drop_negation"]),
        "no_stemming": Features(config["ngram_max"], False, config["drop_negation"]),
        "unigrams": Features(1, config["stemming"], config["drop_negation"]),
        "drop_negation": Features(config["ngram_max"], config["stemming"], True),
    }
    for name, features in variants.items():
        router = Router(train, features)
        matches = [router.match(text) for text, _ in validation]
        best, trials = select_threshold(truth, matches, labels, config["thresholds"])
        results[name] = best
        write_json(args.output / f"{name}-thresholds.json", trials)
        errors = []
        for (text, expected), (label, score, index) in zip(validation, matches, strict=True):
            predicted = reject(label, score, best["threshold"])
            if predicted != expected:
                errors.append({"text": text, "expected": expected, "predicted": predicted,
                               "similarity": score, "training_index": index})
        write_json(args.output / f"{name}-errors.json", errors)
        if name == "baseline":
            write_json(args.output / "selected.json", {"config": config, "threshold": best["threshold"],
                       "dataset_sha256": SHA256})
        print(f"{name}: threshold={best['threshold']:.2f}, selection={best['selection_score']:.4f}", flush=True)
    normalized = lambda text: " ".join(text.lower().split())
    train_texts = {normalized(text) for text, _ in train}
    overlap = sum(normalized(text) in train_texts for text, _ in validation)
    write_json(args.output / "metrics.json", {
        "dataset_revision": REVISION, "dataset_sha256": SHA256,
        "config": config, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "split": "val + oos_val", "training_examples": len(train),
        "validation_examples": len(validation), "intent_count": len(labels),
        "normalized_train_validation_overlap_rows": overlap,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "note": "Thresholds selected on these validation rows; these are not held-out test estimates.",
        "results": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download")
    get.add_argument("--data", type=Path, default=Path("data/raw/data_full.json"))
    experiment = commands.add_parser("run", help="Compare W01–W02 variants on validation only")
    experiment.add_argument("--data", type=Path, default=Path("data/raw/data_full.json"))
    experiment.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    experiment.add_argument("--output", type=Path, default=Path("outputs/baseline"))
    predict = commands.add_parser("predict")
    predict.add_argument("text")
    predict.add_argument("--data", type=Path, default=Path("data/raw/data_full.json"))
    predict.add_argument("--selection", type=Path, default=Path("outputs/baseline/selected.json"))
    args = parser.parse_args()
    if args.command == "download":
        download(args.data)
        print(args.data)
    elif args.command == "run":
        run(args)
    else:
        selected = json.loads(args.selection.read_text(encoding="utf-8"))
        if selected["dataset_sha256"] != SHA256:
            raise ValueError("Selected threshold belongs to a different dataset")
        config = selected["config"]
        train, _, _ = development(load(args.data))
        router = Router(train, Features(config["ngram_max"], config["stemming"], config["drop_negation"]))
        label, score, index = router.match(args.text)
        print(json.dumps({"intent": reject(label, score, selected["threshold"]),
                          "similarity": score, "threshold": selected["threshold"],
                          "nearest_training_example": train[index][0] if index is not None else None}, indent=2))


if __name__ == "__main__":
    main()
