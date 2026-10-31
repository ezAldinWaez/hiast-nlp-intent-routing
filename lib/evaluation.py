"""Small explicit evaluation utilities; out-of-scope is the positive class."""

from collections import Counter

from .model import OOS, reject


def divide(a, b):
    return a / b if b else 0.0


def auc(truth, scores):
    """Mannâ€“Whitney AUROC with average ranks for tied scores."""
    pairs = sorted(zip(scores, [y == OOS for y in truth], strict=True))
    positives = sum(y for _, y in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += sum(y for _, y in pairs[i:j]) * (i + 1 + j) / 2
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def metrics(truth, predicted, labels, oos_scores):
    pairs = list(zip(truth, predicted, strict=True))
    # In-scope metrics consider only in-scope queries; rejection is an error.
    inside = [(y, p) for y, p in pairs if y != OOS]
    true_counts = Counter(y for y, _ in inside)
    pred_counts = Counter(p for _, p in inside)
    hits = Counter(y for y, p in inside if y == p)
    macro = sum(divide(2 * hits[label], true_counts[label] + pred_counts[label])
                for label in labels) / len(labels)
    tp = sum(y == OOS and p == OOS for y, p in pairs)
    fp = sum(y != OOS and p == OOS for y, p in pairs)
    fn = sum(y == OOS and p != OOS for y, p in pairs)
    result = {
        "in_scope_accuracy": divide(sum(hits.values()), len(inside)),
        "in_scope_macro_f1": macro,
        "oos_precision": divide(tp, tp + fp),
        "oos_recall": divide(tp, tp + fn),
        "oos_f1": divide(2 * tp, 2 * tp + fp + fn),
        "oos_auroc": auc(truth, oos_scores),
        "oos_false_acceptance_rate": divide(fn, tp + fn),
        "in_scope_rejection_rate": divide(fp, len(inside)),
    }
    result["selection_score"] = (macro + result["oos_f1"]) / 2
    return result


def select_threshold(truth, matches, labels, thresholds):
    if not thresholds or any(not 0 <= t <= 1 for t in thresholds):
        raise ValueError("Provide thresholds in [0, 1]")
    trials = []
    for threshold in sorted(set(thresholds)):
        predictions = [reject(label, score, threshold) for label, score, _ in matches]
        result = metrics(truth, predictions, labels, [1 - s for _, s, _ in matches])
        trials.append({"threshold": threshold, **result})
    # On equal objectives prefer the lower rejection threshold.
    return max(trials, key=lambda r: (r["selection_score"], -r["threshold"])), trials
