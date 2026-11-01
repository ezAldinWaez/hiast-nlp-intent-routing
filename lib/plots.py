"""Small reusable plots for inspecting milestone data and results."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter

from .evaluation import auc
from .model import OOS, reject

BLUE, ORANGE, TEAL = "#2166ac", "#c56a16", "#17806d"


def _axes(title, xlabel, size):
    fig, ax = plt.subplots(figsize=size, dpi=110)
    ax.set(title=title, xlabel=xlabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.18)
    return fig, ax


def plot_counts(labels, counts, title):
    """Show counts and shares for mutually exclusive categories."""
    fig, ax = _axes(title, "Requests", (7.5, max(2.2, 0.55 * len(labels) + 1)))
    bars = ax.barh(labels, counts, color=BLUE, height=0.55)
    total = sum(counts)
    ax.bar_label(
        bars,
        labels=[f"{n:,} ({n / total:.1%})" if total else "0" for n in counts],
        padding=5,
        fontsize=10,
    )
    ax.set_xlim(0, max(counts, default=0) * 1.45 or 1)
    ax.invert_yaxis()
    fig.tight_layout()
    plt.show()


def plot_f1(results):
    """Compare in-scope and OOS F1 on the same zero-to-one scale."""
    fig, ax = _axes(
        "Validation F1 by configuration",
        "F1 score",
        (8.2, max(3.5, 0.65 * len(results) + 1)),
    )
    positions = list(range(len(results)))
    for offset, key, label, color, hatch in [
        (-0.18, "in_scope_macro_f1", "In-scope macro-F1", BLUE, ""),
        (0.18, "oos_f1", "OOS F1", ORANGE, "//"),
    ]:
        values = [scores[key] for scores in results.values()]
        bars = ax.barh(
            [p + offset for p in positions],
            values,
            height=0.32,
            label=label,
            color=color,
            hatch=hatch,
        )
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=4, fontsize=9)
    ax.set_yticks(positions, list(results))
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_rejection(trials, threshold):
    """Show the two error rates across the tested rejection thresholds."""
    fig, ax = _axes(
        "Rejection trade-off on validation",
        "Rejection threshold (similarity)",
        (7.5, 3.3),
    )
    x = [row["threshold"] for row in trials]
    ax.plot(
        x,
        [row["oos_false_acceptance_rate"] for row in trials],
        color=ORANGE,
        marker="o",
        markersize=3,
        label="Unsupported requests accepted",
    )
    ax.plot(
        x,
        [row["in_scope_rejection_rate"] for row in trials],
        color=TEAL,
        linestyle="--",
        marker="s",
        markersize=3,
        label="Supported requests rejected",
    )
    ax.axvline(
        threshold,
        color=".35",
        linestyle=":",
        label=f"Selected threshold = {threshold:.2f}",
    )
    ax.set(
        xlim=(0, 1), ylim=(0, 1.02), ylabel="Fraction of the corresponding query group"
    )
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_lengths(groups):
    """Compare token counts as shares within each group."""
    maximum = max(max(lengths) for lengths in groups.values())
    token_counts = np.arange(maximum + 1)
    fig, ax = _axes(
        "Request lengths in development data", "Tokens per request", (8.2, 4.2)
    )
    for (name, lengths), color, style, marker in zip(
        groups.items(), [BLUE, ORANGE, TEAL], ["-", "--", ":"], ["o", "s", "^"]
    ):
        counts = np.bincount(lengths, minlength=maximum + 1)
        ax.plot(
            token_counts,
            counts / len(lengths),
            color=color,
            linestyle=style,
            marker=marker,
            markersize=3.5,
            linewidth=1.5,
            label=f"{name} (n={len(lengths):,})",
        )
    ax.set(
        xlim=(0, maximum), ylim=(0, None), ylabel="Share of requests within each group"
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=14, integer=True))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_roc(truth, matches, threshold, name):
    """Plot rejection across all distinct scores, keeping tied scores together."""
    from itertools import groupby

    scores = [1 - similarity for _, similarity, _ in matches]
    pairs = sorted(zip(scores, truth, strict=True), reverse=True)
    unsupported = sum(label == OOS for label in truth)
    supported = len(truth) - unsupported
    if not supported or not unsupported:
        raise ValueError("ROC needs both supported and unsupported requests")
    false_rejections = correct_rejections = 0
    x, y = [0.0], [0.0]
    for _, group in groupby(pairs, key=lambda pair: pair[0]):
        for _, label in group:
            correct_rejections += label == OOS
            false_rejections += label != OOS
        x.append(false_rejections / supported)
        y.append(correct_rejections / unsupported)

    predicted = [reject(label, score, threshold) for label, score, _ in matches]
    selected_x = (
        sum(t != OOS and p == OOS for t, p in zip(truth, predicted)) / supported
    )
    selected_y = (
        sum(t == OOS and p == OOS for t, p in zip(truth, predicted)) / unsupported
    )
    fig, ax = _axes(
        f"ROC curve: {supported:,} supported + {unsupported:,} unsupported validation requests",
        "Supported requests rejected by mistake (false positive rate)",
        (7.5, 5.2),
    )
    ax.plot(x, y, color=BLUE, label=f"{name}: AUROC = {auc(truth, scores):.4f}")
    ax.fill_between(x, y, color=BLUE, alpha=0.10)
    ax.plot(
        [0, 1], [0, 1], color=".45", linestyle="--", label="Random scores: AUROC = 0.5"
    )
    ax.scatter(
        [selected_x],
        [selected_y],
        color=ORANGE,
        marker="o",
        s=60,
        zorder=3,
        label=f"Threshold {threshold:.2f}: {selected_x:.1%} mistaken / {selected_y:.0%} correct rejections",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        ylabel="Unsupported requests correctly rejected (true positive rate)",
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    plt.show()
