"""Small reusable plots for inspecting milestone data and results."""

import matplotlib.pyplot as plt

BLUE, ORANGE, TEAL = "#2166ac", "#c56a16", "#17806d"


def _axes(title, xlabel, size):
    fig, ax = plt.subplots(figsize=size, dpi=110)
    ax.set(title=title, xlabel=xlabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.18)
    return fig, ax


def plot_counts(labels, counts, title):
    """Show counts and shares for mutually exclusive categories."""
    fig, ax = _axes(title, "Requests", (7.5, max(2.2, .55 * len(labels) + 1)))
    bars = ax.barh(labels, counts, color=BLUE, height=.55)
    total = sum(counts)
    ax.bar_label(bars, labels=[f"{n:,} ({n / total:.1%})" if total else "0" for n in counts],
                 padding=5, fontsize=10)
    ax.set_xlim(0, max(counts, default=0) * 1.45 or 1)
    ax.invert_yaxis()
    fig.tight_layout()
    plt.show()


def plot_f1(results):
    """Compare in-scope and OOS F1 on the same zero-to-one scale."""
    fig, ax = _axes("Validation F1 by configuration", "F1 score", (7.5, 3.5))
    positions = list(range(len(results)))
    for offset, key, label, color, hatch in [
        (-.18, "in_scope_macro_f1", "In-scope macro-F1", BLUE, ""),
        (.18, "oos_f1", "OOS F1", ORANGE, "//"),
    ]:
        values = [scores[key] for scores in results.values()]
        bars = ax.barh([p + offset for p in positions], values, height=.32,
                       label=label, color=color, hatch=hatch)
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=4, fontsize=9)
    ax.set_yticks(positions, list(results))
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_rejection(trials, threshold):
    """Show the two error rates across the tested rejection thresholds."""
    fig, ax = _axes("Rejection trade-off on validation", "Rejection threshold (similarity)", (7.5, 3.3))
    x = [row["threshold"] for row in trials]
    ax.plot(x, [row["oos_false_acceptance_rate"] for row in trials],
            color=ORANGE, marker="o", markersize=3, label="Unsupported requests accepted")
    ax.plot(x, [row["in_scope_rejection_rate"] for row in trials],
            color=TEAL, linestyle="--", marker="s", markersize=3, label="Supported requests rejected")
    ax.axvline(threshold, color=".35", linestyle=":", label=f"Selected threshold = {threshold:.2f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), ylabel="Fraction of the corresponding query group")
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    plt.show()
