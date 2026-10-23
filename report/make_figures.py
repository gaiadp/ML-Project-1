"""Figure of the report: what each ingredient is worth in F1, fold by fold.

Reads only the JSON files of results/ and writes report/figures/ablation.pdf.
Every bar is the gain F1(with the ingredient) - F1(without it), mean over the same
5 cross-validation folds; the dots are the 5 per-fold gains. For a run_grid file
the row of a variant is its best row (highest f1_mean over the lambda grid), as in
results/ablation.md.

Usage (from the repository root):
    python report/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "figures" / "ablation.pdf"

INK = "#222222"
MUTED = "#666666"
BAR = "#3a6ea5"
LEAK = "#bbbbbb"


def load(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def folds(result, label=None):
    """F1 per fold of a result file: its best row, or the best row with this label."""
    if label is None:
        return np.array(result["best"]["f1_folds"])
    rows = [r for r in result["rows"] if r["label"] == label and r["finite"]]
    if not rows:
        raise KeyError(f"no row {label!r} in {result['phase']}")
    return np.array(max(rows, key=lambda r: r["f1_mean"])["f1_folds"])


def gains():
    """(group, label, per-fold gain) of every bar, top to bottom."""
    data = load("reg_logistic_newton/ablation_data.json")
    ref = folds(data, "final")
    inter = load("reg_logistic_newton/03_interactions.json")
    newton = folds(load("logistic_boosting/ablation_model.json"))  # n_trees = 0
    gd = folds(load("reg_logistic_regression/ablation_model.json"))
    boosting = folds(load("logistic_boosting/04_final.json"))
    harehab1 = folds(load("logistic_boosting/extra/04_final_harehab1.json"))
    ens = load("ensemble/ensemble.json")["cv"]
    prep = "Data preparation (Newton)"
    model = "Model"
    leak = "Label leakage (not used)"
    return [
        (prep, "One-hot of categories", ref - folds(data, "no one-hot encoding (B9)")),
        (
            prep,
            "One-hot of non-monotone ordinals",
            ref - folds(data, "no one-hot of the ordinals (B9)"),
        ),
        (
            prep,
            "Missing-value indicators",
            ref - folds(data, "no missing indicators (B6)"),
        ),
        (
            prep,
            "Bins of bad-health days",
            ref - folds(data, "no binning of PHYSHLTH (B10)"),
        ),
        (
            prep,
            "Rule-based column selection",
            ref - folds(data, "no feature selection"),
        ),
        (model, "Newton instead of plain GD", newton - gd),
        (
            model,
            "Top-20 products (Newton)",
            folds(inter, "top 20") - folds(inter, "top 0"),
        ),
        (model, "Trees on the residuals", boosting - newton),
        (
            model,
            "Ensemble vs. boosting alone",
            np.array(ens["ensemble"]["f1_folds"])
            - np.array(ens["logistic_boosting alone"]["f1_folds"]),
        ),
        (leak, "HAREHAB1 as a feature", harehab1 - boosting),
    ]


def main():
    bars = gains()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Liberation Serif",
                "TeX Gyre Termes",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "font.size": 7,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.6,
            "xtick.color": MUTED,
            "xtick.major.width": 0.6,
            "pdf.fonttype": 42,
        }
    )
    # y positions: a header line before each group, then its bars
    ticks, labels, header, y = [], [], [], 0.0
    pos = []
    group = None
    for g, label, d in bars:
        if g != group:
            group = g
            if y:
                y += 0.35
            ticks.append(y)
            labels.append(g)
            header.append(True)
            y += 1
        ticks.append(y)
        labels.append(label)
        header.append(False)
        pos.append(y)
        y += 1

    # fixed layout, drawn at the width of one column: label column, plot, numbers
    fig = plt.figure(figsize=(3.45, 2.0))
    ax = fig.add_axes([0.395, 0.17, 0.405, 0.81])
    lab = blended_transform_factory(fig.transFigure, ax.transData)
    for (g, label, d), p in zip(bars, pos):
        leak = g.startswith("Label")
        ax.barh(
            p,
            d.mean(),
            height=0.62,
            color=LEAK if leak else BAR,
            hatch="////" if leak else None,
            edgecolor=MUTED if leak else "white",
            linewidth=0.4,
            zorder=2,
        )
        ax.scatter(
            d,
            np.full(len(d), p),
            s=6,
            facecolor="white",
            edgecolor=INK,
            linewidth=0.5,
            zorder=3,
        )
        n_better = int((d > 0).sum())
        print(f"{label:38s} {d.mean():+.4f}  better in {n_better}/{len(d)}")
        ax.text(0.815, p, f"{d.mean():+.4f}", transform=lab, va="center", color=INK)
        ax.text(0.93, p, f"{n_better}/{len(d)}", transform=lab, va="center", color=INK)
    for t, text, h in zip(ticks, labels, header):
        ax.text(
            0.005,
            t,
            text,
            transform=lab,
            va="center",
            color=MUTED if h else INK,
            fontweight="bold" if h else "normal",
        )
    ax.text(
        0.815,
        ticks[0],
        "\u0394F1",
        transform=lab,
        va="center",
        color=MUTED,
        fontweight="bold",
    )
    ax.text(
        0.93,
        ticks[0],
        "folds",
        transform=lab,
        va="center",
        color=MUTED,
        fontweight="bold",
    )
    ax.axvline(0, color=INK, linewidth=0.6, zorder=1)
    ax.set_yticks([])
    ax.set_ylim(ticks[-1] + 0.6, ticks[0] - 0.6)
    ax.set_xlim(-0.0022, 0.009)
    ax.set_xticks([0, 0.0025, 0.005, 0.0075])
    ax.set_xticklabels(["0", "0.0025", "0.005", "0.0075"])
    ax.set_xlabel("Gain in F1 from adding the ingredient", color=INK, labelpad=2)
    ax.grid(axis="x", color="#e3e3e3", linewidth=0.5, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
