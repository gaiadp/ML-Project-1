"""Plotting utilities (matplotlib/seaborn only). Figures for the report go to report/figures/."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette: one accent for data, one for fitted/secondary marks, muted ink for context.
C_DATA = "#2a78d6"
C_FIT = "#eb6834"
C_MUTED = "#8a8985"
C_TEXT = "#52514e"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": C_MUTED,
        "axes.labelcolor": C_TEXT,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.color": C_TEXT,
        "ytick.color": C_TEXT,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
    }
)


def plot_missing_and_balance(missing_frac, y01, path):
    """Two-panel overview of data quality and class balance, to run before any modelling.

    Left panel - histogram of the fraction of missing values per feature (NaN after
    `preprocessing.codes_to_nan`, i.e. counting "don't know"/"refused" as missing too).
    It shows how many columns are almost complete and how many are almost empty
    (in BRFSS many questions are only asked to a sub-group, so they are mostly NaN).
    Use it to choose and justify the threshold of `drop_high_missing_columns`
    (e.g. "we drop features with more than 50% missing values") and to decide where
    imputation / missing-indicator features are needed.

    Right panel - number of samples per class, annotated with its percentage.
    Only ~9% of the training samples have MICHD, so the problem is strongly imbalanced:
    a model always predicting "no MICHD" already reaches ~91% accuracy. This motivates
    reporting F1 rather than accuracy, and using class weighting / resampling or tuning
    the decision threshold instead of the default 0.5.

    Parameters
    ----------
    missing_frac : np.ndarray, shape (D,), fraction of NaN per feature
    y01 : np.ndarray, shape (N,), labels in {0, 1} (1 = MICHD)
    path : str, output file (e.g. report/figures/missing_balance.png)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw={"width_ratios": [3, 1]})
    ax1.hist(missing_frac, bins=40, color=C_DATA, edgecolor="white", linewidth=1)
    ax1.set_xlabel("fraction of missing values (NaN after cleaning)")
    ax1.set_ylabel("# features")
    ax1.set_title("Missingness per feature")

    counts = np.array([(y01 == 0).sum(), (y01 == 1).sum()])
    ax2.bar(["no MICHD", "MICHD"], counts, color=C_DATA, width=0.6)
    for i, c in enumerate(counts):
        ax2.text(i, c, f"{c / counts.sum():.1%}", ha="center", va="bottom", fontsize=8, color=C_TEXT)
    ax2.set_title("Class balance")
    ax2.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def plot_ranking(names, scores, path, xlabel, title, top=30):
    """Horizontal bar chart of the `top` highest scores."""
    order = np.argsort(scores)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6, 0.22 * len(order) + 0.8))
    ax.barh(np.array(names)[order], scores[order], color=C_DATA, height=0.7)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def plot_mi_vs_corr(names, mi, abs_r, path, n_labels=15):
    """MI (any dependence) vs |Pearson r| (linear dependence).

    Features far above the bulk have high MI but low |r|: the dependence is
    non-linear / non-monotonic or carried by missingness -> need one-hot, binning
    or polynomial terms rather than a single linear coefficient.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(abs_r, mi, s=14, color=C_DATA, alpha=0.7, edgecolors="none")
    idx = np.argsort(mi)[::-1][:n_labels]
    for i in idx:
        ax.annotate(names[i], (abs_r[i], mi[i]), fontsize=7, color=C_TEXT,
                    xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("|Pearson r| with target (non-missing rows)")
    ax.set_ylabel("mutual information with target [bits]")
    ax.set_title("Linear vs total dependence")
    fig.savefig(path)
    plt.close(fig)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def plot_dependence_grid(names, cols, y01, kinds, path, ncols=4, min_count=30):
    """One panel per feature: empirical log-odds of the target per value/bin, with 95% CI.

    - categorical/binary: one point per category -> look for non-monotonic patterns
      (=> one-hot encode instead of treating the code as a number).
    - numeric: one point per quantile bin + weighted linear fit on the log-odds scale.
      Points following the line => a linear term is enough for logistic regression;
      curvature => add polynomial terms / log transform / binning.
    - the hollow point at the right is the NaN group: if it differs from the rest,
      missingness is informative (=> add an "is_missing" indicator).
    """
    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.4 * nrows), squeeze=False)
    base = _logit(y01.mean())

    for ax, name, x, kind in zip(axes.flat, names, cols, kinds):
        nan = np.isnan(x)
        xv, yv = x[~nan], y01[~nan]
        if kind == "numeric":
            edges = np.unique(np.quantile(xv, np.linspace(0, 1, 16)))
            b = np.clip(np.searchsorted(edges, xv, side="right") - 1, 0, len(edges) - 2)
            groups = [b == k for k in range(len(edges) - 1)]
            pos = np.array([np.median(xv[g]) if g.any() else np.nan for g in groups])
        else:
            vals = np.unique(xv)
            groups = [xv == v for v in vals]
            pos = vals.astype(float)

        cnt = np.array([g.sum() for g in groups])
        p = np.array([yv[g].mean() if c else np.nan for g, c in zip(groups, cnt)])
        ok = cnt >= min_count
        lo, se = _logit(p[ok]), 1 / np.sqrt(cnt[ok] * p[ok] * (1 - p[ok]) + 1e-9)
        ax.errorbar(pos[ok], lo, yerr=1.96 * se, fmt="o", ms=4, color=C_DATA,
                    ecolor=C_DATA, elinewidth=1, capsize=0)
        if kind == "numeric" and ok.sum() >= 3:
            w = 1 / se ** 2
            coef = np.polyfit(pos[ok], lo, 1, w=np.sqrt(w))
            xs = np.linspace(pos[ok].min(), pos[ok].max(), 50)
            ax.plot(xs, np.polyval(coef, xs), color=C_FIT, lw=1.5)
        elif kind != "numeric":
            ax.plot(pos[ok], lo, color=C_DATA, lw=0.8, alpha=0.5)

        if nan.sum() >= min_count:
            pn = y01[nan].mean()
            xr = np.nanmax(pos) if np.isfinite(pos).any() else 0
            span = (np.nanmax(pos) - np.nanmin(pos)) if np.isfinite(pos).any() else 1
            xn = xr + 0.12 * (span or 1)
            sen = 1 / np.sqrt(nan.sum() * pn * (1 - pn) + 1e-9)
            ax.errorbar([xn], [_logit(pn)], yerr=1.96 * sen, fmt="o", ms=5, mfc="white",
                        color=C_DATA, elinewidth=1)
            ax.annotate("NaN", (xn, _logit(pn)), xytext=(4, -3), textcoords="offset points",
                        fontsize=7, color=C_TEXT)

        ax.axhline(base, color=C_MUTED, ls="--", lw=0.8)
        ax.set_title(f"{name}  ({kind}, {nan.mean():.0%} NaN)", fontsize=8.5)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("log-odds MICHD")
    fig.suptitle("Empirical log-odds per value/bin (dashed = overall prevalence, "
                 "orange = linear fit)", fontsize=10, color=C_TEXT)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_corr_heatmap(names, x, path):
    """Pearson correlation between features (NaN -> column median). Redundant pairs |r| > ~0.8."""
    x = np.where(np.isnan(x), np.nanmedian(x, axis=0), x)
    c = np.corrcoef(x, rowvar=False)
    fig, ax = plt.subplots(figsize=(0.28 * len(names) + 2, 0.28 * len(names) + 1.5))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_title("Correlation among top features (redundancy check)")
    fig.savefig(path)
    plt.close(fig)
