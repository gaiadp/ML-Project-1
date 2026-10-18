"""Baseline comparison for the report: every method against trivial rules, one table.

Phase 01 of the per-method experiments (experiments/<method>/01_baseline.py) already
runs every method in 5-fold CV on the same folds, with the pipelines v0, full and v1,
and saves the scores in results/<method>/01_baseline.json. This script does not train
the methods again (unless --rerun): it reads those files and puts them in one table,
together with trivial rules computed here on the same folds:
    everyone MICHD     predict MICHD for everybody (F1 = 2p / (1 + p), p = 8.8%)
    age group          predict MICHD for the age groups with the highest prevalence
    age group x sex    the same on the cells age group x sex
    general health     the same on the answers of GENHLTH
For a rule, the prevalence of every value is computed on the training part of the
fold, the cut (which values are predicted MICHD) maximizes F1 on the training part,
and F1 is measured on the validation part, as for the methods. (experiments/eda.py
A5 gives the same rules measured on the rows they are fitted on: slightly higher.)

The hyperparameters are tuned inside the phases (grids of 01_baseline, 02_degree,
03_interactions; the decision threshold inside every fold): the table also shows the
best setting of every method.

Usage (from the repository root):
    python experiments/baselines.py            # table from results/*/01_baseline.json
    python experiments/baselines.py --rerun    # first run phase 01 of every method
                                               # again (several hours)
    python experiments/baselines.py --rerun ridge_regression least_squares
                                               # run again only these methods
Writes results/baselines.md (Markdown) and results/baselines.tex (LaTeX table).
"""

import argparse
import os
import runpy
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import RESULTS_DIR, load_result  # noqa: E402
from src.cross_validation import stratified_k_indices  # noqa: E402
from src.data import load_data  # noqa: E402
from src.metrics import best_threshold, f1_score, predict_labels  # noqa: E402
from src.models import EXTRA_METHODS, METHODS  # noqa: E402
from src.preprocessing import codes_to_nan  # noqa: E402

K_FOLD = 5
SEED = 1  # the folds of the phases: the same in every experiments/<method>/ script
PIPELINES = ("v0", "full", "v1")
# trivial rules: name -> the columns whose values (or combinations of values) it uses
TRIVIAL_RULES = {
    "everyone MICHD": [],
    "age group": ["_AGEG5YR"],
    "age group x sex": ["_AGEG5YR", "SEX"],
    "general health": ["GENHLTH"],
}


def rule_keys(columns):
    """One integer key per row for the combination of its values.

    Args:
        columns: values after codes_to_nan, shape=(N, k); NaN is a value of its own

    Returns:
        keys, shape=(N,), int64; all equal if k = 0
    """
    keys = np.zeros(columns.shape[0], dtype=np.int64)
    for j in range(columns.shape[1]):
        codes = np.where(np.isnan(columns[:, j]), -1, columns[:, j]).astype(np.int64)
        keys = keys * 1000 + codes + 1  # codes < 999: no two combinations collide
    return keys


def prevalence_scores(keys_train, y_train, keys):
    """Score of every row: the MICHD prevalence of its key in the training rows.

    Args:
        keys_train: key of every training row, shape=(N_train,)
        y_train: training labels, shape=(N_train,), 1 = MICHD
        keys: keys of the rows to score; a key never seen in training gets the
            overall training prevalence

    Returns:
        scores, shape=(len(keys),)
    """
    positive = (np.asarray(y_train) == 1).astype(float)
    values, inverse = np.unique(keys_train, return_inverse=True)
    prevalence = np.bincount(inverse, weights=positive) / np.bincount(inverse)
    scores = np.full(len(keys), positive.mean())
    where = np.minimum(np.searchsorted(values, keys), len(values) - 1)
    seen = values[where] == keys
    scores[seen] = prevalence[where[seen]]
    return scores


def trivial_baselines(x, y, names):
    """F1 (mean, std over the folds) of every rule of TRIVIAL_RULES, on the folds of
    the phases (stratified, K_FOLD, SEED)."""
    col = {n: j for j, n in enumerate(names)}
    k_indices = stratified_k_indices(y, K_FOLD, SEED)
    out = {}
    for rule, columns in TRIVIAL_RULES.items():
        keys = rule_keys(codes_to_nan(x[:, [col[c] for c in columns]], columns))
        f1 = []
        for k in range(K_FOLD):
            val = k_indices[k]
            tr = np.concatenate([k_indices[j] for j in range(K_FOLD) if j != k])
            scores_tr = prevalence_scores(keys[tr], y[tr], keys[tr])
            threshold, _ = best_threshold(scores_tr, y[tr])
            scores_va = prevalence_scores(keys[tr], y[tr], keys[val])
            f1.append(f1_score(y[val], predict_labels(scores_va, threshold)))
        out[rule] = (float(np.mean(f1)), float(np.std(f1)))
    return out


def best_rows(rows):
    """Best row of every pipeline: highest mean F1, among the rows that did not
    diverge."""
    best = {}
    for row in rows:
        label = row["label"]
        if row["finite"] and (
            label not in best or row["f1_mean"] > best[label]["f1_mean"]
        ):
            best[label] = row
    return best


def method_results(methods):
    """For every method: (method, {pipeline: best row}), or (method, None) if its
    phase 01 has not been run."""
    out = []
    for method in methods:
        try:
            rows = load_result(method, "01_baseline")["rows"]
        except FileNotFoundError:
            out.append((method, None))
            continue
        out.append((method, best_rows(rows)))
    return out


def _f1_text(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def _setting(best):
    """Label and fit parameters of the best row over all the pipelines."""
    top = max(best.values(), key=lambda r: r["f1_mean"])
    fit = ", ".join(f"{k}={v:g}" for k, v in top["fit"].items()) or "no parameters"
    return f"{top['label']}: {fit}"


def markdown(trivial, results):
    """The two tables (trivial rules, methods) in Markdown."""
    lines = ["| Trivial rule | F1 (5-fold CV) |", "|---|---|"]
    lines += [f"| {rule} | {_f1_text(*f1)} |" for rule, f1 in trivial.items()]
    lines += ["", "| Method | " + " | ".join(PIPELINES) + " | best setting |"]
    lines += ["|---" * (len(PIPELINES) + 2) + "|"]
    for method, best in results:
        if best is None:
            lines.append(f"| {method} | " + "not run | " * len(PIPELINES) + "|")
            continue
        cells = [
            _f1_text(best[p]["f1_mean"], best[p]["f1_std"]) if p in best else "–"
            for p in PIPELINES
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + f" | {_setting(best)} |")
    return "\n".join(lines) + "\n"


def latex(trivial, results):
    """One LaTeX tabular: mean F1 per method and pipeline, trivial rules on top."""
    n = len(PIPELINES)
    lines = [
        r"\begin{tabular}{l" + "c" * n + "}",
        r"\hline",
        "Method & " + " & ".join(PIPELINES) + r" \\",
        r"\hline",
    ]
    for rule, (mean, _) in trivial.items():
        lines.append(f"{rule} & \\multicolumn{{{n}}}{{c}}{{{mean:.3f}}} \\\\")
    lines.append(r"\hline")
    for method, best in results:
        if best is None:
            continue
        cells = [f"{best[p]['f1_mean']:.4f}" if p in best else "--" for p in PIPELINES]
        name = method.replace("_", r"\_")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def rerun(methods):
    """Run phase 01 of the given methods again (their 01_baseline.py, as it is)."""
    for method in methods:
        script = os.path.join(ROOT, "experiments", method, "01_baseline.py")
        print(f"\n=== {method}: {os.path.relpath(script, ROOT)}")
        runpy.run_path(script, run_name="__main__")


def main():
    methods = list(METHODS) + list(EXTRA_METHODS)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rerun",
        nargs="*",
        metavar="METHOD",
        help="run phase 01 again before the table: every method, or only the "
        "given ones (several hours for all)",
    )
    args = parser.parse_args()
    if args.rerun is not None:
        unknown = sorted(set(args.rerun) - set(methods))
        if unknown:
            parser.error(f"unknown methods {unknown}, expected some of {methods}")
        rerun(args.rerun or methods)

    x, x_test, y, _, _, names = load_data()
    del x_test
    trivial = trivial_baselines(x, y, names)
    del x
    results = method_results(methods)

    table = markdown(trivial, results)
    print("\n" + table)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outputs = {"baselines.md": table, "baselines.tex": latex(trivial, results)}
    for name, text in outputs.items():
        path = os.path.join(RESULTS_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"saved {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
