"""Ablation study: how much every choice contributes to the final model.

Two tables, on the folds of all the other experiments (5-fold CV, seed 1). Every row
gives the mean F1 over the folds, its difference from the reference row "final", and
in how many folds the variant is better (paired comparison on the same folds).

1. Data preparation: one step of the pipeline switched off at a time, starting from
   the pipeline of the final model (V1_CONFIG). Model: Newton's logistic regression
   (DATA_METHOD), with lambda searched again for every variant (DATA_LAMBDAS), since
   removing or adding columns can move the best lambda. The final model is Newton
   plus a small correction by trees (+0.0015 F1, results/ablation.md), so the steps
   of the pipeline are measured with Newton: ~1.5 min per variant and lambda instead
   of ~44 min (mean per setting of results/logistic_boosting/01_baseline.json).
2. Model: the components of the final model removed, with its hyperparameters fixed
   (their values were already chosen in CV):
       final model        FINAL_METHOD as trained by phase 04 (FINAL_PHASE)
       no trees           n_trees = 0: the logistic regression alone (Newton)
       plain GD           the logistic regression by plain gradient descent
                          (GD_METHOD with the best setting of GD_PHASE) instead of
                          Newton's exact minimum, no trees
       threshold 0.5      the final model with the natural threshold 0.5 instead of
                          the threshold that maximizes F1
   and, if experiments/leakage.py has been run, two rows about HAREHAB1 (the label
   leakage column): the final model with HAREHAB1 as a feature, and the final model
   with the known-positive rule. They are read from its results, not computed here.

The final model is read from results/<FINAL_METHOD>/04_final.json, the setting phase
04 used for the submission (its PIPELINE_OVERRIDES / FIT_OVERRIDES included): run
phase 04 first (experiments/run_phases.py <method>). The ablation does not change
the model, it only measures what every part of it is worth.

The CV results are saved in results/<method>/ablation_data.json and
ablation_model.json; a part whose results exist is not computed again: use --force
after changing the final model or the settings below. The tables go to
results/ablation.md and results/ablation.tex (LaTeX tabular for the report).

Usage (from the repository root):
    python experiments/ablation.py                # missing parts, then the tables
    python experiments/ablation.py --parts data   # only the data-preparation table
    python experiments/ablation.py --force        # compute everything again
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.baselines import best_rows  # noqa: E402
from experiments.common import (  # noqa: E402
    RESULTS_DIR,
    load_best,
    load_result,
    param_grid,
    result_path,
    run_grid,
)
from experiments.leakage import variant_rows  # noqa: E402
from src.feature_types import SELECTION_RULES  # noqa: E402

# ---- settings of the ablation ----
FINAL_METHOD = "logistic_boosting"  # the final model ...
FINAL_PHASE = "04_final"  # ... as trained by this phase (pipeline + fit)
DATA_METHOD = "reg_logistic_newton"  # model of the data-preparation table
DATA_LAMBDAS = [3e-5, 1e-4, 1e-3]  # searched again for every variant (flat optimum)
DATA_MAX_ITERS = 20  # Newton steps (~10 are enough)
GD_METHOD = "reg_logistic_regression"  # "plain GD" row ...
GD_PHASE = "01_baseline"  # ... with the best hyperparameters of this phase
K_FOLD = 5
SEED = 1  # the folds of all the experiments
# ----------------------------------

FINAL = "final"  # label of the reference row
# pipeline steps switched off in the data-preparation table, with their label
STEP_LABELS = {
    "domain_drop": "no domain drop (B2)",
    "drop_redundant": "no duplicate drop (B2)",
    "convert_units": "no unit conversion (B3)",
    "skip_logic_fill": "no skip-logic fill (B4)",
    "missing_indicators": "no missing indicators (B6)",
    "one_hot": "no one-hot encoding (B9)",
    "one_hot_ordinal": "no one-hot of the ordinals (B9)",
    "binning": "no binning of PHYSHLTH (B10)",
}


def data_variants(base):
    """Pipelines of the data-preparation table: label -> config, "final" first.

    Every variant is the final pipeline base with one step switched off (the steps
    already off in base are left out): each rule of the feature selection, the whole
    selection, the selection replaced by the NaN rule B5, the steps of STEP_LABELS,
    the imputation by type.
    """
    variants = {FINAL: base}
    rules = base.get("select_features") or ()
    rules = SELECTION_RULES if rules is True else tuple(rules)
    for rule in rules:
        kept = tuple(r for r in rules if r != rule) or False
        variants[f"selection without {rule}"] = dict(base, select_features=kept)
    if rules:
        variants["no feature selection"] = dict(base, select_features=False)
        if base["max_missing_frac"] >= 1:
            # the full pipeline (DEFAULT_CONFIG): columns chosen by their NaN
            variants["NaN rule (B5) instead of the selection"] = dict(
                base, select_features=False, max_missing_frac=0.5
            )
    for flag, label in STEP_LABELS.items():
        if base.get(flag):
            variants[label] = dict(base, **{flag: False})
    if base.get("impute") == "by_type":
        variants["median imputation (B7)"] = dict(base, impute="median")
    return variants


def _paired(folds, reference):
    """Mean difference from the reference over the folds, and the number of folds
    where it is higher."""
    diff = np.asarray(folds, dtype=float) - np.asarray(reference, dtype=float)
    return float(diff.mean()), int(np.sum(diff > 0))


def _entry(label, folds, reference, **extra):
    diff, better = _paired(folds, reference)
    return dict(
        label=label,
        f1=float(np.mean(folds)),
        std=float(np.std(folds)),
        diff=diff,
        better=better,
        n_folds=len(folds),
        **extra,
    )


def data_table(rows):
    """The data-preparation table from the CV rows of ablation_data.

    Every variant keeps its best lambda; the differences are paired with the best
    row of "final".

    Returns:
        list of dicts (label, f1, std, diff, better, n_folds, lambda_, n_features),
        "final" first
    """
    best = best_rows(rows)
    reference = best[FINAL]["f1_folds"]
    return [
        _entry(
            label,
            row["f1_folds"],
            reference,
            lambda_=row["fit"].get("lambda_"),
            n_features=row["n_features"],
        )
        for label, row in best.items()
    ]


def model_table(final_rows, gd_row, final_fit, extra=()):
    """The model table.

    Args:
        final_rows: CV rows of the final method in ablation_model (the final
            setting and, if it has trees, n_trees = 0)
        gd_row: CV row of plain gradient descent, or None
        final_fit: fit parameters of the final model
        extra: (label, F1 per fold) of more rows, added at the end (the HAREHAB1
            variants of experiments/leakage.py)

    Returns:
        list of dicts (label, f1, std, diff, better, n_folds), the final model first
    """
    final = [r for r in final_rows if r["fit"] == final_fit]
    if not final:
        raise ValueError(
            "the saved model ablation is for another final setting: use --force"
        )
    final = final[0]
    reference = final["f1_folds"]
    table = [_entry(f"{FINAL} model", reference, reference)]
    for row in final_rows:
        if row["fit"].get("n_trees") == 0:
            label = "no trees on the residuals (logistic regression, Newton)"
            table.append(_entry(label, row["f1_folds"], reference))
    if gd_row is not None:
        label = "no trees, plain GD instead of Newton"
        table.append(_entry(label, gd_row["f1_folds"], reference))
    label = "threshold 0.5 instead of the tuned threshold"
    table.append(_entry(label, final["f1_default_folds"], reference))
    table += [_entry(label, folds, reference) for label, folds in extra]
    return table


def run_data(force):
    """CV rows of the data-preparation table, computed if missing (or forced)."""
    phase = "ablation_data"
    path = result_path(DATA_METHOD, phase)
    if force or not os.path.exists(path):
        base = load_best(FINAL_METHOD, FINAL_PHASE)["pipeline"]
        grid = param_grid(lambda_=DATA_LAMBDAS, max_iters=[DATA_MAX_ITERS])
        run_grid(DATA_METHOD, phase, data_variants(base), grid, K_FOLD, SEED)
    else:
        print(f"skip: {os.path.relpath(path, ROOT)} exists (--force to recompute)")
    return load_result(DATA_METHOD, phase)["rows"]


def run_model(force):
    """CV rows of the model table, computed if missing (or forced).

    Phase 04 already scored the final model in CV, on the same folds: its row
    (f1_folds, f1_default_folds) is reused, only the variants are computed. The
    HAREHAB1 rows come from experiments/leakage.py, if it has been run with the
    setting of the final model.
    """
    phase = "ablation_model"
    final = load_best(FINAL_METHOD, FINAL_PHASE)
    base, final_fit = final["pipeline"], final["fit"]
    reuse = "f1_default_folds" in final  # saved by phase 04 since Oct 18
    grid = [] if reuse else [final_fit]
    if "n_trees" in final_fit:
        grid.append(dict(final_fit, n_trees=0))
    runs = [(FINAL_METHOD, grid)] if grid else []
    if GD_METHOD:
        runs.append((GD_METHOD, [load_best(GD_METHOD, GD_PHASE)["fit"]]))
    for method, method_grid in runs:
        path = result_path(method, phase)
        if force or not os.path.exists(path):
            run_grid(method, phase, {FINAL: base}, method_grid, K_FOLD, SEED)
        else:
            print(f"skip: {os.path.relpath(path, ROOT)} exists (--force to recompute)")
    final_rows = [final] if reuse else []
    if grid:
        final_rows += load_result(FINAL_METHOD, phase)["rows"]
    gd_row = load_result(GD_METHOD, phase)["rows"][0] if GD_METHOD else None
    extra = [
        (label, folds)
        for label, best, folds in variant_rows(FINAL_METHOD)
        if best["fit"] == final_fit  # the same setting as the final model
    ]
    return final_rows, gd_row, final_fit, extra


def _difference(entry, first):
    if first:
        return "reference", "–"
    return f"{entry['diff']:+.4f}", f"{entry['better']}/{entry['n_folds']}"


def markdown(data, model):
    """The tables in Markdown; data or model may be None."""
    lines = []
    if data is not None:
        lines += [
            f"Data preparation: {DATA_METHOD}, lambda in {DATA_LAMBDAS} (best for "
            "every variant), one step switched off at a time",
            "",
            "| Variant | F1 | Δ F1 | folds better | lambda | columns |",
            "|---|---|---|---|---|---|",
        ]
        for i, e in enumerate(data):
            diff, better = _difference(e, i == 0)
            lines.append(
                f"| {e['label']} | {e['f1']:.4f} ± {e['std']:.4f} | {diff} | {better} "
                f"| {e['lambda_']:g} | {e['n_features']} |"
            )
        lines.append("")
    if model is not None:
        lines += [
            f"Model: {FINAL_METHOD} with the setting of {FINAL_PHASE}, "
            "hyperparameters fixed",
            "",
            "| Variant | F1 | Δ F1 | folds better |",
            "|---|---|---|---|",
        ]
        for i, e in enumerate(model):
            diff, better = _difference(e, i == 0)
            lines.append(
                f"| {e['label']} | {e['f1']:.4f} ± {e['std']:.4f} "
                f"| {diff} | {better} |"
            )
        lines.append("")
    return "\n".join(lines)


def latex(data, model):
    """One LaTeX tabular with both tables; data or model may be None."""
    lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Variant & F1 & $\Delta$F1 & better folds \\",
    ]
    for title, table in (("Data preparation", data), ("Model", model)):
        if table is None:
            continue
        lines += [r"\hline", f"\\multicolumn{{4}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for i, e in enumerate(table):
            diff = "--" if i == 0 else f"${e['diff']:+.4f}$"
            better = "--" if i == 0 else f"{e['better']}/{e['n_folds']}"
            label = e["label"].replace("_", r"\_")
            lines.append(f"{label} & {e['f1']:.4f} & {diff} & {better} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("data", "model"),
        default=["data", "model"],
        help="which parts to compute (default: both); saved parts are always shown",
    )
    parser.add_argument(
        "--force", action="store_true", help="compute again the parts already done"
    )
    args = parser.parse_args()

    # --parts chooses what to compute; a table whose results are already saved is
    # shown anyway, so that results/ablation.md always has every table available
    data = model = None
    data_saved = os.path.exists(result_path(DATA_METHOD, "ablation_data"))
    model_saved = os.path.exists(result_path(GD_METHOD, "ablation_model"))
    if "data" in args.parts or data_saved:
        data = data_table(run_data(args.force and "data" in args.parts))
    if "model" in args.parts or model_saved:
        model = model_table(*run_model(args.force and "model" in args.parts))

    table = markdown(data, model)
    print("\n" + table)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outputs = {"ablation.md": table, "ablation.tex": latex(data, model)}
    for name, text in outputs.items():
        path = os.path.join(RESULTS_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"saved {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
