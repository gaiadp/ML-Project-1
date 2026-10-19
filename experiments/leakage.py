"""HAREHAB1, the label leakage column: what it is worth, two extra submissions.

HAREHAB1 is asked only after a heart attack, so every row where it was asked is
MICHD (src/rules.py). The final model never sees it (model A, phase 04 of METHOD).
This script trains two variants of the final model, with the same setting (best row
of FROM_PHASE, the same folds and seed), and writes a submission for each:

    harehab1   model B: HAREHAB1 as a feature of the pipeline (keep_leakage=True).
               The model learns both sides of the leakage: an answer means a heart
               attack (100% MICHD), and in the states that use the module a NaN
               means no heart attack (4.1% MICHD instead of 8.8%).
    rule       model C: model A, then the known-positive rule: the rows where
               HAREHAB1 was asked are set to 1 after the threshold. The model is
               model A; the rule only corrects labels that are certain.

Model A and its results are not touched: the variants go to
results/<method>/extra/04_final_<variant>.json and submissions/<method>_<variant>.csv.
A variant whose result exists is not computed again (--force). experiments/ablation.py
adds both as rows of its model table. At the end, a table compares A, B and C on the
same folds.

Usage (from the repository root):
    python experiments/leakage.py                          # both variants of METHOD
    python experiments/leakage.py --variants rule          # only model C
    python experiments/leakage.py --method reg_logistic_newton
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import load_result, result_path, run_final  # noqa: E402

# ---- settings ----
METHOD = "logistic_boosting"  # the final model (model A = its phase 04)
FROM_PHASE = "01_baseline"  # the phase of the setting of model A (its FROM_PHASE)
K_FOLD = 5
SEED = 1
# -----------------

# variant -> (options of run_final, label in the tables, key of its CV F1 per fold)
VARIANTS = {
    "harehab1": (
        dict(pipeline_overrides={"keep_leakage": True}),
        "with HAREHAB1 as a feature (label leakage)",
        "f1_folds",
    ),
    "rule": (
        dict(known_positive_rule=True),
        "final + known-positive rule (HAREHAB1 asked -> 1)",
        "f1_rule_folds",
    ),
}


def variant_phase(variant):
    """Result name of a variant: results/<method>/extra/04_final_<variant>.json."""
    return f"extra/04_final_{variant}"


def variant_rows(method):
    """Saved variants of a method: list of (label, best row, F1 per fold)."""
    rows = []
    for variant, (_, label, key) in VARIANTS.items():
        path = result_path(method, variant_phase(variant))
        if os.path.exists(path):
            best = load_result(method, variant_phase(variant))["best"]
            if key in best:
                rows.append((label, best, best[key]))
    return rows


def summary(method):
    """Table of model A and its saved variants: F1, difference, folds better."""
    reference = load_result(method, "04_final")["best"]
    entries = [("A: final model, no HAREHAB1", reference, reference["f1_folds"])]
    entries += variant_rows(method)
    lines = [
        f"{method}, {K_FOLD}-fold CV (same folds)",
        f"{'model':52s} {'F1':>7s} {'diff':>8s}  better  submission",
    ]
    for label, best, folds in entries:
        diff = np.asarray(folds) - np.asarray(reference["f1_folds"])
        lines.append(
            f"{label:52s} {np.mean(folds):7.4f} {diff.mean():+8.4f}  "
            f"{np.sum(diff > 0)}/{len(diff)}     {best['submission']}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default=METHOD, help=f"default: {METHOD}")
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument(
        "--force", action="store_true", help="compute again the variants already done"
    )
    args = parser.parse_args()

    for variant in args.variants:
        path = result_path(args.method, variant_phase(variant))
        if os.path.exists(path) and not args.force:
            print(f"skip {variant}: {os.path.relpath(path, ROOT)} exists (--force)")
            continue
        options = VARIANTS[variant][0]
        print(f"\n=== {args.method}, variant {variant}")
        run_final(
            args.method,
            FROM_PHASE,
            k_fold=K_FOLD,
            seed=SEED,
            phase=variant_phase(variant),
            name=f"{args.method}_{variant}",
            **options,
        )
    print("\n" + summary(args.method))


if __name__ == "__main__":
    main()
