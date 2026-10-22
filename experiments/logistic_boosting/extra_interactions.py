"""Extra experiment, logistic_boosting: interaction columns in the linear part.

The trees on the residuals already learn interactions, but as steps and spending
trees on them. Here the products z_i * z_j of the INTERACTION_TOP columns most
correlated with the label (chosen on the training part of each fold) are added to
the pipeline of 01_baseline, so the logistic part fits them exactly and the trees
are left for the rest. No polynomial degree or sqrt / log: trees are invariant to
monotone transforms of a column, and for reg_logistic_newton they added nothing
(02_degree, 02b_transforms).

Two steps, on the same folds as 01_baseline:
1. CV grid -> results/logistic_boosting/extra/03_interactions.json; the rows with
   interaction_top = 0 are already in 01_baseline (same folds and seed), so only
   the expanded pipelines are run and then compared with them fold by fold;
2. final model with the best row of step 1 ->
   results/logistic_boosting/extra/04_final_interactions.json and
   submissions/logistic_boosting_interactions.csv.
04_final.json and submissions/logistic_boosting.csv are not touched. Not a phase:
run_phases.py does not run it. A step whose result exists is skipped (--force).
~45-60 min per fit (5 folds) and as much for the final model: ~3-3.5 h in all.

Run with the Run button on this file, or:
    python experiments/logistic_boosting/extra_interactions.py
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import (  # noqa: E402
    load_best,
    load_result,
    param_grid,
    result_path,
    run_final,
    run_grid,
)

# ---- settings of the experiment ----
METHOD = "logistic_boosting"
FROM_PHASE = "01_baseline"
INTERACTION_TOP = (20,)  # the best for reg_logistic_newton (03_interactions)
INTERACTION_PAIRS = ()
# the best setting of 01_baseline, and fewer trees: with the interactions in the
# linear part the trees may need less depth / fewer rounds
FIT_GRID = param_grid(
    lambda_=[1e-4],
    max_iters=[20],
    n_trees=[200, 400],
    max_depth=[3],
    learning_rate=[0.05],
)
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
GRID_PHASE = "extra/03_interactions"
FINAL_PHASE = "extra/04_final_interactions"
# -------------------------------------


def compare(method, grid_phase, reference_phase):
    """Each row of grid_phase against the row of reference_phase with the same fit."""
    reference = {
        _key(r["fit"]): r for r in load_result(method, reference_phase)["rows"]
    }
    print(f"\nagainst {reference_phase} (no interactions), same folds:")
    for row in load_result(method, grid_phase)["rows"]:
        ref = reference.get(_key(row["fit"]))
        if ref is None:
            continue
        diff = np.asarray(row["f1_folds"]) - np.asarray(ref["f1_folds"])
        fit_text = ", ".join(f"{k}={v:g}" for k, v in row["fit"].items())
        print(
            f"{row['label']:10s} {fit_text:70s} F1 {row['f1_mean']:.4f} vs "
            f"{ref['f1_mean']:.4f}  diff {diff.mean():+.4f}  "
            f"better {np.sum(diff > 0)}/{len(diff)}"
        )


def _key(fit_params):
    return tuple(sorted(fit_params.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force", action="store_true", help="compute again the steps already done"
    )
    args = parser.parse_args()

    if os.path.exists(result_path(METHOD, GRID_PHASE)) and not args.force:
        print(f"skip the grid: {GRID_PHASE}.json exists (--force)")
    else:
        base = load_best(METHOD, FROM_PHASE)["pipeline"]
        pipelines = {
            f"top {k}": dict(
                base, interaction_top=k, interaction_pairs=INTERACTION_PAIRS
            )
            for k in INTERACTION_TOP
        }
        run_grid(METHOD, GRID_PHASE, pipelines, FIT_GRID, K_FOLD, SEED)
    compare(METHOD, GRID_PHASE, FROM_PHASE)

    if os.path.exists(result_path(METHOD, FINAL_PHASE)) and not args.force:
        print(f"skip the final model: {FINAL_PHASE}.json exists (--force)")
        return
    print()
    run_final(
        METHOD,
        GRID_PHASE,
        k_fold=K_FOLD,
        seed=SEED,
        phase=FINAL_PHASE,
        name=f"{METHOD}_interactions",
    )


if __name__ == "__main__":
    main()
