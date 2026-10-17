"""Phase 3, mean_squared_error_sgd: interaction columns.

Starts from the best pipeline of 02_degree and adds the products z_i * z_j of
all the pairs of the INTERACTION_TOP columns most correlated with the label
(chosen on the training part of each fold), plus the pairs of INTERACTION_PAIRS
(output column names, e.g. ("_AGE80", "_BMI5")). k columns give k (k - 1) / 2
products. Writes results/mean_squared_error_sgd/03_interactions.json.

Run with the Run button on this file, or:  python experiments/mean_squared_error_sgd/03_interactions.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import load_best, param_grid, run_grid  # noqa: E402

# ---- settings of the experiment ----
METHOD = "mean_squared_error_sgd"
FROM_PHASE = "02_degree"
INTERACTION_TOP = (0, 10, 20, 30)
INTERACTION_PAIRS = ()
# one sample per step: max_iters is the number of samples seen
FIT_GRID = param_grid(gamma=[1e-4, 1e-3, 1e-2], max_iters=[50000])
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
# -------------------------------------


def main():
    base = load_best(METHOD, FROM_PHASE)["pipeline"]
    pipelines = {
        f"top {k}": dict(base, interaction_top=k, interaction_pairs=INTERACTION_PAIRS)
        for k in INTERACTION_TOP
    }
    run_grid(METHOD, "03_interactions", pipelines, FIT_GRID, K_FOLD, SEED)


if __name__ == "__main__":
    main()
