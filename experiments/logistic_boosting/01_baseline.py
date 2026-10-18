"""Phase 1, logistic_boosting: logistic regression + boosted trees on its residuals.

Writes results/logistic_boosting/01_baseline.json.

Run with the Run button on this file, or:  python experiments/logistic_boosting/01_baseline.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import param_grid, run_grid  # noqa: E402
from src.pipeline import V1_CONFIG  # noqa: E402

# ---- settings of the experiment ----
METHOD = "logistic_boosting"
# trees are slow: v1 only, the best pipeline of reg_logistic_newton
PIPELINES = {"v1": V1_CONFIG}
# lambda_, max_iters: the logistic part (best Newton setting of 01_baseline);
# the trees only correct it: shallow trees, small learning rate. The first grid
# (n_trees 100 / 200, depth 2 / 3, Oct 18) had its best at its edge (200, 3):
# extended to more and deeper trees. ~2-2.5 h in all (5 folds)
FIT_GRID = param_grid(
    lambda_=[1e-4],
    max_iters=[20],
    n_trees=[200, 400],
    max_depth=[3, 4],
    learning_rate=[0.05],
)
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
# -------------------------------------


def main():
    run_grid(METHOD, "01_baseline", PIPELINES, FIT_GRID, K_FOLD, SEED)


if __name__ == "__main__":
    main()
