"""Phase 1, logistic_regression: pipeline v0 vs full pipeline, no expansion.

Writes results/logistic_regression/01_baseline.json.

Run with the Run button on this file, or:  python experiments/logistic_regression/01_baseline.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import param_grid, run_grid  # noqa: E402
from src.pipeline import DEFAULT_CONFIG, V0_CONFIG  # noqa: E402

# ---- settings of the experiment ----
METHOD = "logistic_regression"
PIPELINES = {"v0": V0_CONFIG, "full": DEFAULT_CONFIG}
# gamma < 2 / L, L = lambda_max(X^T X / N) / 4; add pos_weight=[...] or neg_per_pos=[...] for the imbalance
FIT_GRID = param_grid(gamma=[0.1, 0.3, 0.5], max_iters=[500])
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
# -------------------------------------


def main():
    run_grid(METHOD, "01_baseline", PIPELINES, FIT_GRID, K_FOLD, SEED)


if __name__ == "__main__":
    main()
