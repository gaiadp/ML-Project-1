"""Phase 1, reg_logistic_unixgrad: pipelines full and v1, no expansion.

Writes results/reg_logistic_unixgrad/01_baseline.json.

Run with the Run button on this file, or:  python experiments/reg_logistic_unixgrad/01_baseline.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import param_grid, run_grid  # noqa: E402
from src.pipeline import DEFAULT_CONFIG, V1_CONFIG  # noqa: E402

# ---- settings of the experiment ----
METHOD = "reg_logistic_unixgrad"
PIPELINES = {"full": DEFAULT_CONFIG, "v1": V1_CONFIG}
# radius = bound on ||w*|| (the step adapts by itself); gamma is not used
FIT_GRID = param_grid(lambda_=[1e-4], max_iters=[200], radius=[5.0, 20.0])
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
# -------------------------------------


def main():
    run_grid(METHOD, "01_baseline", PIPELINES, FIT_GRID, K_FOLD, SEED)


if __name__ == "__main__":
    main()
