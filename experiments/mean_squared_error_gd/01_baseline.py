"""Phase 1, mean_squared_error_gd: pipelines v0, full and v1, no expansion.

Writes results/mean_squared_error_gd/01_baseline.json.

Run with the Run button on this file, or:  python experiments/mean_squared_error_gd/01_baseline.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import param_grid, run_grid  # noqa: E402
from src.pipeline import DEFAULT_CONFIG, V0_CONFIG, V1_CONFIG  # noqa: E402

# ---- settings of the experiment ----
METHOD = "mean_squared_error_gd"
PIPELINES = {"v0": V0_CONFIG, "full": DEFAULT_CONFIG, "v1": V1_CONFIG}
# gamma < 2 / lambda_max(X^T X / N), smaller with more features
FIT_GRID = param_grid(gamma=[0.01, 0.03, 0.1], max_iters=[300])
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
# -------------------------------------


def main():
    run_grid(METHOD, "01_baseline", PIPELINES, FIT_GRID, K_FOLD, SEED)


if __name__ == "__main__":
    main()
