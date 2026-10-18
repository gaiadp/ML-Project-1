"""Phase 1, reg_logistic_sophia: pipelines full and v1, no expansion.

Writes results/reg_logistic_sophia/01_baseline.json.

Run with the Run button on this file, or:  python experiments/reg_logistic_sophia/01_baseline.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import param_grid, run_grid  # noqa: E402
from src.pipeline import DEFAULT_CONFIG, V1_CONFIG  # noqa: E402

# ---- settings of the experiment ----
METHOD = "reg_logistic_sophia"
PIPELINES = {"full": DEFAULT_CONFIG, "v1": V1_CONFIG}
# gamma = largest move of a coordinate per step; 3000 steps x 2048 rows ~ 23 epochs
FIT_GRID = param_grid(
    lambda_=[1e-4], gamma=[0.02, 0.05], max_iters=[3000], batch_size=[2048]
)
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
# -------------------------------------


def main():
    run_grid(METHOD, "01_baseline", PIPELINES, FIT_GRID, K_FOLD, SEED)


if __name__ == "__main__":
    main()
