"""Phase 1, least_squares: pipelines v0, full and v1, no expansion.

Writes results/least_squares/01_baseline.json.

Run with the Run button on this file, or:  python experiments/least_squares/01_baseline.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import param_grid, run_grid  # noqa: E402
from src.pipeline import DEFAULT_CONFIG, V0_CONFIG, V1_CONFIG  # noqa: E402

# ---- settings of the experiment ----
METHOD = "least_squares"
PIPELINES = {"v0": V0_CONFIG, "full": DEFAULT_CONFIG, "v1": V1_CONFIG}
# no hyperparameters
FIT_GRID = param_grid()
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
# -------------------------------------


def main():
    run_grid(METHOD, "01_baseline", PIPELINES, FIT_GRID, K_FOLD, SEED)


if __name__ == "__main__":
    main()
