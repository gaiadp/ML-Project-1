"""Phase 1, gradient_boosting: pipeline v1 (the fastest, trees need no expansion).

Writes results/gradient_boosting/01_baseline.json.

Run with the Run button on this file, or:  python experiments/gradient_boosting/01_baseline.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import param_grid, run_grid  # noqa: E402
from src.pipeline import V1_CONFIG  # noqa: E402

# ---- settings of the experiment ----
METHOD = "gradient_boosting"
# trees are slow: v1 only (fewer columns); add "full": DEFAULT_CONFIG to compare
PIPELINES = {"v1": V1_CONFIG}
# ~6 min per fit with 200 trees on a CV fold
FIT_GRID = param_grid(n_trees=[200], learning_rate=[0.1], max_depth=[3, 5])
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
# -------------------------------------


def main():
    run_grid(METHOD, "01_baseline", PIPELINES, FIT_GRID, K_FOLD, SEED)


if __name__ == "__main__":
    main()
