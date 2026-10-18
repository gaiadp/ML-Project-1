"""Phase 1, random_forest: pipeline v1 (the fastest, trees need no expansion).

Writes results/random_forest/01_baseline.json.

Run with the Run button on this file, or:  python experiments/random_forest/01_baseline.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import param_grid, run_grid  # noqa: E402
from src.pipeline import V1_CONFIG  # noqa: E402

# ---- settings of the experiment ----
METHOD = "random_forest"
# trees are slow: v1 only (fewer columns); add "full": DEFAULT_CONFIG to compare
PIPELINES = {"v1": V1_CONFIG}
# ~3 min per fit with 100 trees on a CV fold
FIT_GRID = param_grid(n_trees=[100], max_depth=[10, 14], min_samples_leaf=[50])
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
# -------------------------------------


def main():
    run_grid(METHOD, "01_baseline", PIPELINES, FIT_GRID, K_FOLD, SEED)


if __name__ == "__main__":
    main()
