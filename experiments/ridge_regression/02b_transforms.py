"""Phase 2b, ridge_regression: sqrt / log of the continuous and count columns.

Starts from the best pipeline of 02_degree and adds sqrt(x - min) and / or
log(1 + x - min) of the continuous and count columns (pipeline step B11,
extra_transforms), next to the original column and its powers. Writes
results/ridge_regression/02b_transforms.json; its best row also contains the
"none" variant, so 03_interactions can start from 02b_transforms.

Before the CV, a quick check on the whole training set (CHECK = True, see
experiments.common.check_transforms): correlation of the label with x, sqrt and
log, and the partial correlation of the transforms given the powers of x.

Run with the Run button on this file, or:  python experiments/ridge_regression/02b_transforms.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from experiments.common import (  # noqa: E402
    check_transforms,
    load_best,
    param_grid,
    run_grid,
)

# ---- settings of the experiment ----
METHOD = "ridge_regression"
FROM_PHASE = "02_degree"
CHECK = True  # correlation check before the CV
DEGREES = None  # None = only the degree of 02_degree; e.g. (2, 3) to compare
TRANSFORMS = {
    "none": (),
    "sqrt": ("sqrt",),
    "log": ("log",),
    "sqrt+log": ("sqrt", "log"),
}
# one SVD per fold for the whole lambda grid
FIT_GRID = param_grid(lambda_=np.logspace(-7, -1, 7))
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
# -------------------------------------


def main():
    base = load_best(METHOD, FROM_PHASE)["pipeline"]
    degrees = DEGREES or (base["poly_degree"],)
    if CHECK:
        check_transforms(base, max(degrees))
    pipelines = {
        f"deg {d} {label}": dict(base, poly_degree=d, extra_transforms=t)
        for d in degrees
        for label, t in TRANSFORMS.items()
    }
    run_grid(METHOD, "02b_transforms", pipelines, FIT_GRID, K_FOLD, SEED)


if __name__ == "__main__":
    main()
