"""Phase 2, mean_squared_error_sgd: degree of the polynomial expansion.

Starts from the best pipeline of 01_baseline, adds the powers z^2 .. z^degree
(pipeline step B11) and writes results/mean_squared_error_sgd/02_degree.json.
Only the columns with more than p distinct values get z^p: on the full pipeline
~317 columns become ~400 with degree 2, ~480 with degree 3.

Run with the Run button on this file, or:  python experiments/mean_squared_error_sgd/02_degree.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import load_best, param_grid, run_grid  # noqa: E402

# ---- settings of the experiment ----
METHOD = "mean_squared_error_sgd"
FROM_PHASE = "01_baseline"
DEGREES = (1, 2, 3)
# one sample per step: max_iters is the number of samples seen
FIT_GRID = param_grid(gamma=[1e-4, 1e-3, 1e-2], max_iters=[50000])
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
# -------------------------------------


def main():
    base = load_best(METHOD, FROM_PHASE)["pipeline"]
    pipelines = {f"degree {d}": dict(base, poly_degree=d) for d in DEGREES}
    run_grid(METHOD, "02_degree", pipelines, FIT_GRID, K_FOLD, SEED)


if __name__ == "__main__":
    main()
