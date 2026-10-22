"""Phase 2, logistic_regression: degree of the polynomial expansion.

Starts from the best pipeline of 01_baseline, adds the powers z^2 .. z^degree
(pipeline step B11) and writes results/logistic_regression/02_degree.json.
Only the columns with more than p distinct values get z^p (reg_logistic_newton on v1:
334 columns become 390 with degree 2, 444 with degree 3, in its 02_degree.json).

Run with the Run button on this file, or:  python experiments/logistic_regression/02_degree.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import load_best, param_grid, run_grid  # noqa: E402

# ---- settings of the experiment ----
METHOD = "logistic_regression"
FROM_PHASE = "01_baseline"
DEGREES = (1, 2, 3)
# gamma < 2 / L, L = lambda_max(X^T X / N) / 4; add pos_weight=[...] or neg_per_pos=[...] for the imbalance
FIT_GRID = param_grid(gamma=[0.1, 0.3, 0.5], max_iters=[500])
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
# -------------------------------------


def main():
    base = load_best(METHOD, FROM_PHASE)["pipeline"]
    pipelines = {f"degree {d}": dict(base, poly_degree=d) for d in DEGREES}
    run_grid(METHOD, "02_degree", pipelines, FIT_GRID, K_FOLD, SEED)


if __name__ == "__main__":
    main()
