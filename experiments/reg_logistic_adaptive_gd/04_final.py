"""Phase 4, reg_logistic_adaptive_gd: final model with the best setting of a phase.

CV score of the setting, decision threshold chosen on the out-of-fold scores,
fit on the whole training set, test predictions in submissions/reg_logistic_adaptive_gd.csv.
Writes results/reg_logistic_adaptive_gd/04_final.json.

Run with the Run button on this file, or:  python experiments/reg_logistic_adaptive_gd/04_final.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import run_final  # noqa: E402

# ---- settings of the experiment ----
METHOD = "reg_logistic_adaptive_gd"
FROM_PHASE = "01_baseline"
# change the best setting by hand, e.g. dict(lambda_=1e-4)
PIPELINE_OVERRIDES = {}
FIT_OVERRIDES = {}
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
# -------------------------------------


def main():
    run_final(METHOD, FROM_PHASE, PIPELINE_OVERRIDES, FIT_OVERRIDES, K_FOLD, SEED)


if __name__ == "__main__":
    main()
