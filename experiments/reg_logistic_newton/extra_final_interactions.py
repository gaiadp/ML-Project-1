"""Extra final model, reg_logistic_newton: the best setting of 03_interactions.

The same as 04_final.py, but with the expanded pipeline of phases 2-3 (degree,
sqrt / log transforms, interaction columns) instead of the plain v1 pipeline of
01_baseline. 04_final.json and submissions/reg_logistic_newton.csv are not touched:
writes results/reg_logistic_newton/extra/04_final_interactions.json and
submissions/reg_logistic_newton_interactions.csv. Not a phase: run_phases.py does
not run it.

Run with the Run button on this file, or:
    python experiments/reg_logistic_newton/extra_final_interactions.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import run_final  # noqa: E402

# ---- settings of the experiment ----
METHOD = "reg_logistic_newton"
FROM_PHASE = "03_interactions"  # its best row: degree 2, log, top 20 interactions
# change the best setting by hand, e.g. dict(lambda_=1e-4)
PIPELINE_OVERRIDES = {}
FIT_OVERRIDES = {}
K_FOLD = 5
SEED = 1  # folds, SGD and undersampling: same value in every phase
# -------------------------------------


def main():
    run_final(
        METHOD,
        FROM_PHASE,
        PIPELINE_OVERRIDES,
        FIT_OVERRIDES,
        K_FOLD,
        SEED,
        phase="extra/04_final_interactions",
        name=f"{METHOD}_interactions",
    )


if __name__ == "__main__":
    main()
