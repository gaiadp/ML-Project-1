"""Quick check on the real data before a long run (~30 s, writes nothing).

tests/ uses small fake data; this script checks the same code on a random subset of
the real training set (N_ROWS rows), with the real column names:
1. every pipeline variant of the ablation (experiments/ablation.py) runs through
   fit_preprocess / apply_preprocess and gives finite values;
2. the final model and the models of the ablation (logistic regression + trees, no
   trees, Newton, plain GD) train and predict probabilities on the real pipeline,
   with few trees and steps so that it stays fast;
3. the HAREHAB1 variants of experiments/leakage.py: the pipeline with
   keep_leakage=True has the HAREHAB1 columns, the known-positive rule finds the
   rows where it was asked (all positive in the training rows).
The F1 printed is measured on the training rows: it only shows that the models learn,
it is not a score to report.

Run from the repository root:  python experiments/smoke_test.py
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.ablation import FINAL_METHOD, data_variants  # noqa: E402
from experiments.common import load_best  # noqa: E402
from src.data import load_data  # noqa: E402
from src.metrics import best_threshold  # noqa: E402
from src.models import fit, predict_scores  # noqa: E402
from src.pipeline import apply_preprocess, fit_preprocess  # noqa: E402
from src.rules import known_positives  # noqa: E402

N_ROWS = 20000  # random training rows used for the check
SEED = 1
# small versions of the models of the final run
MODELS = [
    ("logistic_boosting", dict(lambda_=1e-4, max_iters=20, n_trees=5, max_depth=3)),
    ("logistic_boosting", dict(lambda_=1e-4, max_iters=20, n_trees=0)),
    ("reg_logistic_newton", dict(lambda_=1e-4, max_iters=20)),
    ("reg_logistic_regression", dict(lambda_=1e-4, gamma=0.3, max_iters=50)),
]


def main():
    start = time.time()
    x, x_test, y, _, _, names = load_data()
    rows = np.random.default_rng(SEED).choice(len(y), size=N_ROWS, replace=False)
    x, y, x_test = x[rows], y[rows], x_test[:2000]
    print(f"data: {x.shape} training rows, {x_test.shape[0]} test rows")

    # the final pipeline: from phase 01 of the final model (phase 04 may not exist yet)
    base = load_best(FINAL_METHOD, "01_baseline")["pipeline"]
    print("1. pipeline variants of the ablation")
    for label, config in data_variants(base).items():
        state = fit_preprocess(x, y, names, config)
        tx_test = apply_preprocess(x_test, names, state)
        assert np.isfinite(tx_test).all(), label
        print(f"   ok  {label:42s} {tx_test.shape[1]} columns")

    print("2. models on the final pipeline")
    state = fit_preprocess(x, y, names, base)
    tx = apply_preprocess(x, names, state)
    tx_test = apply_preprocess(x_test, names, state)
    for method, params in MODELS:
        w, _ = fit(method, y, tx, seed=SEED, **params)
        _, f1 = best_threshold(predict_scores(method, w, tx), y)
        scores = predict_scores(method, w, tx_test)
        assert np.isfinite(scores).all(), method
        assert scores.min() >= 0 and scores.max() <= 1, method
        print(f"   ok  {method:24s} {params}  training F1 {f1:.3f}")
    print("3. HAREHAB1 variants")
    state = fit_preprocess(x, y, names, dict(base, keep_leakage=True))
    leak_names = [n for n in state["out_names"] if n.startswith("HAREHAB1")]
    assert leak_names, "keep_leakage did not keep HAREHAB1"
    known = known_positives(x, names)
    assert np.all(y[known] == 1), "a known positive is not positive"
    print(f"   ok  keep_leakage columns {leak_names}")
    print(f"   ok  rule: {known.sum()} known positives, all MICHD")
    print(f"smoke test passed in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
