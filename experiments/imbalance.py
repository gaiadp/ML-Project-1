"""Class imbalance (8.8% positives): compare the ways to handle it, in 5-fold CV.

Run with the Run button on this file, or:  python experiments/imbalance.py
Prints a table and writes it to build/imbalance.txt (gitignored).

Every variant uses the same folds, the full preprocessing pipeline (DEFAULT_CONFIG,
fitted inside each fold) and regularized logistic regression with the same lambda,
gamma and max_iters:
    threshold 0.5          no handling of the imbalance
    tuned threshold        threshold that maximizes F1 on the training scores of the fold
    pos_weight = w         positives weighted w times in the loss, tuned threshold
    neg_per_pos = r        r random negatives per positive for training, tuned threshold
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# repository root on the import path, so that the Run button works
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.cross_validation import cross_validate, stratified_k_indices  # noqa: E402
from src.data import load_data  # noqa: E402
from src.metrics import f1_score, predict_labels  # noqa: E402
from src.pipeline import DEFAULT_CONFIG  # noqa: E402

K_FOLD = 5
SEED = 1
METHOD = "reg_logistic_regression"
# gamma < 2 / L, L = lambda_max(X^T X / N) / 4 ~ 3.7 on the preprocessed data;
# the loss changes by < 0.1% after 500 steps
FIT = dict(lambda_=1e-4, gamma=0.3, max_iters=500)
POS_WEIGHTS = (3.2, 10.3)  # sqrt(N_neg / N_pos) and N_neg / N_pos
NEG_PER_POS = (1, 3)
OUT_PATH = os.path.join(ROOT, "build", "imbalance.txt")


def f1_at_threshold(oof_scores, y, k_indices, threshold):
    """F1 of each fold with a fixed threshold on the out-of-fold scores.

    Args:
        oof_scores: validation score of each row, shape=(N,)
        y: labels in {-1, 1}, shape=(N,)
        k_indices: the folds of cross_validate
        threshold: fixed decision threshold

    Returns:
        f1 per fold, shape=(len(k_indices),)
    """
    return np.array(
        [f1_score(y[i], predict_labels(oof_scores[i], threshold)) for i in k_indices]
    )


def main():
    """Run every variant and print / save the table."""
    x, x_test, y, _, _, names = load_data()
    del x_test
    n_pos = int(np.sum(y == 1))
    print(f"x {x.shape}, {n_pos} positives ({n_pos / len(y):.2%})")
    k_indices = stratified_k_indices(y, K_FOLD, SEED)

    variants = [("tuned threshold", {})]
    variants += [(f"pos_weight = {w}", dict(pos_weight=w)) for w in POS_WEIGHTS]
    variants += [(f"neg_per_pos = {r}", dict(neg_per_pos=r)) for r in NEG_PER_POS]

    rows = []
    for name, options in variants:
        start = time.time()
        out = cross_validate(
            x, y, names, DEFAULT_CONFIG, METHOD, K_FOLD, SEED, **FIT, **options
        )
        seconds = time.time() - start
        if not options:
            # same models, threshold 0.5 instead of the tuned one
            f1 = f1_at_threshold(out["oof_scores"], y, k_indices, 0.5)
            rows.append(("threshold 0.5", f1, np.full(K_FOLD, 0.5), seconds))
        rows.append((name, out["f1"], out["threshold"], seconds))
        print(
            f"{name}: F1 {out['f1_mean']:.4f} +- {out['f1_std']:.4f} ({seconds:.0f} s)"
        )

    lines = [
        f"{METHOD}, {FIT}, {K_FOLD}-fold stratified CV (seed {SEED}), DEFAULT_CONFIG",
        f"{'variant':20s} {'F1 mean':>8s} {'F1 std':>7s} {'threshold':>9s} {'time':>6s}",
    ]
    for name, f1, threshold, seconds in rows:
        lines.append(
            f"{name:20s} {f1.mean():8.4f} {f1.std():7.4f} "
            f"{threshold.mean():9.3f} {seconds:5.0f}s"
        )
    table = "\n".join(lines)
    print("\n" + table)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(table + "\n")
    print(f"written to {OUT_PATH}")


if __name__ == "__main__":
    main()
