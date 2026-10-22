"""Reproduces our best AIcrowd submission, submissions/logistic_boosting.csv (model A).

Model: regularized logistic regression solved with Newton's method, plus gradient
boosting of trees on its residuals (logistic_boosting), on pipeline v1, fitted on the
whole training set. The test predictions go to submissions/submission.csv.

HAREHAB1 is never used: it is asked only after a heart attack, so it reveals the label
instead of predicting it (LABEL_LEAKAGE, dropped by every pipeline config; see
src/rules.py). The known-positive rule of src/rules.py is not applied.

The constants are the setting saved in results/logistic_boosting/04_final.json by
experiments/logistic_boosting/04_final.py. That script also runs a 5-fold CV to choose
the decision threshold; here the CV is skipped and its threshold is used directly.

Running time: to be measured.

Usage:
    python run.py
"""

import csv
import os
import time

import numpy as np

from helpers import create_csv_submission
from src.data import ROOT, load_data
from src.metrics import predict_labels
from src.models import fit, predict_scores
from src.pipeline import V1_CONFIG, apply_preprocess, fit_preprocess

# method of src/models.py; "method" in 04_final.json
METHOD = "logistic_boosting"
# src/pipeline.py; equal to "pipeline" in 04_final.json (keep_leakage=False is the
# default, the key did not exist yet when the JSON was saved)
PIPELINE = V1_CONFIG
# fit parameters: "fit" in 04_final.json, best row of 01_baseline.json
LAMBDA = 1e-4  # L2 penalty of the logistic part
MAX_ITERS = 20  # Newton steps of the logistic part
N_TREES = 400  # boosting rounds
MAX_DEPTH = 3  # depth of every tree
LEARNING_RATE = 0.05  # shrinkage of every tree
# the other tree parameters keep the defaults of
# src/extra_implementations/logistic_boosting.py: min_samples_leaf=100,
# reg_lambda=1.0, subsample=0.5, colsample=0.5, n_bins=32
# seed of the rows, columns and bins of the trees; "seed" in 04_final.json
SEED = 1
# "threshold_oof" in 04_final.json: maximizes F1 on the out-of-fold scores of the
# 5-fold CV (seed 1)
THRESHOLD = 0.2048126383378518
SUBMISSION_PATH = os.path.join(ROOT, "submissions", "submission.csv")


def log(start, text):
    """Print text with the seconds elapsed since start."""
    print(f"[{time.time() - start:8.1f} s] {text}", flush=True)


def check_submission(path, test_ids):
    """Read the CSV back and check header, Ids (same order as x_test) and labels."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["Id", "Prediction"], rows[0]
    ids = np.array([int(r[0]) for r in rows[1:]])
    labels = np.array([int(r[1]) for r in rows[1:]])
    assert np.array_equal(ids, test_ids), "Ids differ from test_ids"
    assert set(np.unique(labels)) <= {-1, 1}, np.unique(labels)
    return labels


def main():
    """Preprocess, fit on the whole training set, predict x_test, write the CSV."""
    start = time.time()
    log(start, "loading the data (build/data.npz, built from dataset/ if missing)")
    x_train, x_test, y_train, _, test_ids, names = load_data()
    log(start, f"x_train {x_train.shape}, x_test {x_test.shape}")

    state = fit_preprocess(x_train, y_train, names, PIPELINE)
    tx_train = apply_preprocess(x_train, names, state)
    del x_train
    # 04_final.json: n_features 336
    log(start, f"pipeline v1: {tx_train.shape[1]} features (expected 336)")

    log(start, f"fitting {METHOD} ({N_TREES} trees)")
    model, loss = fit(
        METHOD,
        y_train,
        tx_train,
        lambda_=LAMBDA,
        max_iters=MAX_ITERS,
        seed=SEED,
        n_trees=N_TREES,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
    )
    del tx_train
    # 04_final.json: train_loss 0.21625882467986896
    log(start, f"training log-loss {loss:.8f} (expected 0.21625882)")

    tx_test = apply_preprocess(x_test, names, state)
    del x_test
    y_pred = predict_labels(predict_scores(METHOD, model, tx_test), THRESHOLD)
    log(start, f"test predictions with threshold {THRESHOLD}")

    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    create_csv_submission(test_ids, y_pred, SUBMISSION_PATH)
    labels = check_submission(SUBMISSION_PATH, test_ids)
    # 04_final.json: test_positive_rate 0.13538247744082502
    log(
        start,
        f"wrote {SUBMISSION_PATH}: {len(labels)} rows, "
        f"{np.mean(labels == 1):.3%} predicted positive (expected 13.538%)",
    )


if __name__ == "__main__":
    main()
