"""Reproduces our best submission on AIcrowd.

Usage:
    python run.py   (or the Run button on this file)

Loads the data (build/data.npz cache, see src/data.py), splits the training set into
training / validation (stratified, fixed seed), fits the preprocessing on the training
part, trains the model, chooses the decision threshold that maximizes F1 on the
validation part and writes the test predictions to submissions/submission.csv.
"""

import csv
import os

import numpy as np

from helpers import create_csv_submission
from src.cross_validation import stratified_split
from src.data import ROOT, load_data
from src.metrics import accuracy, best_threshold, f1_score, precision, recall
from src.metrics import predict_labels
from src.models import default_threshold, fit, predict_scores
from src.pipeline import apply_preprocess, fit_preprocess

SEED = 1
VAL_RATIO = 0.2
MAX_MISSING_FRAC = 0.5
METHOD = "ridge_regression"
LAMBDA = 1e-4
SUBMISSION_PATH = os.path.join(ROOT, "submissions", "submission.csv")


def report(name, y_true, y_pred):
    """Print F1, precision, recall, accuracy and fraction of predicted positives."""
    print(
        f"{name:28s} F1={f1_score(y_true, y_pred):.4f}  "
        f"P={precision(y_true, y_pred):.4f}  R={recall(y_true, y_pred):.4f}  "
        f"acc={accuracy(y_true, y_pred):.4f}  pred.pos={np.mean(y_pred == 1):.2%}"
    )


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
    x_train, x_test, y_train, train_ids, test_ids, names = load_data()
    print(f"x_train {x_train.shape}, x_test {x_test.shape}")

    tr_idx, va_idx = stratified_split(y_train, VAL_RATIO, SEED)
    x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
    x_va, y_va = x_train[va_idx], y_train[va_idx]
    del x_train
    print(
        f"split: train {len(tr_idx)} ({np.mean(y_tr == 1):.2%} pos), "
        f"val {len(va_idx)} ({np.mean(y_va == 1):.2%} pos)"
    )

    params = fit_preprocess(x_tr, names, MAX_MISSING_FRAC)
    tx_tr = apply_preprocess(x_tr, names, params)
    del x_tr
    tx_va = apply_preprocess(x_va, names, params)
    del x_va
    print(f"kept {params['keep'].sum()} of {len(names)} columns, tx {tx_tr.shape}")

    w, loss = fit(METHOD, y_tr, tx_tr, lambda_=LAMBDA)
    print(f"{METHOD} (lambda={LAMBDA}): training loss {loss:.4f}")

    scores_tr = predict_scores(METHOD, w, tx_tr)
    scores_va = predict_scores(METHOD, w, tx_va)
    threshold, _ = best_threshold(scores_va, y_va)
    t0 = default_threshold(METHOD)
    print(f"threshold chosen on validation: {threshold:.4f} (default {t0})")
    report("val, default threshold", y_va, predict_labels(scores_va, t0))
    report("val, chosen threshold", y_va, predict_labels(scores_va, threshold))
    report("train, chosen threshold", y_tr, predict_labels(scores_tr, threshold))

    tx_te = apply_preprocess(x_test, names, params)
    del x_test
    y_te_pred = predict_labels(predict_scores(METHOD, w, tx_te), threshold)

    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    create_csv_submission(test_ids, y_te_pred, SUBMISSION_PATH)
    labels = check_submission(SUBMISSION_PATH, test_ids)
    print(
        f"wrote {SUBMISSION_PATH}: {len(labels)} rows, "
        f"{np.mean(labels == 1):.2%} predicted positive"
    )


if __name__ == "__main__":
    main()
