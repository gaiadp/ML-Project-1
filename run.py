"""Reproduces our AIcrowd submission, submissions/ensemble_newton_boosting.csv.

The submitted model is the ensemble of experiments/ensemble.py, a weighted mean of
the test probabilities of two final models, each fitted on the whole training set:

    p = a * p_newton + (1 - a) * p_boosting,    a = 0.35

1. reg_logistic_newton: regularized logistic regression solved with Newton's method,
   pipeline v1 with degree 2, log transforms and the products of the 20 columns most
   correlated with the label;
2. logistic_boosting (model A): the same logistic regression on pipeline v1, plus
   gradient boosting of trees on its residuals.
A row is predicted positive iff p >= threshold. The predictions go to
submissions/submission.csv.

HAREHAB1 is never used: it is asked only after a heart attack, so it reveals the label
instead of predicting it (LABEL_LEAKAGE, dropped by every pipeline config; see
src/rules.py). The known-positive rule of src/rules.py is not applied.

The constants are the settings saved in the result files named next to them.
experiments/ensemble.py chooses a and the threshold on the out-of-fold probabilities
of a 5-fold CV of each component; here the CV is skipped and the saved values are
used directly. Model A alone: experiments/run_model_a.py.

Running time: about 11.5 minutes (689 s, measured once on one laptop).

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

# component 1, weight a: results/reg_logistic_newton/extra/04_final_interactions.json
# (experiments/reg_logistic_newton/extra_final_interactions.py)
NEWTON = "reg_logistic_newton"
# "pipeline" in that JSON (keep_leakage=False is the default, the key did not exist
# yet when the JSON was saved)
NEWTON_PIPELINE = dict(
    V1_CONFIG,
    poly_degree=2,
    extra_transforms=("log",),
    interaction_top=20,
)
NEWTON_FIT = dict(lambda_=1e-6, max_iters=20)  # "fit" in that JSON
# checks only: n_features, train_loss, threshold_oof, test_positive_rate of that JSON
NEWTON_CHECKS = dict(
    n_features=619,
    train_loss=0.2175880390871175,
    threshold=0.19595568338104744,
    positive_rate=0.14704833651797877,
)

# component 2, weight 1 - a: model A, results/logistic_boosting/04_final.json
# (experiments/logistic_boosting/04_final.py)
BOOSTING = "logistic_boosting"
BOOSTING_PIPELINE = V1_CONFIG  # "pipeline" in that JSON, as for Newton
# "fit" in that JSON; the other tree parameters keep the defaults of
# src/extra_implementations/logistic_boosting.py: min_samples_leaf=100,
# reg_lambda=1.0, subsample=0.5, colsample=0.5, n_bins=32
BOOSTING_FIT = dict(
    lambda_=1e-4,
    max_iters=20,
    n_trees=400,
    max_depth=3,
    learning_rate=0.05,
)
# checks only: n_features, train_loss, threshold_oof, test_positive_rate of that JSON
BOOSTING_CHECKS = dict(
    n_features=336,
    train_loss=0.21625882467986896,
    threshold=0.2048126383378518,
    positive_rate=0.13538247744082502,
)

# seed passed to fit (trees: rows, columns, bins); SEED of experiments/ensemble.py
SEED = 1
# results/ensemble/ensemble.json, "final": weight a of Newton and threshold, chosen
# on the out-of-fold probabilities (experiments/ensemble.py, choose)
WEIGHT = 0.35
THRESHOLD = 0.20618416941358658
EXPECTED_POSITIVE_RATE = 0.1360498816043299  # "final", test_positive_rate
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


def component_probabilities(method, config, params, checks, data, start):
    """Test probabilities of one component, fitted on the whole training set.

    The same steps as component_scores in experiments/ensemble.py after its CV:
    fit_preprocess and fit on all the training rows, predict_scores on x_test.

    Args:
        method: one of src.models.ALL_METHODS
        config: pipeline config (src/pipeline.py)
        params: fit parameters
        checks: expected values from the result file, printed next to the
            computed ones
        data: (x_train, y_train, x_test, names) as returned by load_data
        start: time.time() at the start of the run

    Returns:
        probabilities of the positive class, shape=(N_test,)
    """
    x_train, y_train, x_test, names = data
    state = fit_preprocess(x_train, y_train, names, config)
    tx_train = apply_preprocess(x_train, names, state)
    log(
        start,
        f"{method}: {tx_train.shape[1]} features (expected {checks['n_features']}), "
        "fitting",
    )
    model, loss = fit(method, y_train, tx_train, seed=SEED, **params)
    del tx_train
    log(
        start,
        f"{method}: training log-loss {loss:.8f} "
        f"(expected {checks['train_loss']:.8f})",
    )
    proba = predict_scores(method, model, apply_preprocess(x_test, names, state))
    alone = np.mean(predict_labels(proba, checks["threshold"]) == 1)
    log(
        start,
        f"{method} alone, threshold {checks['threshold']:.6f}: {alone:.3%} predicted "
        f"positive (expected {checks['positive_rate']:.3%})",
    )
    return proba


def main():
    """Fit both components, average their test probabilities, write the CSV."""
    start = time.time()
    log(start, "loading the data (build/data.npz, built from dataset/ if missing)")
    x_train, x_test, y_train, _, test_ids, names = load_data()
    log(start, f"x_train {x_train.shape}, x_test {x_test.shape}")
    data = (x_train, y_train, x_test, names)

    p_newton = component_probabilities(
        NEWTON, NEWTON_PIPELINE, NEWTON_FIT, NEWTON_CHECKS, data, start
    )
    p_boosting = component_probabilities(
        BOOSTING, BOOSTING_PIPELINE, BOOSTING_FIT, BOOSTING_CHECKS, data, start
    )
    del data, x_train, x_test

    # same expression as experiments/ensemble.py
    p = WEIGHT * p_newton + (1 - WEIGHT) * p_boosting
    y_pred = predict_labels(p, THRESHOLD)
    gap = np.min(np.abs(p - THRESHOLD))
    log(start, f"ensemble a = {WEIGHT}, threshold {THRESHOLD}; min |p - t| = {gap:.2e}")

    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    create_csv_submission(test_ids, y_pred, SUBMISSION_PATH)
    labels = check_submission(SUBMISSION_PATH, test_ids)
    log(
        start,
        f"wrote {SUBMISSION_PATH}: {len(labels)} rows, {np.mean(labels == 1):.3%} "
        f"predicted positive (expected {EXPECTED_POSITIVE_RATE:.3%})",
    )


if __name__ == "__main__":
    main()
