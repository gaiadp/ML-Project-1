"""Our own extensions of the basic methods in implementations.py
(e.g. class-weighted logistic regression, early stopping, decision threshold tuning).

fit / predict_scores / predict give the same interface to the 6 methods, with labels
always in {-1, 1} outside this module.
"""

import numpy as np

from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
    sigmoid,
)
from src.metrics import predict_labels

METHODS = (
    "mean_squared_error_gd",
    "mean_squared_error_sgd",
    "least_squares",
    "ridge_regression",
    "logistic_regression",
    "reg_logistic_regression",
)
LOGISTIC_METHODS = ("logistic_regression", "reg_logistic_regression")


def to_01(y):
    """Map labels from {-1, 1} to {0, 1} (float array)."""
    return (np.asarray(y) == 1).astype(float)


def to_pm1(y01):
    """Map labels from {0, 1} to {-1, 1} (int array)."""
    return np.where(np.asarray(y01) == 1, 1, -1)


def fit(
    method, y, tx, lambda_=0.0, gamma=0.1, max_iters=100, initial_w=None, seed=None
):
    """Train one of the 6 methods of implementations.py.

    Parameters a method does not use are ignored (e.g. lambda_ for least_squares).

    Args:
        method: name of the method, one of METHODS
        y: labels in {-1, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        lambda_: regularization parameter (ridge_regression, reg_logistic_regression)
        gamma: step size (iterative methods)
        max_iters: number of steps (iterative methods)
        initial_w: initial weights, shape=(D,); zeros if None
        seed: seed of np.random before mean_squared_error_sgd; None leaves it as is

    Returns:
        w: trained weights, shape=(D,)
        loss: training loss returned by the method
    """
    if initial_w is None:
        initial_w = np.zeros(tx.shape[1])

    if method == "least_squares":
        return least_squares(y, tx)
    if method == "ridge_regression":
        return ridge_regression(y, tx, lambda_)
    if method == "mean_squared_error_gd":
        return mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)
    if method == "mean_squared_error_sgd":
        if seed is not None:
            np.random.seed(seed)
        return mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)
    if method == "logistic_regression":
        return logistic_regression(to_01(y), tx, initial_w, max_iters, gamma)
    if method == "reg_logistic_regression":
        return reg_logistic_regression(
            to_01(y), tx, lambda_, initial_w, max_iters, gamma
        )
    raise ValueError(f"Unknown method {method!r}, expected one of {METHODS}")


def predict_scores(method, w, tx):
    """Continuous scores, higher = more likely MICHD.

    Probabilities sigma(tx @ w) for the logistic methods, tx @ w for the others.

    Args:
        method: name of the method used to train w
        w: weights, shape=(D,)
        tx: feature matrix, shape=(N, D)

    Returns:
        scores: shape=(N,)
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}, expected one of {METHODS}")
    z = tx @ w
    return sigmoid(z) if method in LOGISTIC_METHODS else z


def default_threshold(method):
    """Natural threshold on the scores: 0.5 for probabilities, 0 for labels in {-1, 1}."""
    return 0.5 if method in LOGISTIC_METHODS else 0.0


def predict(method, w, tx, threshold=None):
    """Predicted labels in {-1, 1}: 1 iff score >= threshold.

    Args:
        method: name of the method used to train w
        w: weights, shape=(D,)
        tx: feature matrix, shape=(N, D)
        threshold: decision threshold on the scores; default_threshold(method) if None

    Returns:
        labels in {-1, 1}, shape=(N,)
    """
    if threshold is None:
        threshold = default_threshold(method)
    return predict_labels(predict_scores(method, w, tx), threshold)
