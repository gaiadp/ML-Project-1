"""Our own extensions of the basic methods in implementations.py.

fit / predict_scores / predict give the same interface to the 6 methods, with labels
always in {-1, 1} outside this module. Class imbalance (8.8% positives) is handled by
two options of fit: class weights in the logistic loss (pos_weight) and undersampling
of the negatives (neg_per_pos); the decision threshold is chosen outside, on the scores
(src.metrics.best_threshold).
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


def class_weights(y01, pos_weight):
    """Weight of each sample: pos_weight for the positives, 1 for the negatives.

    Args:
        y01: labels in {0, 1}, shape=(N,)
        pos_weight: weight of a positive sample relative to a negative one

    Returns:
        weights: shape=(N,)
    """
    return np.where(np.asarray(y01) == 1, float(pos_weight), 1.0)


def weighted_logistic_loss(y01, tx, w, weights):
    """Weighted mean of the logistic loss, without penalty.

        L(w) = sum_n s_n [log(1 + exp(z_n)) - y_n z_n] / sum_n s_n,  z = tx @ w

    Args:
        y01: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        w: weights, shape=(D,)
        weights: sample weights s, shape=(N,)

    Returns:
        loss: scalar
    """
    z = tx @ w
    return np.sum(weights * (np.logaddexp(0, z) - y01 * z)) / np.sum(weights)


def weighted_logistic_gradient(y01, tx, w, weights):
    """Gradient of weighted_logistic_loss: tx^T (s * (sigmoid(z) - y)) / sum_n s_n.

    Args:
        y01: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        w: weights, shape=(D,)
        weights: sample weights s, shape=(N,)

    Returns:
        gradient: shape=(D,)
    """
    return tx.T @ (weights * (sigmoid(tx @ w) - y01)) / np.sum(weights)


def weighted_logistic_regression(
    y01, tx, lambda_, initial_w, max_iters, gamma, pos_weight
):
    """Class-weighted logistic regression with penalty lambda_ * ||w||^2, by GD.

    Same as implementations.reg_logistic_regression when pos_weight = 1. The mean
    is normalized by the sum of the weights, so the gradient keeps the same scale
    (and the same usable gamma) for every pos_weight.

    Args:
        y01: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        lambda_: regularization parameter of the penalty lambda_ * ||w||^2
        initial_w: initial weights, shape=(D,)
        max_iters: number of GD steps
        gamma: step size
        pos_weight: weight of a positive sample relative to a negative one

    Returns:
        w: weights after the last step, shape=(D,)
        loss: weighted logistic loss at the returned w, without the penalty term
    """
    weights = class_weights(y01, pos_weight)
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        gradient = weighted_logistic_gradient(y01, tx, w, weights) + 2 * lambda_ * w
        w = w - gamma * gradient
    return w, weighted_logistic_loss(y01, tx, w, weights)


def undersample_indices(y, neg_per_pos, seed=None):
    """Rows of a training set with all the positives and a random subset of negatives.

    Args:
        y: labels, shape=(N,), positive class = 1 ({-1, 1} or {0, 1})
        neg_per_pos: number of negatives kept per positive (all of them if fewer)
        seed: seed of the random choice; None = not reproducible

    Returns:
        sorted 1D index array
    """
    is_pos = np.asarray(y) == 1
    pos, neg = np.flatnonzero(is_pos), np.flatnonzero(~is_pos)
    n_neg = min(len(neg), int(round(neg_per_pos * len(pos))))
    rng = np.random.default_rng(seed)
    kept_neg = rng.choice(neg, size=n_neg, replace=False)
    return np.sort(np.concatenate([pos, kept_neg]))


def fit(
    method,
    y,
    tx,
    lambda_=0.0,
    gamma=0.1,
    max_iters=100,
    initial_w=None,
    seed=None,
    pos_weight=None,
    neg_per_pos=None,
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
        seed: seed of np.random before mean_squared_error_sgd and of the
            undersampling; None leaves them random
        pos_weight: logistic methods only, weight of the positive samples in the
            loss (weighted_logistic_regression); None = unweighted
        neg_per_pos: if given, train on all the positives and neg_per_pos random
            negatives per positive (undersample_indices); None = all the rows

    Returns:
        w: trained weights, shape=(D,)
        loss: training loss returned by the method (on the rows used for training)
    """
    if initial_w is None:
        initial_w = np.zeros(tx.shape[1])
    if neg_per_pos is not None:
        rows = undersample_indices(y, neg_per_pos, seed)
        y, tx = np.asarray(y)[rows], tx[rows]

    if pos_weight is not None:
        if method not in LOGISTIC_METHODS:
            raise ValueError(f"pos_weight needs one of {LOGISTIC_METHODS}")
        # logistic_regression is the unpenalized case
        lam = lambda_ if method == "reg_logistic_regression" else 0.0
        return weighted_logistic_regression(
            to_01(y), tx, lam, initial_w, max_iters, gamma, pos_weight
        )

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
