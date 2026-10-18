"""Objective shared by the extra optimizers: regularized logistic regression.

The same objective as implementations.reg_logistic_regression, so that all the
optimizers can be compared at equal loss:

    f(w) = mean_n [log(1 + exp(z_n)) - y_n z_n] + lambda_ * ||w||^2,   z = tx @ w

with labels y in {0, 1} and the penalty on every weight, the bias included. As in
implementations.py, the loss returned by the methods is the first term only, without
the penalty.
"""

import numpy as np

from implementations import calculate_loss, penalized_logistic_regression, sigmoid

# rows per block in hessian: a block of 50k rows x 400 columns is ~160 MB
_HESSIAN_BLOCK_ROWS = 50000


def penalized_loss(y, tx, w, lambda_):
    """Objective f(w): logistic loss plus lambda_ * ||w||^2.

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        w: weights, shape=(D,)
        lambda_: regularization parameter

    Returns:
        scalar
    """
    return calculate_loss(y, tx, w) + lambda_ * (w @ w)


def penalized_gradient(y, tx, w, lambda_):
    """Gradient of f: tx^T (sigmoid(tx @ w) - y) / N + 2 lambda_ w, shape=(D,)."""
    return penalized_logistic_regression(y, tx, w, lambda_)


def diag_hessian(tx, w, lambda_):
    """Diagonal of the Hessian of f, shape=(D,).

        H_jj = sum_n s_n x_nj^2 / N + 2 lambda_,   s = sigmoid(z) (1 - sigmoid(z))

    It builds tx**2 (a copy of tx): call it on a minibatch, not on all the rows.
    """
    p = sigmoid(tx @ w)
    return (p * (1 - p)) @ tx**2 / tx.shape[0] + 2 * lambda_


def hessian(tx, w, lambda_):
    """Hessian of f: tx^T diag(s) tx / N + 2 lambda_ I, shape=(D, D).

    Computed by blocks of rows, so that the weighted copy of tx never has more than
    _HESSIAN_BLOCK_ROWS rows (all the training rows at once would be ~0.9 GB).
    """
    n, d = tx.shape
    h = 2 * lambda_ * np.eye(d)
    for start in range(0, n, _HESSIAN_BLOCK_ROWS):
        block = tx[start : start + _HESSIAN_BLOCK_ROWS]
        p = sigmoid(block @ w)
        h += block.T @ (block * (p * (1 - p))[:, None]) / n
    return h
