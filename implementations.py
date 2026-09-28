"""The 6 required ML methods (Step 2 of the project description).

Graders import these functions directly from this file, so:
- do NOT rename this file or change the function signatures;
- every function returns (w, loss): the LAST weight vector and its loss;
- losses of regularized methods must NOT include the penalty term;
- MSE uses the 1/2 factor; vectors are 1D arrays of shape (D,);
- only NumPy is allowed (numpy.linalg.lstsq is forbidden).
"""

import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    raise NotImplementedError


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1)."""
    raise NotImplementedError


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    # Closed-form solution of min ||y - Xw||^2: solve X^T X w = X^T y.
    # X^T X can be rank-deficient or ill-conditioned (e.g. collinear/redundant
    # features), so np.linalg.solve is not safe here: it assumes a
    # nonsingular matrix and can blow up or return garbage otherwise.
    # X^T X is symmetric PSD, so instead we eigendecompose it (np.linalg.eigh,
    # the symmetric-matrix eigensolver) and build the Moore-Penrose
    # pseudo-inverse by inverting only the eigenvalues that are numerically
    # nonzero, which gives the minimum-norm solution even when rank-deficient.
    a = tx.T @ tx
    eigvals, eigvecs = np.linalg.eigh(a)
    tol = eigvals.max() * a.shape[0] * np.finfo(eigvals.dtype).eps
    inv_eigvals = np.where(eigvals > tol, 1.0 / eigvals, 0.0)
    a_pinv = (eigvecs * inv_eigvals) @ eigvecs.T
    w = a_pinv @ (tx.T @ y)
    # MSE loss with the 1/2 factor, as required by the grading convention
    loss = np.mean((y - tx @ w) ** 2) / 2
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    # min (L(w) + Omega(w))
    # solve (X^T * X + 2 * N * lambda * I) * w = X^T * y
    # The resulting matrix is never singular because the regularization term is positive definite,
    # so we can use np.linalg.solve to solve the system of equations.
    n = tx.shape[0]
    w = np.linalg.solve(tx.T @ tx + 2 * n * lambda_ * np.eye(tx.shape[1]), tx.T @ y)
    # Loss excludes the penalty term, per the grading convention
    loss = np.mean((y - tx @ w) ** 2) / 2
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1})."""
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using GD (y in {0, 1}, penalty lambda_ * ||w||^2)."""
    raise NotImplementedError
