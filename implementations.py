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
    """Linear regression using gradient descent.

    Args:
        y: targets, shape=(N,)
        tx: feature matrix, shape=(N, D)
        initial_w: initial weights, shape=(D,)
        max_iters: number of GD steps
        gamma: step size

    Returns:
        w: weights after the last step, shape=(D,)
        loss: MSE loss (with the 1/2 factor) at the returned w
    """
    n = y.shape[0]
    w = initial_w
    for _ in range(max_iters):
        # Gradient of L(w) = 1/(2N) ||y - Xw||^2 is -X^T (y - Xw) / N
        gradient = -tx.T @ (y - tx @ w) / n
        w = w - gamma * gradient
    # Loss of the final w (with max_iters=0 this is the loss of initial_w)
    loss = np.mean((y - tx @ w) ** 2) / 2
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (mini-batch size 1).

    Args:
        y: targets, shape=(N,)
        tx: feature matrix, shape=(N, D)
        initial_w: initial weights, shape=(D,)
        max_iters: number of SGD steps (one sampled data point per step)
        gamma: step size

    Returns:
        w: weights after the last step, shape=(D,)
        loss: MSE loss (with the 1/2 factor) on the full dataset at the returned w
    """
    n = y.shape[0]
    w = initial_w
    for _ in range(max_iters):
        # Sample one data point uniformly at random
        i = np.random.randint(n)
        # Gradient of the single-sample loss 1/2 (y_i - x_i^T w)^2:
        # (y_i - x_i^T w) is a scalar, so multiply (not matmul) by the vector x_i
        stoch_gradient = -tx[i] * (y[i] - tx[i] @ w)
        w = w - gamma * stoch_gradient
    # Loss of the final w, evaluated on the whole dataset
    loss = np.mean((y - tx @ w) ** 2) / 2
    return w, loss


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
    # Cost: forming X^T X is O(N D^2) and dominates; solve only adds O(D^3).
    # A decomposition of X (e.g. SVD, X = U S V^T) costs several times more and only
    # pays off when reused for many lambdas, as in cross-validation
    # (see src/cross_validation.py: ridge_weights_svd). Working on X^T X squares the
    # condition number, but with lambda > 0 this only matters for tiny lambdas and
    # near-collinear features; for a single fit, solve is the cheaper choice.
    n = tx.shape[0]
    w = np.linalg.solve(tx.T @ tx + 2 * n * lambda_ * np.eye(tx.shape[1]), tx.T @ y)
    # Loss excludes the penalty term, per the grading convention
    loss = np.mean((y - tx @ w) ** 2) / 2
    return w, loss


# ---------------------------------------------------------------------------
# Logistic regression, adapted from the solution of lab ex05
# ---------------------------------------------------------------------------
# Same functions and names as in the lab, with these differences:
# - y, w and the gradient are 1D arrays of shape (N,) / (D,), not (N, 1) / (D, 1);
# - sigmoid and loss use np.logaddexp, so no overflow of exp and no log(0);
# - the step functions return only the new w: the loss is computed once, at the
#   LAST w, instead of at the old w before every update;
# - w is never updated in place (w = w - ..., not w -= ...), so the caller's
#   initial_w is never modified.


def sigmoid(t):
    """Apply the sigmoid function sigma(t) = 1 / (1 + exp(-t)) elementwise.

    Numerically stable form: log(1 + exp(-t)) = logaddexp(0, -t), hence
    sigma(t) = exp(-logaddexp(0, -t)). The exponent is always <= 0, so exp never
    overflows, while the direct formula overflows in exp(-t) for very negative t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array of the same shape, with values in [0, 1]
    """
    return np.exp(-np.logaddexp(0, -t))


def calculate_loss(y, tx, w):
    """Compute the logistic loss (mean negative log-likelihood), y in {0, 1}.

    With z = tx @ w and sigma(z) the predicted probability of class 1:
        -[y log sigma(z) + (1 - y) log(1 - sigma(z))] = log(1 + exp(z)) - y z,
    evaluated with np.logaddexp(0, z) = log(1 + exp(z)), which never overflows and
    never takes log(0) (the lab formula does when sigma(z) rounds to 0 or 1).

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        w: weights, shape=(D,)

    Returns:
        loss: non-negative scalar (0-dimensional numpy float)
    """
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def calculate_gradient(y, tx, w):
    """Compute the gradient of the logistic loss: tx^T (sigma(tx @ w) - y) / N.

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        w: weights, shape=(D,)

    Returns:
        gradient: shape=(D,)
    """
    pred = sigmoid(tx @ w)
    return tx.T @ (pred - y) / y.shape[0]


def learning_by_gradient_descent(y, tx, w, gamma):
    """Do one step of gradient descent on the logistic loss and return the new w.

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        w: current weights, shape=(D,), not modified
        gamma: step size

    Returns:
        w: updated weights, shape=(D,) (a new array)
    """
    return w - gamma * calculate_gradient(y, tx, w)


def penalized_logistic_regression(y, tx, w, lambda_):
    """Return the gradient of the penalized objective L(w) + lambda_ * ||w||^2.

    The derivative of lambda_ * w^T w is 2 * lambda_ * w. All weights, bias
    included, are penalized, as in the lab and in the project description. The
    loss returned by reg_logistic_regression is calculate_loss, without penalty.

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        w: weights, shape=(D,)
        lambda_: regularization parameter

    Returns:
        gradient: shape=(D,)
    """
    return calculate_gradient(y, tx, w) + 2 * lambda_ * w


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """Do one step of gradient descent on the penalized objective, return the new w.

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        w: current weights, shape=(D,), not modified
        gamma: step size
        lambda_: regularization parameter

    Returns:
        w: updated weights, shape=(D,) (a new array)
    """
    return w - gamma * penalized_logistic_regression(y, tx, w, lambda_)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        initial_w: initial weights, shape=(D,), not modified
        max_iters: number of GD steps
        gamma: step size

    Returns:
        w: weights after the last step, shape=(D,)
        loss: logistic loss at the returned w (max_iters=0: initial_w and its loss)
    """
    # np.array makes a copy: even with max_iters=0 the returned w is a new array
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = learning_by_gradient_descent(y, tx, w, gamma)
    loss = calculate_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using GD (y in {0, 1}, penalty lambda_ ||w||^2).

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        lambda_: regularization parameter
        initial_w: initial weights, shape=(D,), not modified
        max_iters: number of GD steps
        gamma: step size

    Returns:
        w: weights after the last step, shape=(D,)
        loss: logistic loss at the returned w, WITHOUT the penalty term
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    loss = calculate_loss(y, tx, w)
    return w, loss
