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


def sigmoid(t):
    """Apply the sigmoid function elementwise.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array of the same shape, with values in [0, 1]
    """
    # 1 / (1 + exp(-t)) = exp(-log(1 + exp(-t))): the exponent is <= 0, no overflow
    return np.exp(-np.logaddexp(0, -t))


def calculate_loss(y, tx, w):
    """Compute the logistic loss (mean negative log-likelihood).

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        w: weights, shape=(D,)

    Returns:
        loss: scalar
    """
    z = tx @ w
    # -[y log(sigma(z)) + (1 - y) log(1 - sigma(z))] = log(1 + exp(z)) - y z,
    # with logaddexp(0, z) = log(1 + exp(z)): no overflow and no log(0)
    return np.mean(np.logaddexp(0, z) - y * z)


def calculate_gradient(y, tx, w):
    """Compute the gradient of the logistic loss.

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
    """Do one step of gradient descent on the logistic loss.

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        w: current weights, shape=(D,)
        gamma: step size

    Returns:
        w: updated weights, shape=(D,)
    """
    return w - gamma * calculate_gradient(y, tx, w)


def penalized_logistic_regression(y, tx, w, lambda_):
    """Compute the gradient of the logistic loss plus lambda_ * ||w||^2.

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        w: weights, shape=(D,)
        lambda_: regularization parameter

    Returns:
        gradient: shape=(D,)
    """
    # the gradient of lambda_ * w^T w is 2 * lambda_ * w
    return calculate_gradient(y, tx, w) + 2 * lambda_ * w


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """Do one step of gradient descent on the penalized logistic loss.

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        w: current weights, shape=(D,)
        gamma: step size
        lambda_: regularization parameter

    Returns:
        w: updated weights, shape=(D,)
    """
    return w - gamma * penalized_logistic_regression(y, tx, w, lambda_)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        initial_w: initial weights, shape=(D,)
        max_iters: number of GD steps
        gamma: step size

    Returns:
        w: weights after the last step, shape=(D,)
        loss: logistic loss at the returned w
    """
    # copy: initial_w is never modified, also when max_iters = 0
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = learning_by_gradient_descent(y, tx, w, gamma)
    loss = calculate_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y in {0, 1}).

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        lambda_: regularization parameter of the penalty lambda_ * ||w||^2
        initial_w: initial weights, shape=(D,)
        max_iters: number of GD steps
        gamma: step size

    Returns:
        w: weights after the last step, shape=(D,)
        loss: logistic loss at the returned w, without the penalty term
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    loss = calculate_loss(y, tx, w)
    return w, loss
