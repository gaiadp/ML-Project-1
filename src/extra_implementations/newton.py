"""Newton's method for regularized logistic regression (IRLS).

Each step solves H d = g, with g and H the gradient and the Hessian of the objective
of implementations.reg_logistic_regression (logistic_objective.py), and moves to
w - t d. Near the minimum the full step t = 1 is accepted and the convergence is
quadratic: ~10 steps reach the minimum to machine precision, with no step size to
choose. Far from it, t is halved until f decreases enough (damped Newton).
For logistic regression this is iteratively reweighted least squares (IRLS): each
step is a weighted least-squares problem with weights s = sigmoid (1 - sigmoid).
Cost of a step: O(N D^2) for H and O(D^3) for the linear system, small for D ~ 400.
"""

import numpy as np

from implementations import calculate_loss
from src.extra_implementations.logistic_objective import (
    hessian,
    penalized_gradient,
    penalized_loss,
)

_ARMIJO_C = 1e-4  # sufficient decrease constant of the Armijo condition
_MIN_STEP = 1e-10
# added to the diagonal of H: with lambda_ = 0 and collinear columns H is singular
_JITTER = 1e-10


def reg_logistic_newton(y, tx, lambda_, initial_w, max_iters, tol=1e-10):
    """Regularized logistic regression by damped Newton steps.

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        lambda_: regularization parameter of the penalty lambda_ * ||w||^2
        initial_w: initial weights, shape=(D,)
        max_iters: maximum number of Newton steps
        tol: stop when half the squared Newton decrement g^T H^-1 g / 2 (an
            estimate of f(w) - min f) is below tol

    Returns:
        w: weights after the last step, shape=(D,)
        loss: logistic loss at the returned w, without the penalty term
    """
    w = np.array(initial_w, dtype=float)
    f = penalized_loss(y, tx, w, lambda_)
    for _ in range(max_iters):
        g = penalized_gradient(y, tx, w, lambda_)
        h = hessian(tx, w, lambda_)
        d = np.linalg.solve(h + _JITTER * np.eye(len(w)), g)
        decrement = g @ d
        if decrement / 2 <= tol:
            break
        # damped step: halve t until the Armijo condition holds along -d
        step = 1.0
        w_new = w - d
        f_new = penalized_loss(y, tx, w_new, lambda_)
        while f_new > f - _ARMIJO_C * step * decrement and step > _MIN_STEP:
            step /= 2
            w_new = w - step * d
            f_new = penalized_loss(y, tx, w_new, lambda_)
        if f_new > f:
            break  # no decrease even with the smallest step: numerical precision
        w, f = w_new, f_new
    return w, calculate_loss(y, tx, w)
