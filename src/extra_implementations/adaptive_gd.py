"""Gradient descent with an adaptive step: Barzilai-Borwein step + Armijo backtracking.

Minimizes the regularized logistic objective of implementations.reg_logistic_regression
(logistic_objective.py) without a step size to tune:
- Barzilai-Borwein step (Barzilai & Borwein, 1988): t = (s . s) / (s . r), with
  s = w_k - w_{k-1} the last move and r = g_k - g_{k-1} the change of the gradient.
  (s . r) / (s . s) is the curvature of f along s, so t is the inverse of the
  curvature in the direction of the last move: a quasi-Newton step with a single
  number instead of a matrix. Large steps where f is flat, small ones where it bends.
- Armijo backtracking: the trial step is halved until f decreases enough,
  f(w - t g) <= f(w) - c t ||g||^2. The BB step alone can increase f; with the
  backtracking every iteration decreases it.
"""

import numpy as np

from implementations import calculate_loss
from src.extra_implementations.logistic_objective import (
    penalized_gradient,
    penalized_loss,
)

_ARMIJO_C = 1e-4  # sufficient decrease constant of the Armijo condition
_MIN_STEP, _MAX_STEP = 1e-10, 1e10  # bounds of the step


def reg_logistic_adaptive_gd(y, tx, lambda_, initial_w, max_iters, gamma=1.0):
    """Regularized logistic regression by gradient descent with an adaptive step.

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        lambda_: regularization parameter of the penalty lambda_ * ||w||^2
        initial_w: initial weights, shape=(D,)
        max_iters: number of gradient steps
        gamma: trial step of the first iteration; then the Barzilai-Borwein step

    Returns:
        w: weights after the last step, shape=(D,)
        loss: logistic loss at the returned w, without the penalty term
    """
    w = np.array(initial_w, dtype=float)
    f = penalized_loss(y, tx, w, lambda_)
    g = penalized_gradient(y, tx, w, lambda_)
    step = gamma
    for _ in range(max_iters):
        g_norm2 = g @ g
        if g_norm2 == 0:
            break  # exact minimum
        # Armijo backtracking, starting from the trial step
        w_new = w - step * g
        f_new = penalized_loss(y, tx, w_new, lambda_)
        while f_new > f - _ARMIJO_C * step * g_norm2 and step > _MIN_STEP:
            step /= 2
            w_new = w - step * g
            f_new = penalized_loss(y, tx, w_new, lambda_)
        if f_new > f:
            break  # no decrease even with the smallest step: numerical precision
        g_new = penalized_gradient(y, tx, w_new, lambda_)
        # Barzilai-Borwein trial step for the next iteration; s . r > 0 for a convex
        # f, the fallback only guards against rounding errors
        s, r = w_new - w, g_new - g
        s_r = s @ r
        step = float(np.clip(s @ s / s_r, _MIN_STEP, _MAX_STEP)) if s_r > 0 else gamma
        w, f, g = w_new, f_new, g_new
    return w, calculate_loss(y, tx, w)
