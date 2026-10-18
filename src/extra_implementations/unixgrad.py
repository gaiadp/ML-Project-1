"""UniXGrad: universal adaptive extra-gradient method (Kavis et al., 2019).

Applied to the regularized logistic objective of implementations.reg_logistic_regression
(logistic_objective.py), in the Euclidean case, on the ball K = {w : ||w|| <= radius}
of diameter D = 2 radius. With weights a_t = t and A_t = a_1 + ... + a_t, step t:

    z_t  = (a_t v_{t-1} + sum_{i<t} a_i x_i) / A_t,   M_t = grad f(z_t)   (look-ahead)
    x_t  = proj_K(v_{t-1} - a_t eta_t M_t)
    xb_t = (a_t x_t + sum_{i<t} a_i x_i) / A_t,      g_t = grad f(xb_t)
    v_t  = proj_K(v_{t-1} - a_t eta_t g_t)
    eta_t = 2 D / sqrt(1 + sum_{i<t} a_i^2 ||g_i - M_i||^2)

and the result is the weighted average xb_T. The step eta_t adapts by itself: it
shrinks only as much as the gradient changes between the look-ahead point and the
average. Without knowing the smoothness constant of f, the method reaches the rate
O(1 / T^2) of accelerated gradient descent on smooth functions (and O(1 / sqrt(T)) on
non-smooth ones: "universal"). Two gradients per step.
The minimum of f must lie inside K: radius is a bound on ||w*||, too large only
makes the first steps longer.
"""

import numpy as np

from implementations import calculate_loss
from src.extra_implementations.logistic_objective import penalized_gradient


def _project(w, radius):
    """Euclidean projection on the ball ||w|| <= radius."""
    norm = np.linalg.norm(w)
    return w if norm <= radius else w * (radius / norm)


def reg_logistic_unixgrad(y, tx, lambda_, initial_w, max_iters, radius=10.0):
    """Regularized logistic regression by UniXGrad.

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        lambda_: regularization parameter of the penalty lambda_ * ||w||^2
        initial_w: initial weights, shape=(D,)
        max_iters: number of steps (two gradients each)
        radius: radius of the ball that must contain the minimum

    Returns:
        w: weighted average of the iterates after the last step, shape=(D,)
        loss: logistic loss at the returned w, without the penalty term
    """
    w = np.array(initial_w, dtype=float)
    diameter = 2 * radius
    v = _project(w, radius)  # the sequence v_t (called y_t in the paper)
    weighted_sum = np.zeros_like(w)  # sum_{i<t} a_i x_i
    a_sum = 0.0  # A_t
    sq_sum = 0.0  # sum_{i<t} a_i^2 ||g_i - M_i||^2
    for t in range(1, max_iters + 1):
        a = float(t)
        a_sum += a
        eta = 2 * diameter / np.sqrt(1 + sq_sum)
        m_grad = penalized_gradient(y, tx, (a * v + weighted_sum) / a_sum, lambda_)
        x = _project(v - a * eta * m_grad, radius)
        w = (a * x + weighted_sum) / a_sum
        g = penalized_gradient(y, tx, w, lambda_)
        v = _project(v - a * eta * g, radius)
        sq_sum += a**2 * np.sum((g - m_grad) ** 2)
        weighted_sum += a * x
    return w, calculate_loss(y, tx, w)
