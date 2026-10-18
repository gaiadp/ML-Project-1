"""Sophia: second-order clipped stochastic optimization (Liu et al., 2023).

Applied to the regularized logistic objective of implementations.reg_logistic_regression
(logistic_objective.py), on random minibatches. Each step:

    m = beta1 m + (1 - beta1) g          momentum of the minibatch gradient g
    h = beta2 h + (1 - beta2) h_hat      every hess_every steps, h_hat = curvature
    w = w - gamma * clip(m / max(rho h, eps), -1, 1)

m / h is a diagonal Newton step: long where the curvature is small, short where it
is large. The clip limits the move of every coordinate to gamma per step, which
protects against a curvature estimate that is too small or noisy (at the start h is
close to 0 and the steps are clipped, sign-gradient steps of size gamma).
When nothing is clipped the step is (gamma / rho) * m / h.

Differences from the paper, which targets neural networks:
- h_hat is the exact diagonal of the Hessian on the minibatch (logistic_objective.
  diag_hessian), cheap for logistic regression; the paper estimates it (Hutchinson
  or Gauss-Newton-Bartlett) because for a network it cannot be computed;
- the L2 penalty is in the objective, as in reg_logistic_regression, instead of the
  decoupled weight decay of the paper;
- the clipping scale called gamma in the paper is rho here: gamma is the step size,
  as in the other methods.
"""

import numpy as np

from implementations import calculate_loss
from src.extra_implementations.logistic_objective import (
    diag_hessian,
    penalized_gradient,
)


def reg_logistic_sophia(
    y,
    tx,
    lambda_,
    initial_w,
    max_iters,
    gamma=0.05,
    batch_size=2048,
    beta1=0.965,
    beta2=0.99,
    rho=0.05,
    eps=1e-12,
    hess_every=10,
    seed=None,
):
    """Regularized logistic regression by Sophia on random minibatches.

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        lambda_: regularization parameter of the penalty lambda_ * ||w||^2
        initial_w: initial weights, shape=(D,)
        max_iters: number of steps (one minibatch each)
        gamma: step size, the largest move of a coordinate in one step
        batch_size: rows per minibatch, drawn with replacement; >= N uses all the
            rows at every step (deterministic, full-batch Sophia)
        beta1, beta2: decay of the moving averages of the gradient and of the curvature
        rho: clipping scale; when not clipped the step is (gamma / rho) * m / h
        eps: lower bound of rho * h, avoids a division by 0
        hess_every: steps between two updates of the curvature
        seed: seed of the minibatches; None = not reproducible

    Returns:
        w: weights after the last step, shape=(D,)
        loss: logistic loss on all the rows at the returned w, without the penalty
    """
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    full_batch = batch_size >= n
    w = np.array(initial_w, dtype=float)
    m = np.zeros_like(w)
    h = np.zeros_like(w)
    for t in range(max_iters):
        if full_batch:
            yb, xb = y, tx
        else:
            rows = rng.integers(0, n, size=batch_size)
            yb, xb = y[rows], tx[rows]
        m = beta1 * m + (1 - beta1) * penalized_gradient(yb, xb, w, lambda_)
        if t % hess_every == 0:
            h = beta2 * h + (1 - beta2) * diag_hessian(xb, w, lambda_)
        w = w - gamma * np.clip(m / np.maximum(rho * h, eps), -1, 1)
    return w, calculate_loss(y, tx, w)
