"""Hybrid model: regularized logistic regression + gradient boosting on its residuals.

    F(x) = x^T w + learning_rate * (tree_1(x) + ... + tree_T(x)),   p = sigmoid(F)

1. w: the exact minimum of the objective of reg_logistic_regression (Newton,
   newton.py). The linear model captures the additive part of the signal, which
   dominates in this dataset (the log-odds of most features are close to linear,
   EDA A6), smoothly and exactly.
2. The trees: gradient boosting (gradient_boosting.py) started from the margin x^T w
   instead of the prevalence. At the minimum of the logistic loss the residuals
   y - p are (up to the penalty) uncorrelated with every column, so the trees can
   only learn what a linear model cannot: interactions between columns (age x sex,
   ...) and curved effects of the numeric columns. Trees alone waste their depth
   approximating the linear part with steps.
With n_trees = 0 the model is the logistic regression itself: the CV chooses how
much the trees add.
"""

from src.extra_implementations.gradient_boosting import (
    gradient_boosting,
    gradient_boosting_predict_proba,
)
from src.extra_implementations.newton import reg_logistic_newton


def logistic_boosting(
    y,
    tx,
    lambda_,
    initial_w,
    max_iters,
    n_trees=100,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=100,
    reg_lambda=1.0,
    subsample=0.5,
    colsample=0.5,
    n_bins=32,
    seed=None,
):
    """Train the logistic regression, then boosted trees on its residuals.

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        lambda_: regularization parameter of the logistic regression
        initial_w: initial weights of Newton's method, shape=(D,)
        max_iters: maximum number of Newton steps
        n_trees, learning_rate, max_depth, min_samples_leaf, reg_lambda, subsample,
            colsample, n_bins, seed: parameters of the boosted trees
            (gradient_boosting.gradient_boosting)

    Returns:
        model: dict with the weights w and the boosted trees, for
            logistic_boosting_predict_proba
        loss: log-loss of the predicted probabilities on the training rows
    """
    w, _ = reg_logistic_newton(y, tx, lambda_, initial_w, max_iters)
    trees, loss = gradient_boosting(
        y,
        tx,
        n_trees=n_trees,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        reg_lambda=reg_lambda,
        subsample=subsample,
        colsample=colsample,
        n_bins=n_bins,
        seed=seed,
        base_margin=tx @ w,
    )
    return dict(w=w, trees=trees), loss


def logistic_boosting_predict_proba(model, tx):
    """Predicted probability of the positive class, shape=(N,).

    Args:
        model: dict returned by logistic_boosting
        tx: feature matrix, shape=(N, D), same columns as the training one

    Returns:
        probabilities in [0, 1], shape=(N,)
    """
    return gradient_boosting_predict_proba(
        model["trees"], tx, base_margin=tx @ model["w"]
    )
