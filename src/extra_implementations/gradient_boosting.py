"""Gradient boosting of histogram trees for the log-loss (Friedman, 2001), NumPy only.

The model is a sum of small trees on the log-odds scale:

    F(x) = F_0 + learning_rate * (tree_1(x) + ... + tree_T(x)),   p = sigmoid(F)

with F_0 = log-odds of the training prevalence. Each new tree is fitted to the error
of the current model: for every row g = y - p (minus the gradient of the log-loss)
and h = p (1 - p) (its second derivative), and the tree maximizes the second-order
gain of XGBoost (Chen & Guestrin, 2016), with leaf values -sum(gradient) /
(sum(h) + reg_lambda), i.e. a Newton step in each leaf (trees.py). The learning rate
shrinks every tree, so many small corrections add up. Every tree uses a random
fraction of the rows (subsample) and of the columns (colsample), as in stochastic
gradient boosting: faster, and less overfitting.
"""

import numpy as np

from implementations import sigmoid
from src.extra_implementations.trees import (
    apply_bins,
    fit_bins,
    grow_tree,
    log_loss,
    tree_values,
)


def gradient_boosting(
    y,
    tx,
    n_trees=200,
    learning_rate=0.1,
    max_depth=4,
    min_samples_leaf=100,
    reg_lambda=1.0,
    subsample=0.5,
    colsample=0.5,
    n_bins=32,
    seed=None,
    base_margin=None,
):
    """Train a gradient boosting classifier (log-loss).

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        n_trees: number of trees (boosting rounds)
        learning_rate: factor applied to every tree
        max_depth: depth of the deepest leaf
        min_samples_leaf: minimum number of training rows in a leaf
        reg_lambda: L2 regularization of the leaf values
        subsample: fraction of the rows drawn (without replacement) for every tree
        colsample: fraction of the columns drawn for every tree
        n_bins: maximum number of bins per column (trees.fit_bins)
        seed: seed of the rows, of the columns and of the bins; None = random
        base_margin: starting log-odds F_0 of every row, shape=(N,), e.g. the
            margin tx @ w of a logistic regression (logistic_boosting.py);
            None = the log-odds of the training prevalence, the same for all rows

    Returns:
        model: dict with the bin edges, F_0 and the trees, for
            gradient_boosting_predict_proba
        loss: log-loss of the predicted probabilities on the training rows
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    n, d = tx.shape
    edges = fit_bins(tx, n_bins, seed=seed)
    xb = apply_bins(tx, edges)
    if base_margin is None:
        prevalence = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        base = float(np.log(prevalence / (1 - prevalence)))
        f = np.full(n, base)
    else:
        # the trees correct the given margin; it is passed again to predict_proba
        base = 0.0
        f = np.array(base_margin, dtype=float)
    n_rows = max(1, int(round(subsample * n)))
    n_cols = max(1, int(round(colsample * d)))
    trees = []
    for _ in range(n_trees):
        p = sigmoid(f)
        rows = rng.choice(n, size=n_rows, replace=False)
        features = np.sort(rng.choice(d, size=n_cols, replace=False))
        tree = grow_tree(
            xb,
            rows,
            y - p,
            p * (1 - p),
            n_bins,
            max_depth,
            min_samples_leaf,
            reg_lambda,
            features,
        )
        tree["value"] *= learning_rate  # the shrinkage is stored in the tree
        f += tree_values(tree, xb)  # update of all the rows, also those not drawn
        trees.append(tree)
    model = dict(edges=edges, base=base, trees=trees)
    return model, log_loss(y, sigmoid(f))


def gradient_boosting_predict_proba(model, tx, base_margin=None):
    """Predicted probability of the positive class, shape=(N,).

    Args:
        model: dict returned by gradient_boosting
        tx: feature matrix, shape=(N, D), same columns as the training one
        base_margin: starting log-odds of every row of tx, shape=(N,): needed if
            the model was trained with a base_margin, computed the same way

    Returns:
        probabilities in [0, 1], shape=(N,)
    """
    xb = apply_bins(tx, model["edges"])
    f = np.full(tx.shape[0], model["base"])
    if base_margin is not None:
        f += base_margin
    for tree in model["trees"]:
        f += tree_values(tree, xb)
    return sigmoid(f)
