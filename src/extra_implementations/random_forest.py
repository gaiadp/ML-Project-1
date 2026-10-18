"""Random forest of histogram trees (Breiman, 2001), with NumPy only.

Every tree is grown on a random subset of the rows, and every node tries only
max_features random columns (trees.grow_tree): the trees make different errors, and
their average has a lower variance than a single deep tree. The predicted probability
of MICHD is the mean, over the trees, of the fraction of positives in the leaf.
The rows are drawn without replacement (subsample < 1) instead of the bootstrap of the
paper: a common variant, faster since every row is used at most once per tree.
Trees do not need standardized features (only the order of the values matters): the
pipeline output can be used as it is.
"""

import numpy as np

from src.extra_implementations.trees import (
    apply_bins,
    fit_bins,
    grow_tree,
    log_loss,
    tree_values,
)


def random_forest(
    y,
    tx,
    n_trees=100,
    max_depth=12,
    min_samples_leaf=50,
    max_features=None,
    subsample=0.5,
    n_bins=32,
    seed=None,
):
    """Train a random forest classifier.

    Args:
        y: labels in {0, 1}, shape=(N,)
        tx: feature matrix, shape=(N, D)
        n_trees: number of trees
        max_depth: depth of the deepest leaf
        min_samples_leaf: minimum number of training rows in a leaf
        max_features: columns tried at every node; None = sqrt(D)
        subsample: fraction of the rows drawn (without replacement) for every tree
        n_bins: maximum number of bins per column (trees.fit_bins)
        seed: seed of the rows, of the columns and of the bins; None = random

    Returns:
        model: dict with the bin edges and the list of trees, for
            random_forest_predict_proba
        loss: log-loss of the predicted probabilities on the training rows
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    n, d = tx.shape
    edges = fit_bins(tx, n_bins, seed=seed)
    xb = apply_bins(tx, edges)
    if max_features is None:
        max_features = max(1, int(np.sqrt(d)))
    n_rows = max(1, int(round(subsample * n)))
    features = np.arange(d)
    trees = []
    for _ in range(n_trees):
        rows = rng.choice(n, size=n_rows, replace=False)
        trees.append(
            grow_tree(
                xb,
                rows,
                y,
                None,
                n_bins,
                max_depth,
                min_samples_leaf,
                0.0,
                features,
                max_features,
                rng,
            )
        )
    model = dict(edges=edges, trees=trees)
    return model, log_loss(y, _mean_of_trees(trees, xb))


def _mean_of_trees(trees, xb):
    """Mean leaf value of the trees for every row of the binned matrix xb."""
    return np.mean([tree_values(tree, xb) for tree in trees], axis=0)


def random_forest_predict_proba(model, tx):
    """Predicted probability of the positive class, shape=(N,).

    Args:
        model: dict returned by random_forest
        tx: feature matrix, shape=(N, D), same columns as the training one

    Returns:
        probabilities in [0, 1], shape=(N,)
    """
    return _mean_of_trees(model["trees"], apply_bins(tx, model["edges"]))
