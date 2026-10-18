"""Histogram decision trees, shared by random_forest.py and gradient_boosting.py.

The features are first cut into at most n_bins bins at quantiles of the training data
(fit_bins, apply_bins), as the "hist" method of LightGBM / XGBoost does: a split is
then "bin <= b goes left", and the best split of a node comes from per-bin sums
computed with np.bincount, instead of sorting every column at every node.

One split criterion for both models. Every row has a value g and a weight h, and the
best split maximizes

    G_L^2 / (H_L + reg) + G_R^2 / (H_R + reg)

where G, H are the sums of g, h over the rows of each side; the leaf value is
G / (H + reg).
- Random forest: g = y in {0, 1}, h = 1, reg = 0. On each side G^2 / H = n * mean^2,
  so the split minimizes the squared error of predicting the mean of y on each side,
  i.e. the Gini impurity; the leaf value is the fraction of positives.
- Gradient boosting: g = y - p (minus the gradient of the log-loss), h = p (1 - p),
  reg > 0: the second-order gain of XGBoost (Chen & Guestrin, 2016); the leaf value
  is one Newton step on the log-loss.
"""

import numpy as np

_MIN_GAIN = 1e-12  # a split must improve the criterion by more than this


def fit_bins(tx, n_bins=32, max_rows=100000, seed=None):
    """Bin edges of every column, at quantiles of the training data.

    Args:
        tx: training features, shape=(N, D)
        n_bins: maximum number of bins per column, at most 256 (stored as uint8)
        max_rows: the quantiles are computed on at most this many random rows
        seed: seed of that sample of rows

    Returns:
        list of D sorted arrays: a value v goes to the bin k such that
        edges[k-1] <= v < edges[k], bins 0 .. len(edges). A binary column has one
        edge, a constant column (e.g. the bias) none.
    """
    if not 2 <= n_bins <= 256:
        raise ValueError(f"n_bins must be in 2 .. 256, got {n_bins}")
    sample = tx
    if tx.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        sample = tx[rng.choice(tx.shape[0], size=max_rows, replace=False)]
    levels = np.linspace(0, 1, n_bins + 1)[1:-1]  # n_bins - 1 inner quantiles
    quantiles = np.quantile(sample, levels, axis=0)
    lowest = sample.min(axis=0)
    # an edge at the minimum would leave bin 0 empty
    return [np.unique(q[q > low]) for q, low in zip(quantiles.T, lowest)]


def apply_bins(tx, edges):
    """Bin index of every value, shape=(N, D), uint8 (column-major: the trees read
    one column at a time)."""
    xb = np.empty(tx.shape, dtype=np.uint8, order="F")
    for j, e in enumerate(edges):
        xb[:, j] = np.searchsorted(e, tx[:, j], side="right")
    return xb


def best_split(xb, rows, features, g, h, n_bins, reg, min_samples_leaf):
    """Best split "bin <= b goes left" of a node, among the given features.

    Args:
        xb: binned features, shape=(N, D)
        rows: indices of the rows in the node
        features: column indices to try
        g, h: value and weight of every row of xb, shape=(N,); h None = all 1
        n_bins: number of bins (bin indices are < n_bins)
        reg: regularization added to H in the criterion
        min_samples_leaf: minimum number of rows on each side

    Returns:
        (feature, bin) of the best split, or None if no split improves the criterion
    """
    g_node = g[rows]
    h_node = None if h is None else h[rows]
    n = len(rows)
    g_tot = g_node.sum()
    h_tot = n if h is None else h_node.sum()
    parent = g_tot**2 / (h_tot + reg)
    best_gain, best = _MIN_GAIN, None
    for j in features:
        b = xb[rows, j]
        count = np.bincount(b, minlength=n_bins)
        g_sum = np.bincount(b, weights=g_node, minlength=n_bins)
        h_sum = count if h is None else np.bincount(b, weights=h_node, minlength=n_bins)
        # left side = bins 0 .. k, for the thresholds k = 0 .. n_bins - 2
        n_left = np.cumsum(count)[:-1]
        g_left = np.cumsum(g_sum)[:-1]
        h_left = np.cumsum(h_sum)[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = (
                g_left**2 / (h_left + reg)
                + (g_tot - g_left) ** 2 / (h_tot - h_left + reg)
                - parent
            )
        valid = (n_left >= min_samples_leaf) & (n - n_left >= min_samples_leaf)
        gain = np.where(valid, gain, -np.inf)
        k = int(np.argmax(gain))
        if gain[k] > best_gain:
            best_gain, best = gain[k], (int(j), k)
    return best


def grow_tree(
    xb,
    rows,
    g,
    h,
    n_bins,
    max_depth,
    min_samples_leaf,
    reg,
    features,
    max_features=None,
    rng=None,
):
    """Grow one tree on the given rows, depth first.

    Args:
        xb: binned features, shape=(N, D)
        rows: indices of the rows the tree is trained on
        g, h: value and weight of every row of xb, shape=(N,); h None = all 1
        n_bins: number of bins (bin indices are < n_bins)
        max_depth: depth of the deepest leaf (the root has depth 0)
        min_samples_leaf: minimum number of rows in a leaf, at least 1
        reg: regularization of the split criterion and of the leaf values
        features: column indices the tree may split on
        max_features: if given, every node tries only this many random features
            among them (random forest)
        rng: np.random.Generator, needed with max_features

    Returns:
        tree: dict of arrays, one entry per node: feature, bin (the split), left,
            right (indices of the children, -1 for a leaf), value (leaf value
            G / (H + reg), also stored for the inner nodes)
    """
    min_samples_leaf = max(1, int(min_samples_leaf))
    feature, split_bin, left, right, value = [], [], [], [], []

    def new_node(node_rows):
        h_sum = len(node_rows) if h is None else h[node_rows].sum()
        value.append(g[node_rows].sum() / (h_sum + reg))
        feature.append(-1)
        split_bin.append(0)
        left.append(-1)
        right.append(-1)
        return len(value) - 1

    stack = [(new_node(rows), rows, 0)]
    while stack:
        node, node_rows, depth = stack.pop()
        if depth >= max_depth or len(node_rows) < 2 * min_samples_leaf:
            continue
        tried = features
        if max_features is not None and max_features < len(features):
            tried = rng.choice(features, size=max_features, replace=False)
        split = best_split(xb, node_rows, tried, g, h, n_bins, reg, min_samples_leaf)
        if split is None:
            continue
        j, k = split
        goes_left = xb[node_rows, j] <= k
        feature[node], split_bin[node] = j, k
        left[node] = new_node(node_rows[goes_left])
        right[node] = new_node(node_rows[~goes_left])
        stack.append((right[node], node_rows[~goes_left], depth + 1))
        stack.append((left[node], node_rows[goes_left], depth + 1))
    return dict(
        feature=np.array(feature, dtype=np.intp),
        bin=np.array(split_bin, dtype=np.intp),
        left=np.array(left, dtype=np.intp),
        right=np.array(right, dtype=np.intp),
        value=np.array(value, dtype=float),
    )


def tree_values(tree, xb):
    """Leaf value of every row of xb, shape=(N,): all the rows go down one level
    per pass of the loop."""
    node = np.zeros(xb.shape[0], dtype=np.intp)
    active = np.flatnonzero(tree["left"][node] >= 0)
    while len(active):
        at = node[active]
        goes_left = xb[active, tree["feature"][at]] <= tree["bin"][at]
        node[active] = np.where(goes_left, tree["left"][at], tree["right"][at])
        active = active[tree["left"][node[active]] >= 0]
    return tree["value"][node]


def log_loss(y, p):
    """Mean log-loss of the probabilities p for the labels y in {0, 1}."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p)))
