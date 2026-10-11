"""Feature engineering: one-hot encoding of categorical features, polynomial expansion,
feature combinations, bias column.
"""

import numpy as np


def add_bias(x):
    """Prepend a column of ones."""
    return np.hstack([np.ones((x.shape[0], 1)), x])


def one_hot_encode(x_col, categories=None):
    """One-hot encode a single categorical column (1D array).

    Parameters
    ----------
    x_col : np.ndarray, shape (N,)
    categories : array-like or None
        Fixed set/order of categories to encode against (fit on train, reuse on
        val/test so both get the same columns even if a category is missing
        from one split). If None, categories are taken from x_col itself.

    Returns
    -------
    encoded : np.ndarray, shape (N, len(categories))
    categories : np.ndarray, the categories used (in column order)
    """
    if categories is None:
        categories = np.unique(x_col[~np.isnan(x_col)])
    categories = np.asarray(categories)
    encoded = (x_col[:, None] == categories[None, :]).astype(float)
    return encoded, categories


def fit_categories(x_col, min_rows=1):
    """B9: categories of a column for one_hot_encode, fitted on the training set.

    The values with at least min_rows rows, without the most frequent one: it is the
    reference (all-zero row, its effect goes in the bias), together with the rarer
    values and the values never seen in training. Without the reference column the
    one-hot columns do not sum to the bias column (no exact collinearity).

    Args:
        x_col: shape=(N,), training values (NaN ignored)
        min_rows: minimum number of rows of a category

    Returns:
        categories: shape=(K,)
    """
    values, counts = np.unique(x_col[~np.isnan(x_col)], return_counts=True)
    if len(values) == 0:
        return values
    reference = values[np.argmax(counts)]
    frequent = values[counts >= min_rows]
    return frequent[frequent != reference]


def centered_powers(x, center, degree):
    """B11: powers 2..degree of the columns of x, centered at center (train means).

    Centering keeps x and x^2 from being almost the same column when x is far from 0.

    Args:
        x: shape=(N, K)
        center: shape=(K,)
        degree: highest power; below 2 there are no columns

    Returns:
        shape=(N, K * (degree - 1)): all the squares, then all the cubes, ...
    """
    z = x - center
    return np.hstack([np.empty((len(x), 0))] + [z**d for d in range(2, degree + 1)])


def build_poly(x_col, degree):
    """Polynomial expansion of a single numeric column: [x, x^2, ..., x^degree]."""
    return np.hstack([(x_col**d)[:, None] for d in range(1, degree + 1)])


def pairwise_products(x, col_pairs):
    """Build interaction features x[:, i] * x[:, j] for each (i, j) in col_pairs."""
    return np.hstack([(x[:, i] * x[:, j])[:, None] for i, j in col_pairs])
