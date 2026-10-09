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


def build_poly(x_col, degree):
    """Polynomial expansion of a single numeric column: [x, x^2, ..., x^degree]."""
    return np.hstack([(x_col**d)[:, None] for d in range(1, degree + 1)])


def pairwise_products(x, col_pairs):
    """Build interaction features x[:, i] * x[:, j] for each (i, j) in col_pairs."""
    return np.hstack([(x[:, i] * x[:, j])[:, None] for i, j in col_pairs])
