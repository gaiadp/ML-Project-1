"""Train/validation splits, k-fold cross-validation and hyperparameter grid search."""

import numpy as np


def build_k_indices(n, k_fold, seed=1):
    """Randomly partition range(n) into k_fold folds of (almost) equal size.

    Returns a list of k_fold 1D index arrays; the first n % k_fold folds get one extra sample,
    so every sample is used exactly once for validation.
    """
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(n), k_fold)


def ridge_weights_svd(y, tx, lambdas):
    """Ridge solutions for a whole grid of lambdas from a single SVD of tx.

    Same objective as implementations.ridge_regression:
        (X^T X + 2 N lambda I) w = X^T y.
    With the thin SVD X = U S V^T this becomes
        w(lambda) = V diag(s / (s^2 + 2 N lambda)) U^T y,
    so after the one-off O(N D^2) SVD each lambda costs only O(D^2) (no linear system to solve).
    Working on X directly (instead of X^T X) also avoids squaring the condition number.

    Returns W of shape (len(lambdas), D): row i is the weight vector for lambdas[i].
    """
    n = tx.shape[0]
    u, s, vt = np.linalg.svd(tx, full_matrices=False)
    uty = u.T @ y                                               # (r,)
    lambdas = np.asarray(lambdas, dtype=float)
    # shrinkage factors s / (s^2 + 2 N lambda), one row per lambda: (L, r)
    factors = s / (s ** 2 + 2 * n * lambdas[:, None])
    return (factors * uty) @ vt                                 # (L, D)


def cross_validation_ridge(y, tx, lambdas, k_fold=5, seed=1):
    """Pick the ridge lambda with the lowest mean validation MSE over k folds.

    For each fold the SVD of the training block is computed once and reused for every
    lambda in the grid (see ridge_weights_svd).
    tx must already be preprocessed; if the preprocessing fits statistics on the data
    (standardization, imputation, ...), the fold-wise fit is the caller's responsibility.

    Returns (best_lambda, mean_val_losses, mean_train_losses), where the loss arrays have
    shape (len(lambdas),) and use the same 1/2 * MSE convention as implementations.py.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    k_indices = build_k_indices(len(y), k_fold, seed)
    val_losses = np.zeros((k_fold, len(lambdas)))
    train_losses = np.zeros((k_fold, len(lambdas)))

    for k in range(k_fold):
        val_idx = k_indices[k]
        train_idx = np.concatenate([k_indices[j] for j in range(k_fold) if j != k])
        x_tr, y_tr = tx[train_idx], y[train_idx]
        x_va, y_va = tx[val_idx], y[val_idx]

        w_all = ridge_weights_svd(y_tr, x_tr, lambdas)          # (L, D)
        # predictions for every lambda at once: (n_samples, L)
        train_losses[k] = np.mean((y_tr[:, None] - x_tr @ w_all.T) ** 2, axis=0) / 2
        val_losses[k] = np.mean((y_va[:, None] - x_va @ w_all.T) ** 2, axis=0) / 2

    mean_val = val_losses.mean(axis=0)
    mean_train = train_losses.mean(axis=0)
    best_lambda = lambdas[np.argmin(mean_val)]
    return best_lambda, mean_val, mean_train
