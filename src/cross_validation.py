"""Train/validation splits, k-fold cross-validation and hyperparameter grid search.

The cross-validation functions take the RAW data and fit the preprocessing
(src/pipeline.py) inside each fold, on its training part only: the statistics
(informative NaN columns chosen with the labels, medians, one-hot categories,
mean / std) never see the validation part, as on the real test set.
"""

import numpy as np

from src.metrics import best_threshold, f1_score, predict_labels
from src.models import fit, predict_scores
from src.pipeline import apply_preprocess, fit_preprocess


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
    uty = u.T @ y  # (r,)
    lambdas = np.asarray(lambdas, dtype=float)
    # shrinkage factors s / (s^2 + 2 N lambda), one row per lambda: (L, r)
    factors = s / (s**2 + 2 * n * lambdas[:, None])
    return (factors * uty) @ vt  # (L, D)


def stratified_k_indices(y, k_fold, seed=1):
    """Partition range(len(y)) into k_fold folds with the class proportions of y.

    The positives (label 1) and the negatives are permuted and split separately, so
    every fold gets ~1/k_fold of each class (8.8% positives in every fold).

    Returns:
        list of k_fold sorted 1D index arrays, disjoint, together range(len(y))
    """
    rng = np.random.default_rng(seed)
    is_pos = np.asarray(y) == 1
    folds = [[] for _ in range(k_fold)]
    for idx in (np.flatnonzero(is_pos), np.flatnonzero(~is_pos)):
        for k, part in enumerate(np.array_split(rng.permutation(idx), k_fold)):
            folds[k].append(part)
    return [np.sort(np.concatenate(parts)) for parts in folds]


def _fold(k_indices, k):
    """Training and validation indices of fold k."""
    train_idx = np.concatenate([idx for j, idx in enumerate(k_indices) if j != k])
    return np.sort(train_idx), k_indices[k]


def _preprocess_fold(x, y, names, config, train_idx, val_idx):
    """Fit the preprocessing on the training rows only and apply it to both parts."""
    state = fit_preprocess(x[train_idx], y[train_idx], names, config)
    tx_tr = apply_preprocess(x[train_idx], names, state)
    tx_va = apply_preprocess(x[val_idx], names, state)
    return tx_tr, tx_va, state


def cross_validate(x, y, names, config, method, k_fold=5, seed=1, **fit_kwargs):
    """k-fold cross-validation of preprocessing + model, scored with F1.

    For each stratified fold: fit_preprocess on the training part, fit the model,
    choose the decision threshold that maximizes F1 on the TRAINING scores, and
    measure F1 on the validation part with that threshold. f1_val_best (threshold
    chosen on the validation part itself) is optimistic, as the single-split
    number of run.py: report f1, compare with f1_val_best.

    Args:
        x: raw data, shape=(N, D) (Id column removed)
        y: labels in {-1, 1}, shape=(N,)
        names: the D column names
        config: preprocessing config (src/pipeline.py)
        method: one of src.models.ALL_METHODS
        k_fold, seed: folds and their seed, also the seed of mean_squared_error_sgd
        **fit_kwargs: lambda_, gamma, max_iters, ... for src.models.fit

    Returns:
        dict with f1_mean, f1_std (mean and standard deviation of f1 over the
        folds); per fold (shape=(k_fold,)): f1, f1_val_best, threshold, n_features;
        and oof_scores, shape=(N,): validation score of each row (out of fold)
    """
    return cross_validate_grid(x, y, names, config, method, [fit_kwargs], k_fold, seed)[
        0
    ]


def cross_validate_grid(x, y, names, config, method, grid, k_fold=5, seed=1):
    """cross_validate for several sets of fit parameters on the same folds.

    Each fold is preprocessed once and reused for every entry of the grid. For
    ridge_regression with a grid of lambdas > 0 only, the weights of all the lambdas
    come from a single SVD per fold (ridge_weights_svd).

    Args:
        x, y, names, config, method, k_fold, seed: as cross_validate
        grid: list of dicts of fit parameters (lambda_, gamma, max_iters, ...)

    Returns:
        list of dicts as cross_validate returns, one per entry of grid
    """
    y = np.asarray(y)
    k_indices = stratified_k_indices(y, k_fold, seed)
    outs = []
    for _ in grid:
        out = {key: np.zeros(k_fold) for key in ("f1", "f1_val_best", "threshold")}
        out["n_features"] = np.zeros(k_fold, dtype=int)
        out["oof_scores"] = np.full(len(y), np.nan)
        outs.append(out)
    # lambda = 0 would divide by a zero singular value: solved by least_squares
    svd_lambdas = None
    if method == "ridge_regression" and all(set(p) <= {"lambda_"} for p in grid):
        lambdas = np.array([p.get("lambda_", 0.0) for p in grid], dtype=float)
        svd_lambdas = lambdas if np.all(lambdas > 0) else None

    for k in range(k_fold):
        train_idx, val_idx = _fold(k_indices, k)
        tx_tr, tx_va, _ = _preprocess_fold(x, y, names, config, train_idx, val_idx)
        y_tr = y[train_idx]
        if svd_lambdas is not None:
            weights = ridge_weights_svd(y_tr.astype(float), tx_tr, svd_lambdas)
        else:
            weights = [fit(method, y_tr, tx_tr, seed=seed, **p)[0] for p in grid]
        for out, w in zip(outs, weights):
            threshold, _ = best_threshold(predict_scores(method, w, tx_tr), y_tr)
            scores = predict_scores(method, w, tx_va)
            out["threshold"][k] = threshold
            out["f1"][k] = f1_score(y[val_idx], predict_labels(scores, threshold))
            out["f1_val_best"][k] = best_threshold(scores, y[val_idx])[1]
            out["n_features"][k] = tx_va.shape[1]
            out["oof_scores"][val_idx] = scores
        del tx_tr, tx_va
    for out in outs:
        out["f1_mean"] = float(out["f1"].mean())
        out["f1_std"] = float(out["f1"].std())
    return outs


def cross_validation_ridge(x, y, names, config, lambdas, k_fold=5, seed=1):
    """Ridge lambda grid by k-fold CV, the preprocessing fitted inside each fold.

    For each stratified fold the preprocessing is fitted on the training part, then
    the SVD of the training block is computed once and reused for every lambda of
    the grid (see ridge_weights_svd). F1 uses, for each lambda, the threshold that
    maximizes F1 on the training scores of the fold (as cross_validate).

    Args:
        x: raw data, shape=(N, D); y: labels in {-1, 1}, shape=(N,)
        names: the D column names; config: preprocessing config (src/pipeline.py)
        lambdas: grid, shape=(L,)

    Returns:
        dict with best_lambda (highest mean validation F1) and, shape=(L,): val_f1,
        val_loss, train_loss (means over the folds; losses = 1/2 MSE, as in
        implementations.py)
    """
    y = np.asarray(y, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    k_indices = stratified_k_indices(y, k_fold, seed)
    val_f1 = np.zeros((k_fold, len(lambdas)))
    val_losses = np.zeros((k_fold, len(lambdas)))
    train_losses = np.zeros((k_fold, len(lambdas)))

    for k in range(k_fold):
        train_idx, val_idx = _fold(k_indices, k)
        tx_tr, tx_va, _ = _preprocess_fold(x, y, names, config, train_idx, val_idx)
        y_tr, y_va = y[train_idx], y[val_idx]

        w_all = ridge_weights_svd(y_tr, tx_tr, lambdas)  # (L, D)
        # predictions for every lambda at once: (n_samples, L)
        scores_tr, scores_va = tx_tr @ w_all.T, tx_va @ w_all.T
        train_losses[k] = np.mean((y_tr[:, None] - scores_tr) ** 2, axis=0) / 2
        val_losses[k] = np.mean((y_va[:, None] - scores_va) ** 2, axis=0) / 2
        for i in range(len(lambdas)):
            threshold, _ = best_threshold(scores_tr[:, i], y_tr)
            val_f1[k, i] = f1_score(y_va, predict_labels(scores_va[:, i], threshold))

    mean_f1 = val_f1.mean(axis=0)
    return dict(
        best_lambda=lambdas[np.argmax(mean_f1)],
        val_f1=mean_f1,
        val_loss=val_losses.mean(axis=0),
        train_loss=train_losses.mean(axis=0),
    )


def stratified_split(y, val_ratio=0.2, seed=1):
    """Split range(len(y)) into train / validation indices, stratified by class.

    The same fraction val_ratio of the positives (label 1) and of the negatives goes to
    validation, so both parts keep the class proportions of y.

    Args:
        y: labels, shape=(N,), positive class = 1
        val_ratio: fraction of each class put in the validation set
        seed: seed of the random permutation

    Returns:
        train_idx, val_idx: sorted 1D index arrays, disjoint, together range(N)
    """
    rng = np.random.default_rng(seed)
    is_pos = np.asarray(y) == 1
    train_parts, val_parts = [], []
    for idx in (np.flatnonzero(is_pos), np.flatnonzero(~is_pos)):
        idx = rng.permutation(idx)
        n_val = int(round(val_ratio * len(idx)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))
