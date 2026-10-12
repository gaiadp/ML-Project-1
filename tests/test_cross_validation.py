"""Tests for the k-fold cross-validation of src/cross_validation.py: stratified folds,
preprocessing fitted on the training part of each fold only, ridge lambda grid."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.cross_validation as cv  # noqa: E402
from implementations import ridge_regression  # noqa: E402
from src.pipeline import DEFAULT_CONFIG  # noqa: E402
from test_pipeline import fake_brfss  # noqa: E402

CONFIG = dict(DEFAULT_CONFIG, min_missing_rows=5, one_hot_min_rows=5)


def test_stratified_k_indices():
    y = np.where(np.arange(1003) % 10 == 0, 1, -1)
    folds = cv.stratified_k_indices(y, 5, seed=0)
    assert len(folds) == 5
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(1003))
    # 101 positives: 20 or 21 per fold
    assert all(np.sum(y[f] == 1) in (20, 21) for f in folds)
    assert all(abs(len(f) - 1003 / 5) <= 2 for f in folds)
    np.testing.assert_array_equal(folds[0], cv.stratified_k_indices(y, 5, seed=0)[0])


def test_preprocessing_never_sees_the_validation_rows(monkeypatch):
    x, y, names = fake_brfss()
    seen = []
    fit_preprocess = cv.fit_preprocess

    def spy(x_train, y_train, names, config):
        seen.append(x_train[:, names.index("SEQNO")].copy())  # row ids
        return fit_preprocess(x_train, y_train, names, config)

    monkeypatch.setattr(cv, "fit_preprocess", spy)
    out = cv.cross_validate(x, y, names, CONFIG, "ridge_regression", k_fold=4, lambda_=1e-3)
    folds = cv.stratified_k_indices(y, 4, seed=1)
    ids = x[:, names.index("SEQNO")]
    assert len(seen) == 4
    for fitted_on, val_idx in zip(seen, folds):
        assert not np.isin(fitted_on, ids[val_idx]).any()
        assert len(fitted_on) == len(y) - len(val_idx)
    # every row gets exactly one out-of-fold score
    assert np.isfinite(out["oof_scores"]).all()
    assert out["f1"].shape == (4,) and np.all((out["f1"] >= 0) & (out["f1"] <= 1))
    assert np.all(out["f1_val_best"] >= out["f1"] - 1e-12)
    assert np.isclose(out["f1_mean"], out["f1"].mean())
    assert np.isclose(out["f1_std"], out["f1"].std())


def test_cross_validation_ridge_matches_ridge_regression(monkeypatch):
    x, y, names = fake_brfss()
    lambdas = np.array([1e-4, 1e-2, 1.0])
    weights = []
    svd = cv.ridge_weights_svd

    def spy(y_tr, tx_tr, lambdas):
        w_all = svd(y_tr, tx_tr, lambdas)
        weights.append((y_tr, tx_tr, w_all))
        return w_all

    monkeypatch.setattr(cv, "ridge_weights_svd", spy)
    out = cv.cross_validation_ridge(x, y, names, CONFIG, lambdas, k_fold=4)
    assert len(weights) == 4
    y_tr, tx_tr, w_all = weights[0]
    for lam, w in zip(lambdas, w_all):
        np.testing.assert_allclose(w, ridge_regression(y_tr, tx_tr, lam)[0], atol=1e-8)
    assert out["best_lambda"] in lambdas
    assert out["val_f1"].shape == out["val_loss"].shape == (3,)
