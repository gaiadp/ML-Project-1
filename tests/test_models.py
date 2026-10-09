"""Tests for src/models.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import implementations as impl  # noqa: E402
from src.models import (  # noqa: E402
    METHODS,
    fit,
    predict,
    predict_scores,
    to_01,
    to_pm1,
)

N, D = 300, 4


@pytest.fixture
def data():
    """Linearly separable labels in {-1, 1}, bias column + 3 features."""
    rng = np.random.default_rng(0)
    tx = np.c_[np.ones(N), rng.normal(size=(N, D - 1))]
    y = np.where(tx @ np.array([0.5, 2.0, -1.0, 1.0]) > 0, 1, -1)
    return y, tx


def test_label_conversion():
    y = np.array([-1, 1, 1, -1])
    np.testing.assert_array_equal(to_01(y), [0.0, 1.0, 1.0, 0.0])
    np.testing.assert_array_equal(to_pm1(to_01(y)), y)
    assert to_01(y).dtype == float and to_pm1([0, 1]).dtype.kind == "i"


def test_fit_logistic_uses_01_labels(data):
    y, tx = data
    w, loss = fit("reg_logistic_regression", y, tx, lambda_=0.1, gamma=0.5)
    w_ref, loss_ref = impl.reg_logistic_regression(
        to_01(y), tx, 0.1, np.zeros(D), 100, 0.5
    )
    np.testing.assert_array_equal(w, w_ref)
    assert loss == loss_ref


def test_fit_linear_uses_pm1_labels(data):
    y, tx = data
    np.testing.assert_array_equal(
        fit("ridge_regression", y, tx, lambda_=0.01)[0],
        impl.ridge_regression(y, tx, 0.01)[0],
    )


def test_sgd_seed_is_reproducible(data):
    y, tx = data
    w1, _ = fit("mean_squared_error_sgd", y, tx, gamma=0.01, seed=3)
    w2, _ = fit("mean_squared_error_sgd", y, tx, gamma=0.01, seed=3)
    np.testing.assert_array_equal(w1, w2)


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_classify_separable_data(method, data):
    y, tx = data
    w, _ = fit(method, y, tx, lambda_=1e-4, gamma=0.1, max_iters=500, seed=1)
    scores = predict_scores(method, w, tx)
    labels = predict(method, w, tx)
    assert w.shape == (D,) and scores.shape == (N,)
    assert set(np.unique(labels)) <= {-1, 1}
    assert np.mean(labels == y) > 0.9


def test_logistic_scores_are_probabilities(data):
    y, tx = data
    w, _ = fit("logistic_regression", y, tx, gamma=0.5)
    scores = predict_scores("logistic_regression", w, tx)
    assert np.all((scores >= 0) & (scores <= 1))
    np.testing.assert_allclose(scores, impl.sigmoid(tx @ w))
    np.testing.assert_array_equal(
        predict("logistic_regression", w, tx, threshold=0.5),
        np.where(scores >= 0.5, 1, -1),
    )


def test_unknown_method_raises(data):
    y, tx = data
    with pytest.raises(ValueError):
        fit("lasso", y, tx)
    with pytest.raises(ValueError):
        predict_scores("lasso", np.zeros(D), tx)
