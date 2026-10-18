"""Tests for src/models.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import implementations as impl  # noqa: E402
from src.extra_implementations import reg_logistic_newton  # noqa: E402
from src.models import (  # noqa: E402
    EXTRA_METHODS,
    LOGISTIC_OPTIMIZERS,
    METHODS,
    class_weights,
    default_threshold,
    fit,
    predict,
    predict_scores,
    to_01,
    to_pm1,
    undersample_indices,
    weighted_logistic_gradient,
    weighted_logistic_loss,
    weighted_logistic_regression,
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


def imbalanced(n=400, seed=0):
    """Noisy labels in {-1, 1} with ~10% positives, bias column + 3 features."""
    rng = np.random.default_rng(seed)
    tx = np.c_[np.ones(n), rng.normal(size=(n, D - 1))]
    z = tx @ np.array([-1.8, 1.0, -0.5, 0.5]) + rng.normal(size=n)
    return np.where(z > 0, 1, -1), tx


def test_weighted_logistic_with_weight_one_is_reg_logistic():
    y, tx = imbalanced()
    w0 = np.zeros(D)
    w, loss = weighted_logistic_regression(to_01(y), tx, 0.01, w0, 50, 0.5, 1.0)
    w_ref, loss_ref = impl.reg_logistic_regression(to_01(y), tx, 0.01, w0, 50, 0.5)
    np.testing.assert_allclose(w, w_ref, rtol=1e-12)
    assert np.isclose(loss, loss_ref)


def test_weighted_logistic_gradient_finite_differences():
    y, tx = imbalanced()
    y01 = to_01(y)
    s = class_weights(y01, 7.0)
    w = np.array([0.3, -0.2, 0.1, 0.5])
    eps = 1e-6
    numeric = [
        (
            weighted_logistic_loss(y01, tx, w + eps * e, s)
            - weighted_logistic_loss(y01, tx, w - eps * e, s)
        )
        / (2 * eps)
        for e in np.eye(D)
    ]
    np.testing.assert_allclose(
        weighted_logistic_gradient(y01, tx, w, s), numeric, rtol=1e-6, atol=1e-9
    )


def test_class_weights_balance_the_weighted_predictions():
    # at the minimum, the bias component of the gradient is 0:
    # sum_n s_n sigmoid(z_n) = sum_n s_n y_n
    y, tx = imbalanced()
    y01 = to_01(y)
    kwargs = dict(gamma=0.5, max_iters=2000)
    w1, _ = fit("logistic_regression", y, tx, **kwargs)
    w9, _ = fit("logistic_regression", y, tx, pos_weight=9.0, **kwargs)
    p1 = predict_scores("logistic_regression", w1, tx)
    p9 = predict_scores("logistic_regression", w9, tx)
    s = class_weights(y01, 9.0)
    assert np.isclose(p1.mean(), y01.mean(), atol=1e-3)
    assert np.isclose(
        np.sum(s * p9) / np.sum(s), np.sum(s * y01) / np.sum(s), atol=1e-3
    )
    assert p9[y == 1].mean() > p1[y == 1].mean()


def test_pos_weight_only_for_logistic_methods(data):
    y, tx = data
    with pytest.raises(ValueError):
        fit("ridge_regression", y, tx, pos_weight=2.0)


def test_undersample_indices():
    y = np.where(np.arange(1000) % 10 == 0, 1, -1)
    rows = undersample_indices(y, 3, seed=0)
    assert np.sum(y[rows] == 1) == 100 and np.sum(y[rows] == -1) == 300
    assert len(np.unique(rows)) == len(rows) and np.all(np.diff(rows) > 0)
    np.testing.assert_array_equal(rows, undersample_indices(y, 3, seed=0))
    assert not np.array_equal(rows, undersample_indices(y, 3, seed=1))
    # more negatives asked than available: all the rows
    np.testing.assert_array_equal(undersample_indices(y, 100, seed=0), np.arange(1000))


def test_fit_with_undersampling_trains_on_the_subset():
    y, tx = imbalanced()
    rows = undersample_indices(y, 1, seed=4)
    w, _ = fit("ridge_regression", y, tx, lambda_=0.01, neg_per_pos=1, seed=4)
    np.testing.assert_array_equal(w, impl.ridge_regression(y[rows], tx[rows], 0.01)[0])


def test_unknown_method_raises(data):
    y, tx = data
    with pytest.raises(ValueError):
        fit("lasso", y, tx)
    with pytest.raises(ValueError):
        predict_scores("lasso", np.zeros(D), tx)


# small trees for the fake data (extra options of fit)
TREE_OPTIONS = dict(n_trees=10, max_depth=4, min_samples_leaf=5)


@pytest.mark.parametrize("method", EXTRA_METHODS)
def test_extra_methods_classify_separable_data(method, data):
    y, tx = data
    options = {} if method in LOGISTIC_OPTIMIZERS else TREE_OPTIONS
    w, loss = fit(method, y, tx, lambda_=1e-4, max_iters=300, seed=1, **options)
    scores = predict_scores(method, w, tx)
    labels = predict(method, w, tx)
    assert scores.shape == (N,) and np.all((scores >= 0) & (scores <= 1))
    assert set(np.unique(labels)) <= {-1, 1} and np.ndim(loss) == 0
    assert np.mean(labels == y) > 0.9
    assert default_threshold(method) == 0.5
    if method in LOGISTIC_OPTIMIZERS:
        assert w.shape == (D,)


def test_fit_extra_optimizer_uses_01_labels(data):
    y, tx = data
    w, loss = fit("reg_logistic_newton", y, tx, lambda_=0.1, max_iters=20)
    w_ref, loss_ref = reg_logistic_newton(to_01(y), tx, 0.1, np.zeros(D), 20)
    np.testing.assert_array_equal(w, w_ref)
    assert loss == loss_ref


def test_options_only_for_extra_methods(data):
    y, tx = data
    with pytest.raises(ValueError):
        fit("ridge_regression", y, tx, n_trees=10)
    # an option of another extra method: rejected by the method itself
    with pytest.raises(TypeError):
        fit("reg_logistic_sophia", y, tx, n_trees=10)
    with pytest.raises(ValueError):
        fit("random_forest", y, tx, pos_weight=2.0, **TREE_OPTIONS)
