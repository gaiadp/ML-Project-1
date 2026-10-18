"""Tests for src/extra_implementations (small fake data, a few seconds)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import implementations as impl  # noqa: E402
from src.extra_implementations import (  # noqa: E402
    gradient_boosting,
    gradient_boosting_predict_proba,
    logistic_boosting,
    logistic_boosting_predict_proba,
    random_forest,
    random_forest_predict_proba,
    reg_logistic_adaptive_gd,
    reg_logistic_newton,
    reg_logistic_sophia,
    reg_logistic_unixgrad,
)
from src.extra_implementations.logistic_objective import (  # noqa: E402
    diag_hessian,
    hessian,
    penalized_loss,
)
from src.extra_implementations.trees import (  # noqa: E402
    apply_bins,
    best_split,
    fit_bins,
    log_loss,
)

LAMBDA = 1e-3


@pytest.fixture
def logistic_data():
    """Noisy labels in {0, 1} (~10% positives), bias column + 4 features."""
    rng = np.random.default_rng(0)
    tx = np.c_[np.ones(1000), rng.normal(size=(1000, 4))]
    z = tx @ np.array([-2.0, 1.0, -0.5, 0.5, 0.0]) + rng.normal(size=1000)
    return (z > 0).astype(float), tx


@pytest.fixture
def minimum(logistic_data):
    """Minimum of the objective, by Newton's method."""
    y, tx = logistic_data
    w, _ = reg_logistic_newton(y, tx, LAMBDA, np.zeros(5), 50)
    return w, penalized_loss(y, tx, w, LAMBDA)


# the 4 optimizers with settings that converge on logistic_data
OPTIMIZERS = {
    "adaptive_gd": lambda y, tx, w0, k: reg_logistic_adaptive_gd(
        y, tx, LAMBDA, w0, k, 1.0
    ),
    "newton": lambda y, tx, w0, k: reg_logistic_newton(y, tx, LAMBDA, w0, k),
    "sophia": lambda y, tx, w0, k: reg_logistic_sophia(
        y, tx, LAMBDA, w0, k, 0.05, batch_size=10**9, hess_every=1
    ),
    "unixgrad": lambda y, tx, w0, k: reg_logistic_unixgrad(
        y, tx, LAMBDA, w0, k, radius=10.0
    ),
}
ITERS = {"adaptive_gd": 100, "newton": 20, "sophia": 2000, "unixgrad": 500}


def test_hessian_matches_finite_differences(logistic_data):
    y, tx = logistic_data
    w = np.array([-1.0, 0.5, -0.2, 0.3, 0.1])
    eps = 1e-6
    numeric = np.array(
        [
            (
                impl.penalized_logistic_regression(y, tx, w + eps * e, LAMBDA)
                - impl.penalized_logistic_regression(y, tx, w - eps * e, LAMBDA)
            )
            / (2 * eps)
            for e in np.eye(5)
        ]
    )
    h = hessian(tx, w, LAMBDA)
    np.testing.assert_allclose(h, numeric, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(diag_hessian(tx, w, LAMBDA), np.diag(h))


def test_newton_matches_long_gradient_descent(logistic_data, minimum):
    y, tx = logistic_data
    w_gd, _ = impl.reg_logistic_regression(y, tx, LAMBDA, np.zeros(5), 20000, 0.5)
    np.testing.assert_allclose(minimum[0], w_gd, atol=1e-5)


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_optimizers_reach_the_minimum(name, logistic_data, minimum):
    y, tx = logistic_data
    w, _ = OPTIMIZERS[name](y, tx, np.zeros(5), ITERS[name])
    assert penalized_loss(y, tx, w, LAMBDA) - minimum[1] < 1e-8


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_optimizers_follow_the_conventions(name, logistic_data):
    y, tx = logistic_data
    w0 = np.full(5, 0.1)
    # max_iters = 0: initial_w and its loss
    w, loss = OPTIMIZERS[name](y, tx, w0, 0)
    np.testing.assert_array_equal(w, w0)
    assert loss == impl.calculate_loss(y, tx, w0) and np.ndim(loss) == 0
    # the loss has no penalty term; initial_w is not modified
    w, loss = OPTIMIZERS[name](y, tx, w0, 5)
    assert w.shape == (5,) and np.isclose(loss, impl.calculate_loss(y, tx, w))
    np.testing.assert_array_equal(w0, np.full(5, 0.1))


def test_adaptive_gd_decreases_the_objective(logistic_data):
    y, tx = logistic_data
    f = []
    for k in range(12):
        w, _ = reg_logistic_adaptive_gd(y, tx, LAMBDA, np.zeros(5), k)
        f.append(penalized_loss(y, tx, w, LAMBDA))
    assert all(b <= a for a, b in zip(f, f[1:]))


def test_sophia_moves_each_coordinate_at_most_gamma(logistic_data):
    y, tx = logistic_data
    w0 = np.zeros(5)
    w1, _ = reg_logistic_sophia(y, tx, LAMBDA, w0, 1, 0.05, batch_size=64, seed=0)
    w5, _ = reg_logistic_sophia(y, tx, LAMBDA, w0, 5, 0.05, batch_size=64, seed=0)
    assert np.all(np.abs(w1 - w0) <= 0.05 + 1e-12)
    assert np.all(np.abs(w5 - w0) <= 5 * 0.05 + 1e-12)


def test_sophia_seed_is_reproducible(logistic_data):
    y, tx = logistic_data

    def run(seed):
        w, _ = reg_logistic_sophia(
            y, tx, LAMBDA, np.zeros(5), 50, 0.05, batch_size=32, seed=seed
        )
        return w

    np.testing.assert_array_equal(run(3), run(3))
    assert not np.array_equal(run(3), run(4))


def test_unixgrad_stays_in_the_ball(logistic_data):
    y, tx = logistic_data
    w, _ = reg_logistic_unixgrad(y, tx, LAMBDA, np.zeros(5), 50, radius=0.5)
    assert np.linalg.norm(w) <= 0.5 + 1e-12


def test_bins():
    rng = np.random.default_rng(0)
    x = np.c_[np.ones(500), rng.normal(size=500), rng.integers(0, 2, 500)]
    edges = fit_bins(x, n_bins=8)
    assert [len(e) for e in edges] == [0, 7, 1]
    xb = apply_bins(x, edges)
    assert xb.dtype == np.uint8 and xb.max() < 8
    np.testing.assert_array_equal(xb[:, 0], 0)  # constant column: one bin
    np.testing.assert_array_equal(xb[:, 2], x[:, 2])  # binary column: 0 / 1
    # values outside the training range go to the first / last bin
    out = apply_bins(np.array([[1.0, -100.0, 0.0], [1.0, 100.0, 1.0]]), edges)
    np.testing.assert_array_equal(out[:, 1], [0, 7])


def test_best_split_finds_the_threshold():
    xb = np.repeat(np.arange(10), 20).astype(np.uint8)[:, None]
    y = (xb[:, 0] >= 6).astype(float)
    rows, features = np.arange(200), np.array([0])
    assert best_split(xb, rows, features, y, None, 10, 0.0, 1) == (0, 5)
    # no split leaves 150 rows on both sides of 200
    assert best_split(xb, rows, features, y, None, 10, 0.0, 150) is None


def xor_data(n, seed):
    """Labels in {0, 1}: 1 iff x1 and x2 have the same sign; bias + 3 features."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 3))
    return (x[:, 0] * x[:, 1] > 0).astype(float), np.c_[np.ones(n), x]


TREES = {
    "random_forest": (
        lambda y, tx, seed: random_forest(
            y, tx, n_trees=20, max_depth=8, min_samples_leaf=5, seed=seed
        ),
        random_forest_predict_proba,
    ),
    "gradient_boosting": (
        lambda y, tx, seed: gradient_boosting(
            y,
            tx,
            n_trees=30,
            learning_rate=0.3,
            max_depth=3,
            min_samples_leaf=10,
            colsample=1.0,
            seed=seed,
        ),
        gradient_boosting_predict_proba,
    ),
}


@pytest.mark.parametrize("name", TREES)
def test_trees_learn_xor(name):
    y, tx = xor_data(3000, seed=0)
    y_test, tx_test = xor_data(2000, seed=1)
    train, predict_proba = TREES[name]
    model, loss = train(y, tx, 0)
    p = predict_proba(model, tx_test)
    assert p.shape == (2000,) and np.all((p >= 0) & (p <= 1))
    assert np.mean((p >= 0.5) == y_test) > 0.9 and np.ndim(loss) == 0
    # a linear model cannot separate a XOR
    w, _ = impl.logistic_regression(y, tx, np.zeros(4), 500, 0.5)
    assert np.mean((impl.sigmoid(tx_test @ w) >= 0.5) == y_test) < 0.7
    # same seed, same model
    np.testing.assert_array_equal(p, predict_proba(train(y, tx, 0)[0], tx_test))


def test_trees_without_splits_predict_the_prevalence():
    y, tx = xor_data(500, seed=0)
    forest, _ = random_forest(y, tx, n_trees=3, max_depth=0, subsample=1.0)
    np.testing.assert_allclose(random_forest_predict_proba(forest, tx), y.mean())
    boosting, _ = gradient_boosting(y, tx, n_trees=0)
    np.testing.assert_allclose(gradient_boosting_predict_proba(boosting, tx), y.mean())


def test_boosting_loss_decreases_with_more_trees():
    y, tx = xor_data(1000, seed=0)
    losses = []
    for n_trees in (1, 10, 50):
        _, loss = gradient_boosting(y, tx, n_trees=n_trees, max_depth=3, seed=0)
        losses.append(loss)
    assert losses[0] > losses[1] > losses[2]


def test_logistic_boosting_without_trees_is_newton(logistic_data, minimum):
    y, tx = logistic_data
    model, loss = logistic_boosting(y, tx, LAMBDA, np.zeros(5), 50, n_trees=0)
    np.testing.assert_array_equal(model["w"], minimum[0])
    np.testing.assert_array_equal(
        logistic_boosting_predict_proba(model, tx), impl.sigmoid(tx @ minimum[0])
    )
    assert np.isclose(loss, impl.calculate_loss(y, tx, minimum[0]))


def interaction_data(n, seed):
    """Labels in {0, 1}: a linear effect of x1 plus an interaction of x2 and x3."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 3))
    z = 3 * x[:, 0] + 3 * np.sign(x[:, 1] * x[:, 2]) + rng.logistic(size=n)
    return (z > 0).astype(float), np.c_[np.ones(n), x]


def test_logistic_boosting_learns_what_the_linear_model_misses():
    y, tx = interaction_data(3000, seed=0)
    y_test, tx_test = interaction_data(2000, seed=1)
    scores = []
    for n_trees in (0, 50):
        model, _ = logistic_boosting(
            y,
            tx,
            LAMBDA,
            np.zeros(4),
            50,
            n_trees=n_trees,
            learning_rate=0.3,
            min_samples_leaf=10,
            colsample=1.0,
            seed=0,
        )
        p = logistic_boosting_predict_proba(model, tx_test)
        scores.append((log_loss(y_test, p), np.mean((p >= 0.5) == y_test)))
    (loss_linear, acc_linear), (loss_hybrid, acc_hybrid) = scores
    assert loss_hybrid < loss_linear - 0.1
    assert acc_hybrid > 0.8 and acc_hybrid > acc_linear + 0.15
