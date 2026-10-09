"""Our own sanity tests for implementations.py.

The official grading tests live at
https://github.com/epfml/ML_course/tree/main/projects/project1/grading_tests
Run them with:  pytest --github_link <repo-url> .

Run ours from the project root with:  pytest tests/
"""

import os
import sys

import numpy as np
import pytest

# implementations.py lives in the project root, one level above this file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from implementations import (  # noqa: E402
    calculate_gradient,
    calculate_loss,
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    penalized_logistic_regression,
    reg_logistic_regression,
    ridge_regression,
    sigmoid,
)

N, D = 200, 5


# ---------------------------------------------------------------------------
# Fixtures and reference losses (written independently of implementations.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_data():
    """Well-conditioned linear regression problem: bias column + 4 features."""
    rng = np.random.default_rng(0)
    tx = np.c_[np.ones(N), rng.normal(size=(N, D - 1))]
    w_true = rng.normal(size=D)
    y = tx @ w_true + 0.1 * rng.normal(size=N)
    return y, tx


@pytest.fixture
def classification_data():
    """Binary classification problem with labels in {0, 1}."""
    rng = np.random.default_rng(1)
    tx = np.c_[np.ones(N), rng.normal(size=(N, D - 1))]
    w_true = rng.normal(size=D)
    p = 1 / (1 + np.exp(-tx @ w_true))
    y = (rng.random(N) < p).astype(float)
    return y, tx


def mse_loss(y, tx, w):
    """MSE with the 1/2 factor, as in the lecture notes."""
    return 0.5 * np.mean((y - tx @ w) ** 2)


def logistic_loss(y, tx, w):
    """Mean negative log-likelihood for y in {0, 1}."""
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def numerical_gradient(f, w, eps=1e-6):
    """Central finite-difference gradient of the scalar function f at w."""
    grad = np.zeros_like(w)
    for j in range(w.size):
        e = np.zeros_like(w)
        e[j] = eps
        grad[j] = (f(w + e) - f(w - e)) / (2 * eps)
    return grad


# ---------------------------------------------------------------------------
# 1. GD converges to the least squares solution
# ---------------------------------------------------------------------------


def test_gd_converges_to_least_squares(regression_data):
    y, tx = regression_data
    w_ls, loss_ls = least_squares(y, tx)
    # step size 1/L, with L the largest eigenvalue of the Hessian X^T X / N
    gamma = 1 / np.linalg.eigvalsh(tx.T @ tx / N).max()
    w_gd, loss_gd = mean_squared_error_gd(y, tx, np.zeros(D), 2000, gamma)
    np.testing.assert_allclose(w_gd, w_ls, atol=1e-6)
    np.testing.assert_allclose(loss_gd, loss_ls, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Ridge with lambda = 0 is least squares
# ---------------------------------------------------------------------------


def test_ridge_lambda_zero_equals_least_squares(regression_data):
    y, tx = regression_data
    w_ls, loss_ls = least_squares(y, tx)
    w_ridge, loss_ridge = ridge_regression(y, tx, 0.0)
    np.testing.assert_allclose(w_ridge, w_ls, atol=1e-8)
    np.testing.assert_allclose(loss_ridge, loss_ls, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Regularized logistic regression with lambda = 0 is logistic regression
# ---------------------------------------------------------------------------


def test_reg_logistic_lambda_zero_equals_logistic(classification_data):
    y, tx = classification_data
    w0 = np.zeros(D)
    w_log, loss_log = logistic_regression(y, tx, w0, 100, 0.5)
    w_reg, loss_reg = reg_logistic_regression(y, tx, 0.0, w0, 100, 0.5)
    np.testing.assert_allclose(w_reg, w_log, atol=1e-12)
    np.testing.assert_allclose(loss_reg, loss_log, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Gradients checked with finite differences
# ---------------------------------------------------------------------------
# implementations.py does not expose the gradients, so we recover them from a
# single step: with max_iters=1 and gamma=1, w_1 = w_0 - grad(w_0).
# The loss at an arbitrary w is obtained with max_iters=0, which also checks
# that the returned loss is computed at the returned w.


def test_mse_gd_gradient(regression_data):
    y, tx = regression_data
    w0 = np.random.default_rng(2).normal(size=D)
    w1, _ = mean_squared_error_gd(y, tx, w0, 1, 1.0)
    grad_impl = w0 - w1
    grad_num = numerical_gradient(
        lambda w: mean_squared_error_gd(y, tx, w, 0, 1.0)[1], w0
    )
    np.testing.assert_allclose(grad_impl, grad_num, rtol=1e-5, atol=1e-7)


def test_mse_sgd_gradient(regression_data):
    # With a single data point, the stochastic gradient IS the full gradient
    y, tx = regression_data
    y1, tx1 = y[:1], tx[:1]
    w0 = np.random.default_rng(3).normal(size=D)
    w1, _ = mean_squared_error_sgd(y1, tx1, w0, 1, 1.0)
    grad_impl = w0 - w1
    grad_num = numerical_gradient(lambda w: mse_loss(y1, tx1, w), w0)
    np.testing.assert_allclose(grad_impl, grad_num, rtol=1e-5, atol=1e-7)


def test_logistic_gradient(classification_data):
    y, tx = classification_data
    w0 = np.random.default_rng(4).normal(size=D)
    w1, _ = logistic_regression(y, tx, w0, 1, 1.0)
    grad_impl = w0 - w1
    grad_num = numerical_gradient(
        lambda w: logistic_regression(y, tx, w, 0, 1.0)[1], w0
    )
    np.testing.assert_allclose(grad_impl, grad_num, rtol=1e-5, atol=1e-7)


def test_reg_logistic_gradient(classification_data):
    # The objective is loss + lambda * ||w||^2, so the gradient must include
    # 2 * lambda * w even though the returned loss does not include the penalty
    y, tx = classification_data
    lambda_ = 0.1
    w0 = np.random.default_rng(5).normal(size=D)
    w1, _ = reg_logistic_regression(y, tx, lambda_, w0, 1, 1.0)
    grad_impl = w0 - w1
    grad_num = numerical_gradient(
        lambda w: logistic_loss(y, tx, w) + lambda_ * w @ w, w0
    )
    np.testing.assert_allclose(grad_impl, grad_num, rtol=1e-5, atol=1e-7)


def test_mse_losses_match_reference_formula(regression_data):
    # max_iters=0 returns the loss at initial_w: compare with our own formula
    y, tx = regression_data
    w = np.random.default_rng(6).normal(size=D)
    np.testing.assert_allclose(
        mean_squared_error_gd(y, tx, w, 0, 0.1)[1], mse_loss(y, tx, w)
    )
    np.testing.assert_allclose(
        mean_squared_error_sgd(y, tx, w, 0, 0.1)[1], mse_loss(y, tx, w)
    )


def test_logistic_loss_matches_reference_formula(classification_data):
    y, tx = classification_data
    w = np.random.default_rng(6).normal(size=D)
    np.testing.assert_allclose(
        logistic_regression(y, tx, w, 0, 0.1)[1], logistic_loss(y, tx, w)
    )


# ---------------------------------------------------------------------------
# 5. Output shapes: w is 1D (D,), loss is a scalar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        "mean_squared_error_gd",
        "mean_squared_error_sgd",
        "least_squares",
        "ridge_regression",
        "logistic_regression",
        "reg_logistic_regression",
    ],
)
def test_output_shapes(method, classification_data):
    # {0, 1} labels are valid targets for the regression methods too
    y, tx = classification_data
    w0 = np.zeros(D)
    calls = {
        "mean_squared_error_gd": lambda: mean_squared_error_gd(y, tx, w0, 10, 0.1),
        "mean_squared_error_sgd": lambda: mean_squared_error_sgd(y, tx, w0, 10, 0.1),
        "least_squares": lambda: least_squares(y, tx),
        "ridge_regression": lambda: ridge_regression(y, tx, 0.1),
        "logistic_regression": lambda: logistic_regression(y, tx, w0, 10, 0.1),
        "reg_logistic_regression": lambda: reg_logistic_regression(
            y, tx, 0.1, w0, 10, 0.1
        ),
    }
    result = calls[method]()
    assert isinstance(result, tuple) and len(result) == 2, "must return (w, loss)"
    w, loss = result
    assert isinstance(w, np.ndarray) and w.shape == (D,)
    assert np.ndim(loss) == 0


# ---------------------------------------------------------------------------
# 6. Regularized methods return the loss WITHOUT the penalty term
# ---------------------------------------------------------------------------


def test_ridge_loss_excludes_penalty(regression_data):
    y, tx = regression_data
    lambda_ = 0.5
    w, loss = ridge_regression(y, tx, lambda_)
    np.testing.assert_allclose(loss, mse_loss(y, tx, w))
    assert not np.isclose(loss, mse_loss(y, tx, w) + lambda_ * w @ w)


def test_reg_logistic_loss_excludes_penalty(classification_data):
    y, tx = classification_data
    lambda_ = 0.5
    w, loss = reg_logistic_regression(y, tx, lambda_, np.zeros(D), 50, 0.5)
    np.testing.assert_allclose(loss, logistic_loss(y, tx, w))
    assert not np.isclose(loss, logistic_loss(y, tx, w) + lambda_ * w @ w)


# ---------------------------------------------------------------------------
# 7. Logistic regression: lab ex05 values, public grader values, numerical
#    stability, initial_w left untouched
# ---------------------------------------------------------------------------


def test_logistic_helpers_match_lab_ex05():
    # Same inputs as the doctests of lab ex05, with 1D arrays instead of (N, 1)
    np.testing.assert_allclose(sigmoid(np.array([0.1, 0.1])), [0.52497919] * 2)
    y = np.array([0.0, 1.0])
    tx = np.arange(4).reshape(2, 2)
    np.testing.assert_allclose(calculate_loss(y, tx, np.array([2.0, 3.0])), 1.52429481)
    tx = np.arange(6).reshape(2, 3)
    w = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(
        calculate_gradient(y, tx, w), [-0.10370763, 0.2067104, 0.51712843], rtol=1e-6
    )
    np.testing.assert_allclose(
        penalized_logistic_regression(y, tx, w, 0.1),
        [-0.08370763, 0.2467104, 0.57712843],
        rtol=1e-6,
    )
    # One GD step with gamma=0.1 gives the same w as the lab...
    w1, loss1 = logistic_regression(y, tx, w, 1, 0.1)
    np.testing.assert_allclose(w1, [0.11037076, 0.17932896, 0.24828716], rtol=1e-6)
    w1_reg, _ = reg_logistic_regression(y, tx, 0.1, w, 1, 0.1)
    np.testing.assert_allclose(w1_reg, [0.10837076, 0.17532896, 0.24228716], rtol=1e-6)
    # ...but the lab returned the loss at the OLD w (0.62137268), we return it at w1
    np.testing.assert_allclose(logistic_regression(y, tx, w, 0, 0.1)[1], 0.62137268)
    np.testing.assert_allclose(loss1, calculate_loss(y, tx, w1))


@pytest.fixture
def public_data():
    """Data of the public grading tests (projects/project1/grading_tests)."""
    y = (np.array([0.1, 0.3, 0.5]) > 0.2) * 1.0
    tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
    return y, tx


def test_logistic_public_values(public_data):
    y, tx = public_data
    w, loss = logistic_regression(y, tx, np.array([0.5, 1.0]), 2, 0.1)
    np.testing.assert_allclose(w, [0.378561, 0.801131], rtol=1e-4)
    np.testing.assert_allclose(loss, 1.348358, rtol=1e-4)


def test_reg_logistic_public_values(public_data):
    y, tx = public_data
    w, loss = reg_logistic_regression(y, tx, 1.0, np.array([0.5, 1.0]), 2, 0.1)
    np.testing.assert_allclose(w, [0.216062, 0.467747], rtol=1e-4)
    np.testing.assert_allclose(loss, 0.972165, rtol=1e-4)


def test_logistic_no_overflow_for_large_scores():
    # Scores z = tx @ w of +-1e4: the lab formulas overflow in exp(-z) and take
    # log(0). Overflow, division by zero and NaN are turned into errors here.
    y = np.array([0.0, 1.0, 0.0, 1.0])
    tx = np.array([[1.0, 1e4], [1.0, -1e4], [1.0, 5e3], [1.0, -5e3]])
    w0 = np.array([0.0, 1.0])
    with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
        probs = sigmoid(tx @ w0)
        w, loss = reg_logistic_regression(y, tx, 0.1, w0, 3, 1e-6)
        loss0 = logistic_regression(y, tx, w0, 0, 0.1)[1]
    np.testing.assert_allclose(probs, [1.0, 0.0, 1.0, 0.0])
    assert np.isfinite(loss) and np.all(np.isfinite(w))
    # Every point is misclassified with margin |z|, so each loss term is ~|z|
    np.testing.assert_allclose(loss0, 7500.0)


@pytest.mark.parametrize("max_iters", [0, 5])
def test_logistic_does_not_modify_initial_w(classification_data, max_iters):
    y, tx = classification_data
    w0 = np.random.default_rng(7).normal(size=D)
    w0_before = w0.copy()
    w, _ = logistic_regression(y, tx, w0, max_iters, 0.5)
    w_reg, _ = reg_logistic_regression(y, tx, 0.1, w0, max_iters, 0.5)
    np.testing.assert_array_equal(w0, w0_before)
    # The returned arrays are new objects, not the caller's initial_w
    assert w is not w0 and w_reg is not w0


def test_logistic_gd_decreases_loss(classification_data):
    y, tx = classification_data
    w0 = np.zeros(D)
    loss0 = logistic_regression(y, tx, w0, 0, 0.5)[1]
    loss100 = logistic_regression(y, tx, w0, 100, 0.5)[1]
    # With w = 0 every probability is 1/2, so the loss is log 2
    np.testing.assert_allclose(loss0, np.log(2))
    assert loss100 < loss0
