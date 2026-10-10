"""Tests for the statistics helpers of experiments/eda.py (small fake data)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eda import (  # noqa: E402
    best_rule_f1,
    determination,
    discretize,
    log_odds_fit,
    mutual_information,
    pairwise_correlation,
    shift_distance,
)


def test_mutual_information():
    a = np.array([0, 0, 1, 1] * 25)
    b = np.array([0, 1] * 50)  # every pair (a, b) equally often: independent
    np.testing.assert_allclose(mutual_information(a, a), 1.0)  # H(a) = 1 bit
    np.testing.assert_allclose(mutual_information(a, b), 0.0, atol=1e-12)


def test_discretize():
    col = np.array([3.0, 1.0, np.nan, 3.0])
    codes, positions, n_values = discretize(col, "ordinal")
    np.testing.assert_array_equal(codes, [1, 0, -1, 1])
    np.testing.assert_array_equal(positions, [1.0, 3.0])
    assert n_values == 2
    # a continuous column with many values is cut in deciles
    codes, positions, n_values = discretize(np.arange(1000.0), "continuous")
    assert n_values == 1000 and len(positions) == 10
    np.testing.assert_array_equal(np.bincount(codes), [100] * 10)


def test_determination():
    a = np.array([0, 1, 2, 3] * 10)
    b = a // 2  # a function of a
    u_b_given_a, u_a_given_b = determination(a, b, 4, 2)
    np.testing.assert_allclose([u_b_given_a, u_a_given_b], [1.0, 0.5])


def test_pairwise_correlation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    x[:, 2] = 2 * x[:, 0] + 1
    x[:20, 1] = np.nan
    corr, overlap = pairwise_correlation(x)
    answered = ~np.isnan(x[:, 1])
    expected = np.corrcoef(x[answered, 0], x[answered, 1])[0, 1]
    np.testing.assert_allclose(corr[0, 1], expected)
    np.testing.assert_allclose(corr[0, 2], 1.0)
    assert overlap[0, 1] == 180 and overlap[0, 2] == 200


def test_best_rule_f1():
    codes = np.repeat([0, 1], 50)
    y01 = np.zeros(100)
    y01[60:] = 1.0  # 40 of the 50 rows of code 1 are positive
    # predict MICHD for code 1: TP 40, 50 predicted, 40 positives
    np.testing.assert_allclose(best_rule_f1(codes, y01), 80 / 90)


def test_log_odds_fit_of_a_linear_column():
    # prevalence with log-odds exactly linear in the value: R^2 of the line = 1
    values = np.arange(5.0)
    p = 1 / (1 + np.exp(-(values - 2)))
    n = 10_000
    codes = np.repeat(np.arange(5), n)
    y01 = np.concatenate([(np.arange(n) < round(q * n)).astype(float) for q in p])
    r2_line, _, signal, n_groups = log_odds_fit(codes, values, y01)
    assert n_groups == 5 and signal > 100
    np.testing.assert_allclose(r2_line, 1.0, atol=1e-3)


def test_shift_distance():
    a = np.array([1.0, 1.0, 2.0, 2.0])
    b = np.array([1.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(shift_distance(a, b, "binary"), 0.25)
    np.testing.assert_allclose(shift_distance(a, a, "binary"), 0.0)
    # NaN is a value of its own, and so is a value never seen in a
    c = np.array([1.0, np.nan, 2.0, 3.0])
    np.testing.assert_allclose(shift_distance(a, c, "nominal"), 0.5)
