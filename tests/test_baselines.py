"""Tests for the helpers of experiments/baselines.py (small fake data)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.baselines import (  # noqa: E402
    best_rows,
    prevalence_scores,
    rule_keys,
    trivial_baselines,
)


def test_rule_keys():
    columns = np.array([[1.0, 2.0], [1.0, np.nan], [2.0, 2.0], [1.0, 2.0]])
    keys = rule_keys(columns)
    # same combination, same key; NaN is a value of its own
    assert keys[0] == keys[3] and len(set(keys)) == 3
    np.testing.assert_array_equal(rule_keys(np.empty((4, 0))), 0)


def test_prevalence_scores():
    keys_train = np.array([1, 1, 2, 2, 2])
    y_train = np.array([1, -1, 1, 1, -1])
    scores = prevalence_scores(keys_train, y_train, np.array([1, 2, 7]))
    # key 7 is not in the training rows: overall prevalence 3 / 5
    np.testing.assert_allclose(scores, [1 / 2, 2 / 3, 3 / 5])


def test_trivial_baselines():
    rng = np.random.default_rng(0)
    n = 2000
    names = ["_AGEG5YR", "SEX", "GENHLTH"]
    x = np.c_[rng.integers(1, 14, n), rng.integers(1, 3, n), rng.integers(1, 6, n)]
    # MICHD only for the 3 oldest age groups: the age rule is perfect
    y = np.where(x[:, 0] >= 11, 1, -1)
    out = trivial_baselines(x.astype(float), y, names)
    p = np.mean(y == 1)
    assert np.isclose(out["everyone MICHD"][0], 2 * p / (1 + p), atol=0.01)
    assert out["age group"] == (1.0, 0.0)
    assert out["age group x sex"][0] == 1.0


def test_best_rows():
    rows = [
        dict(label="v0", f1_mean=0.40, finite=True),
        dict(label="v0", f1_mean=0.42, finite=True),
        dict(label="v1", f1_mean=0.50, finite=False),  # diverged: ignored
        dict(label="v1", f1_mean=0.41, finite=True),
    ]
    best = best_rows(rows)
    assert best["v0"]["f1_mean"] == 0.42 and best["v1"]["f1_mean"] == 0.41
