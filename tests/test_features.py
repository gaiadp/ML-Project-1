"""Tests for the feature generation helpers of src/features.py."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (  # noqa: E402
    centered_powers,
    fit_categories,
    one_hot_encode,
)


def test_one_hot_with_fitted_categories():
    x_train = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, np.nan])
    # 1 is the most frequent value (reference), 3 has fewer than 2 rows
    categories = fit_categories(x_train, min_rows=2)
    np.testing.assert_array_equal(categories, [2.0])
    # reference, rare, unseen values and NaN all get a row of zeros
    encoded, _ = one_hot_encode(np.array([1.0, 2.0, 3.0, 5.0, np.nan]), categories)
    np.testing.assert_array_equal(encoded[:, 0], [0, 1, 0, 0, 0])


def test_centered_powers():
    x = np.array([[1.0, 10.0], [3.0, 20.0]])
    center = np.array([2.0, 15.0])
    np.testing.assert_array_equal(centered_powers(x, center, 2), [[1, 25], [1, 25]])
    cubes = centered_powers(x, center, 3)[:, 2:]
    np.testing.assert_array_equal(cubes, [[-1, -125], [1, 125]])
    assert centered_powers(x, center, 1).shape == (2, 0)
