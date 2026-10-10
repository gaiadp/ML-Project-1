"""Tests for stratified_split (src/cross_validation.py) and src/pipeline.py."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cross_validation import stratified_split  # noqa: E402
from src.pipeline import apply_preprocess, fit_preprocess  # noqa: E402


def test_stratified_split():
    y = np.where(np.arange(1000) % 10 == 0, 1, -1)
    tr, va = stratified_split(y, val_ratio=0.2, seed=0)
    assert len(tr) == 800 and len(va) == 200
    np.testing.assert_array_equal(np.sort(np.concatenate([tr, va])), np.arange(1000))
    assert np.sum(y[va] == 1) == 20 and np.sum(y[tr] == 1) == 80
    tr2, va2 = stratified_split(y, val_ratio=0.2, seed=0)
    np.testing.assert_array_equal(va, va2)
    assert not np.array_equal(va, stratified_split(y, val_ratio=0.2, seed=1)[1])


def test_pipeline_uses_training_statistics_only():
    names = ["GENHLTH", "_BMI5", "MOSTLY_NAN"]
    x_train = np.array(
        [
            [1.0, 2000.0, np.nan],
            [2.0, 3000.0, np.nan],
            [9.0, np.nan, 5.0],
            [4.0, 4000.0, np.nan],
        ]
    )
    params = fit_preprocess(x_train, names, max_missing_frac=0.5)
    np.testing.assert_array_equal(params["keep"], [True, True, False])
    np.testing.assert_allclose(params["medians"], [2.0, 3000.0])

    tx_train = apply_preprocess(x_train, names, params)
    assert tx_train.shape == (4, 3)
    np.testing.assert_array_equal(tx_train[:, 0], 1.0)
    np.testing.assert_allclose(tx_train[:, 1:].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(tx_train[:, 1:].std(axis=0), 1.0)

    x_test = np.array([[7.0, 100000.0, 1.0], [np.nan, np.nan, np.nan]])
    tx_test = apply_preprocess(x_test, names, params)
    assert not np.isnan(tx_test).any()
    np.testing.assert_allclose(
        tx_test[1, 1:], (params["medians"] - params["mean"]) / params["std"]
    )
    np.testing.assert_allclose(tx_test[0, 1], tx_test[1, 1])
