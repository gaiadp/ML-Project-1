"""Tests for stratified_split (src/cross_validation.py) and src/pipeline.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cross_validation import stratified_split  # noqa: E402
from src.features import add_bias  # noqa: E402
from src.pipeline import (  # noqa: E402
    DEFAULT_CONFIG,
    V0_CONFIG,
    apply_preprocess,
    fit_preprocess,
)
from src.preprocessing import (  # noqa: E402
    apply_impute,
    apply_standardize,
    codes_to_nan,
    drop_high_missing_columns,
    fit_impute_median,
    fit_standardize,
)


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
    state = fit_preprocess(x_train, None, names, V0_CONFIG)
    np.testing.assert_array_equal(state["cols"], [0, 1])
    np.testing.assert_allclose(state["fill"], [2.0, 3000.0])
    assert state["out_names"] == ["bias", "GENHLTH", "_BMI5"]

    tx_train = apply_preprocess(x_train, names, state)
    assert tx_train.shape == (4, 3)
    np.testing.assert_array_equal(tx_train[:, 0], 1.0)
    np.testing.assert_allclose(tx_train[:, 1:].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(tx_train[:, 1:].std(axis=0), 1.0)

    x_test = np.array([[7.0, 100000.0, 1.0], [np.nan, np.nan, np.nan]])
    tx_test = apply_preprocess(x_test, names, state)
    assert not np.isnan(tx_test).any()
    np.testing.assert_allclose(
        tx_test[1, 1:], (state["fill"] - state["mean"]) / state["std"]
    )
    np.testing.assert_allclose(tx_test[0, 1], tx_test[1, 1])


def first_pipeline(x_train, x, names):
    """Pipeline v0 as it was written before the config, step by step."""
    x_train = codes_to_nan(x_train, names)
    keep = drop_high_missing_columns(x_train, 0.5)
    medians = fit_impute_median(x_train[:, keep])
    mean, std = fit_standardize(apply_impute(x_train[:, keep], medians))
    x = apply_impute(codes_to_nan(x, names)[:, keep], medians)
    return add_bias(apply_standardize(x, mean, std))


def test_v0_config_gives_the_first_pipeline():
    rng = np.random.default_rng(0)
    n = 300
    names = ["SEQNO", "GENHLTH", "PHYSHLTH", "_BMI5", "SEX", "INSULIN"]
    bmi = rng.normal(2700, 500, n)
    bmi[rng.random(n) < 0.1] = np.nan
    x = np.column_stack(
        [
            np.arange(n) + 2015000000.0,
            rng.choice([1, 2, 3, 4, 5, 7, 9], n),
            rng.choice([1, 2, 5, 15, 30, 77, 88, 99], n),
            bmi,
            rng.choice([1, 2], n),
            np.where(rng.random(n) < 0.8, np.nan, rng.choice([1, 2, 9], n)),
        ]
    )
    state = fit_preprocess(x[:200], None, names, V0_CONFIG)
    expected = first_pipeline(x[:200], x[200:], names)
    np.testing.assert_array_equal(apply_preprocess(x[200:], names, state), expected)


def fake_brfss(n=200):
    """Small fake dataset with the columns that each B step changes.

    Every other row is positive. INSULIN is answered by 20 positive rows only
    (informative missingness), MOSTLY_NAN by 10 positive and 10 negative rows.
    """
    i = np.arange(n)
    y = np.where(i % 2 == 0, 1, -1)
    asked = (i < 40) & (i % 2 == 0)
    no_exercise = i % 3 == 0
    columns = {
        "SEQNO": i + 1000.0,  # B2: id
        "HAREHAB1": np.where(asked, 1.0, np.nan),  # B2: label leakage
        "WEIGHT2": 150.0 + i % 50,  # B3: raw, WTKG3 is its converted version
        "WTKG3": 70.0 + i % 30,
        "CHECKUP1": np.where(i % 10 == 0, 8.0, 1.0 + i % 4),  # B3: 8 = never
        "EXERANY2": np.where(no_exercise, 2.0, 1.0),  # B4 gate, B8 binary
        "PAFREQ1_": np.where(no_exercise, np.nan, 3.0),  # B4: 0 when not asked
        "SEX": 1.0 + (i // 2) % 2,  # B8 binary
        "MARITAL": np.where(i % 11 == 0, 9.0, 1.0 + i % 6),  # B7 nominal
        "GENHLTH": np.where(i % 7 == 0, 7.0, 1.0 + i % 5),
        "INSULIN": np.where(asked, 1.0, np.nan),  # B5/B6: 90% NaN, informative
        "MOSTLY_NAN": np.where(i < 20, 1.0 + i % 2, np.nan),  # B5: 90% NaN
    }
    names = list(columns)
    return np.column_stack([columns[name] for name in names]), y, names


def test_full_pipeline():
    x, y, names = fake_brfss()
    state = fit_preprocess(x, y, names, dict(DEFAULT_CONFIG, min_missing_rows=5))
    out = state["out_names"]
    assert out[0] == "bias"
    for name in ["SEQNO", "HAREHAB1", "WEIGHT2", "MOSTLY_NAN"]:
        assert name not in out
    for name in ["WTKG3", "PAFREQ1_", "INSULIN", "INSULIN_isnan"]:
        assert name in out
    assert len(out) == len(set(out))

    tx = apply_preprocess(x, names, state)
    assert tx.shape == (len(x), len(out))
    np.testing.assert_allclose(tx[:, 1:].mean(axis=0), 0.0, atol=1e-12)
    # a person with every answer missing still gets a complete row
    tx_test = apply_preprocess(np.full((3, len(names)), np.nan), names, state)
    assert tx_test.shape == (3, len(out)) and np.isfinite(tx_test).all()


def test_labels_are_needed_only_for_the_missing_steps():
    x, y, names = fake_brfss()
    with pytest.raises(ValueError):
        fit_preprocess(x, None, names, DEFAULT_CONFIG)
    no_labels = dict(DEFAULT_CONFIG, keep_informative=False, missing_indicators=False)
    fit_preprocess(x, None, names, no_labels)


def test_each_step_can_be_switched_off():
    x, y, names = fake_brfss()
    base = dict(DEFAULT_CONFIG, min_missing_rows=5)
    variants = [dict(base, **{key: False}) for key, v in base.items() if v is True]
    variants += [dict(base, impute="median"), dict(base, max_missing_frac=1.0)]
    for config in variants:
        state = fit_preprocess(x, y, names, config)
        tx = apply_preprocess(x[:50], names, state)
        assert tx.shape == (50, len(state["out_names"]))
        assert np.isfinite(tx).all()
