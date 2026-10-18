"""Tests for stratified_split (src/cross_validation.py) and src/pipeline.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cross_validation import stratified_split  # noqa: E402
from src.features import (  # noqa: E402
    add_bias,
    fit_one_hot_categories,
    one_hot_encode,
)
from src.pipeline import (  # noqa: E402
    DEFAULT_CONFIG,
    V0_CONFIG,
    V1_CONFIG,
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


def v0_data(n=300):
    rng = np.random.default_rng(0)
    names = ["SEQNO", "GENHLTH", "PHYSHLTH", "_BMI5", "SEX", "INSULIN", "EMPLOY1"]
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
            rng.choice([1, 2, 7, 9], n, p=[0.5, 0.2, 0.2, 0.1]),  # nominal
        ]
    )
    return x, names


def test_v0_config_without_one_hot_gives_the_first_pipeline():
    x, names = v0_data()
    state = fit_preprocess(x[:200], None, names, dict(V0_CONFIG, one_hot=False))
    expected = first_pipeline(x[:200], x[200:], names)
    np.testing.assert_array_equal(apply_preprocess(x[200:], names, state), expected)


def test_v0_config_one_hot_encodes_the_nominal_columns():
    x, names = v0_data()
    state = fit_preprocess(x[:200], None, names, dict(V0_CONFIG, one_hot_min_rows=5))
    # EMPLOY1: 1 most frequent (reference), 9 -> NaN -> median 1
    assert "EMPLOY1" not in state["out_names"]
    assert state["out_names"][-2:] == ["EMPLOY1=2", "EMPLOY1=7"]
    first = fit_preprocess(x[:200], None, names, dict(V0_CONFIG, one_hot=False))
    # the other columns are the ones of the first pipeline
    assert state["out_names"][:-2] == first["out_names"][:-1]
    tx = apply_preprocess(x[200:], names, state)
    np.testing.assert_allclose(
        tx[:, :-2], apply_preprocess(x[200:], names, first)[:, :-1]
    )


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


def test_fit_one_hot_categories():
    # 3 most frequent (reference), 7 rare
    v = np.array([3.0] * 5 + [1.0] * 3 + [2.0] * 2 + [7.0])
    np.testing.assert_array_equal(fit_one_hot_categories(v, 2), [1.0, 2.0])
    np.testing.assert_array_equal(fit_one_hot_categories(v, 3), [1.0])
    assert len(fit_one_hot_categories(np.full(4, 5.0), 1)) == 0
    # reference, rare and unseen values all get the all-zero row
    encoded, _ = one_hot_encode(np.array([1.0, 3.0, 7.0, 42.0]), [1.0, 2.0])
    np.testing.assert_array_equal(encoded, [[1, 0], [0, 0], [0, 0], [0, 0]])


def test_one_hot_step():
    x, y, names = fake_brfss()
    # nominal columns only (the ordinal ones: test_one_hot_ordinal_step)
    config = dict(
        DEFAULT_CONFIG, min_missing_rows=5, one_hot_min_rows=5, one_hot_ordinal=False
    )
    state = fit_preprocess(x, y, names, config)
    out = state["out_names"]
    # MARITAL 1..6 (9 -> NaN -> imputed with the mode 2): 2 is the reference
    assert "MARITAL" not in out
    marital = [name for name in out if name.startswith("MARITAL=")]
    assert marital == [f"MARITAL={v}" for v in (1, 3, 4, 5, 6)]
    # the other columns are unchanged, the one-hot columns come before the indicators
    assert "GENHLTH" in out and out[-1] == "INSULIN_isnan"

    j = names.index("MARITAL")
    x_test = np.repeat(x[:1], 4, axis=0)
    x_test[:, j] = [3.0, 2.0, 9.0, 42.0]  # a category, the reference, NaN, unseen
    tx = apply_preprocess(x_test, names, state)
    cols = [out.index(name) for name in marital]
    hot = tx[:, cols] > 0
    assert hot[0].tolist() == [False, True, False, False, False]
    # reference, NaN (imputed with the reference) and unseen value: all-zero row
    np.testing.assert_array_equal(tx[1], tx[2])
    np.testing.assert_array_equal(tx[1], tx[3])
    assert not hot[1].any()

    no_one_hot = fit_preprocess(x, y, names, dict(config, one_hot=False))
    assert "MARITAL" in no_one_hot["out_names"]
    assert len(out) == len(no_one_hot["out_names"]) - 1 + len(marital)


def test_binning_step():
    x, y, names = fake_brfss()
    physhlth = np.tile([88.0, 88.0, 3.0, 10.0, 20.0, 25.0, 30.0, 77.0], 25)
    x = np.column_stack([x, physhlth])
    names = names + ["PHYSHLTH"]
    config = dict(DEFAULT_CONFIG, min_missing_rows=5, one_hot_min_rows=5)
    out = fit_preprocess(x, y, names, config)["out_names"]
    # bins 0 (0 days, the reference: most frequent, and NaN imputed with it),
    # 1 (1-7), 2 (8-14), 3 (15-21), 4 (22-29), 5 (30)
    assert "PHYSHLTH" not in out
    assert [n for n in out if n.startswith("PHYSHLTH=")] == [
        f"PHYSHLTH={k}" for k in range(1, 6)
    ]
    out = fit_preprocess(x, y, names, dict(config, binning=False))["out_names"]
    assert "PHYSHLTH" in out and "PHYSHLTH=1" not in out


def test_one_hot_ordinal_step():
    x, y, names = fake_brfss()
    config = dict(DEFAULT_CONFIG, min_missing_rows=5, one_hot_min_rows=5)
    out = fit_preprocess(x, y, names, config)["out_names"]
    # CHECKUP1 (in ONE_HOT_ORDINAL): 1..4, 8 = never -> 5 (convert_units)
    assert "CHECKUP1" not in out
    checkup = [n for n in out if n.startswith("CHECKUP1=")]
    assert len(checkup) == 4  # 5 values, the most frequent is the reference
    # GENHLTH is ordinal but not in ONE_HOT_ORDINAL: one numeric column
    assert "GENHLTH" in out and not any(n.startswith("GENHLTH=") for n in out)
    out = fit_preprocess(x, y, names, dict(config, one_hot_ordinal=False))["out_names"]
    assert "CHECKUP1" in out and not any(n.startswith("CHECKUP1=") for n in out)


def test_drop_redundant_step():
    x, y, names = fake_brfss()
    pa = ["_PACAT1", "_PA150R2", "_PA300R2", "_PA30021", "_PAINDX1"]
    x = np.column_stack([x, np.tile([[1.0, 1.0, 1.0, 1.0, 1.0]], (len(x), 1))])
    x[1::2, -5:] = [4.0, 3.0, 3.0, 2.0, 2.0]  # inactive
    names = names + pa
    config = dict(DEFAULT_CONFIG, min_missing_rows=5, one_hot_min_rows=5)
    out = fit_preprocess(x, y, names, config)["out_names"]
    assert any(n.startswith("_PACAT1") for n in out)
    assert not any(n.startswith(tuple(pa[1:])) for n in out)
    out = fit_preprocess(x, y, names, dict(config, drop_redundant=False))["out_names"]
    assert all(any(n.startswith(name) for n in out) for name in pa)


def test_label_leakage_is_dropped_without_domain_drop():
    # HAREHAB1 is answered by positive rows only: its missingness is "informative"
    x, y, names = fake_brfss()
    config = dict(DEFAULT_CONFIG, min_missing_rows=5, domain_drop=False)
    out = fit_preprocess(x, y, names, config)["out_names"]
    assert "SEQNO" in out
    assert not any(n.startswith("HAREHAB1") for n in out)


def test_v1_config_selects_the_features():
    x, y, names = fake_brfss()
    seatbelt = 1.0 + np.arange(len(x)) % 5  # 1 always .. 5 never, in NOT_SELECTED
    x, names = np.column_stack([x, seatbelt]), names + ["SEATBELT"]
    config = dict(V1_CONFIG, min_missing_rows=5)
    state = fit_preprocess(x, y, names, config)
    out = state["out_names"]
    assert "SEATBELT" not in out
    # the selection replaces B5: MOSTLY_NAN (90% NaN, not informative, dropped by
    # DEFAULT_CONFIG) is kept; B6 still adds the missing indicators
    assert "MOSTLY_NAN" in out and "INSULIN_isnan" in out
    tx = apply_preprocess(x, names, state)
    assert tx.shape == (len(x), len(out)) and np.isfinite(tx).all()
    out = fit_preprocess(x, y, names, dict(config, select_features=False))["out_names"]
    assert "SEATBELT" in out


def test_labels_are_needed_only_for_the_missing_steps():
    x, y, names = fake_brfss()
    with pytest.raises(ValueError):
        fit_preprocess(x, None, names, DEFAULT_CONFIG)
    no_labels = dict(DEFAULT_CONFIG, keep_informative=False, missing_indicators=False)
    fit_preprocess(x, None, names, no_labels)


def test_expansion_step():
    x, y, names = fake_brfss()
    base = dict(DEFAULT_CONFIG, min_missing_rows=5, one_hot_min_rows=5)
    plain = fit_preprocess(x, y, names, base)
    config = dict(
        base,
        poly_degree=3,
        interaction_top=3,
        interaction_pairs=(("WTKG3", "SEX"), ("SEX", "WTKG3")),
    )
    state = fit_preprocess(x, y, names, config)
    out = state["out_names"]
    n_base = len(plain["out_names"])
    assert out[:n_base] == plain["out_names"]
    assert len(out) == len(set(out))
    # WTKG3 has 30 distinct values: squared and cubed; SEX is binary: no powers
    assert "WTKG3^2" in out and "WTKG3^3" in out
    assert not any(n.startswith("SEX^") for n in out)
    # 3 pairs of the top 3, plus WTKG3 * SEX given twice (in either order)
    products = [n for n in out if "*" in n]
    pair = [n for n in products if set(n.split("*")) == {"WTKG3", "SEX"}]
    assert len(products) in (3, 4) and len(pair) == 1

    tx = apply_preprocess(x, names, state)
    assert tx.shape == (len(x), len(out))
    np.testing.assert_allclose(tx[:, :n_base], apply_preprocess(x, names, plain))
    np.testing.assert_allclose(tx[:, 1:].mean(axis=0), 0.0, atol=1e-10)
    # a product is the standardized product of the two standardized columns
    i, j, k = out.index("WTKG3"), out.index("SEX"), out.index(pair[0])
    p = tx[:, i] * tx[:, j]
    np.testing.assert_allclose(tx[:, k], (p - p.mean()) / p.std(), atol=1e-10)
    # test rows use the training statistics
    tx_test = apply_preprocess(np.full((2, len(names)), np.nan), names, state)
    assert tx_test.shape == (2, len(out)) and np.isfinite(tx_test).all()


def test_expansion_errors():
    x, y, names = fake_brfss()
    no_labels = dict(DEFAULT_CONFIG, keep_informative=False, missing_indicators=False)
    with pytest.raises(ValueError):
        fit_preprocess(x, None, names, dict(no_labels, interaction_top=2))
    with pytest.raises(ValueError):
        fit_preprocess(x, y, names, dict(no_labels, interaction_pairs=(("A", "B"),)))
    # a config without the B11 keys (saved before they existed) still works
    old = {k: v for k, v in no_labels.items() if not k.startswith(("poly", "inter"))}
    assert fit_preprocess(x, None, names, old)["pairs"] == []


def test_each_step_can_be_switched_off():
    x, y, names = fake_brfss()
    for full in (DEFAULT_CONFIG, V1_CONFIG):
        base = dict(full, min_missing_rows=5)
        variants = [dict(base, **{key: False}) for key, v in base.items() if v is True]
        variants += [dict(base, impute="median"), dict(base, max_missing_frac=1.0)]
        for config in variants:
            state = fit_preprocess(x, y, names, config)
            tx = apply_preprocess(x[:50], names, state)
            assert tx.shape == (50, len(state["out_names"]))
            assert np.isfinite(tx).all()
