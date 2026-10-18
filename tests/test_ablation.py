"""Tests for experiments/ablation.py (fake rows and fake data, no real training)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.ablation import (  # noqa: E402
    FINAL,
    data_table,
    data_variants,
    model_table,
)
from src.pipeline import (  # noqa: E402
    DEFAULT_CONFIG,
    V1_CONFIG,
    apply_preprocess,
    fit_preprocess,
)
from test_pipeline import fake_brfss  # noqa: E402


def test_data_variants():
    variants = data_variants(V1_CONFIG)
    assert list(variants)[0] == FINAL and variants[FINAL] == V1_CONFIG
    # 4 rules, no selection, NaN rule instead, 8 steps, median imputation
    assert len(variants) == 1 + 4 + 1 + 1 + 8 + 1
    assert variants["selection without R3"]["select_features"] == ("R1", "R2", "R4")
    nan_rule = variants["NaN rule (B5) instead of the selection"]
    assert not nan_rule["select_features"] and nan_rule["max_missing_frac"] == 0.5
    # every variant switches off one thing only
    for label, config in variants.items():
        changed = [k for k in config if config[k] != V1_CONFIG[k]]
        assert len(changed) <= 2, label
    # no selection in the base: no selection rows; a step already off: no row
    variants = data_variants(dict(DEFAULT_CONFIG, binning=False))
    assert not any("selection" in label for label in variants)
    assert "no binning of PHYSHLTH (B10)" not in variants


def test_every_variant_runs_through_the_pipeline():
    x, y, names = fake_brfss()
    for label, config in data_variants(V1_CONFIG).items():
        state = fit_preprocess(x, y, names, dict(config, min_missing_rows=5))
        tx = apply_preprocess(x[:20], names, state)
        assert np.isfinite(tx).all(), label


def row(label, f1_folds, **fit):
    """A CV row as saved by experiments.common.run_grid."""
    return dict(
        label=label,
        fit=fit,
        f1_mean=float(np.mean(f1_folds)),
        f1_std=float(np.std(f1_folds)),
        f1_folds=f1_folds,
        f1_default_folds=[f - 0.2 for f in f1_folds],
        n_features=10,
        finite=True,
    )


def test_data_table():
    rows = [
        row(FINAL, [0.40, 0.40], lambda_=1e-4),
        row(FINAL, [0.42, 0.42], lambda_=1e-3),  # best lambda of the final pipeline
        row("no x", [0.43, 0.41], lambda_=1e-4),
        row("no x", [0.39, 0.39], lambda_=1e-3),
    ]
    table = data_table(rows)
    assert [e["label"] for e in table] == [FINAL, "no x"]
    final, no_x = table
    assert final["lambda_"] == 1e-3 and final["diff"] == 0.0
    # each variant keeps its own best lambda, paired with the best final row
    assert no_x["lambda_"] == 1e-4
    assert np.isclose(no_x["diff"], 0.0) and no_x["better"] == 1


def test_model_table():
    final_fit = dict(lambda_=1e-4, n_trees=200)
    final_rows = [
        row(FINAL, [0.44, 0.44], **final_fit),
        row(FINAL, [0.43, 0.42], lambda_=1e-4, n_trees=0),
    ]
    gd_row = row(FINAL, [0.42, 0.45], lambda_=1e-4, gamma=0.3)
    table = model_table(final_rows, gd_row, final_fit)
    labels = [e["label"] for e in table]
    assert labels[0] == "final model" and "threshold" in labels[-1]
    assert len(table) == 4
    assert np.isclose(table[1]["diff"], -0.015) and table[1]["better"] == 0
    assert np.isclose(table[2]["diff"], -0.005) and table[2]["better"] == 1
    # threshold 0.5: the F1 with the default threshold of the final row
    assert np.isclose(table[3]["f1"], 0.24)
    # saved results of another final setting
    with pytest.raises(ValueError):
        model_table(final_rows, gd_row, dict(lambda_=1e-4, n_trees=400))
