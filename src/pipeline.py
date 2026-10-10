"""Preprocessing pipeline: statistics fitted on the training set only, then applied
unchanged to training, validation and test.

Steps (block B of PLAN.md), each one switched on or off by the config for the ablation:
    B1  codes_to_nan        BRFSS special codes -> NaN (don't know) or 0 (none)
    B3  convert_units       one numeric scale per column; raw mixed-units columns dropped
    B4  skip_logic_fill     NaN of the questions that were not asked -> "no" / 0
    B2  domain_drop         drop ids, survey design, label leakage (DOMAIN_DROP)
    B5  max_missing_frac    drop columns with too many NaN ...
        keep_informative    ... except those whose missingness is informative
    B6  missing_indicators  one isnan column per group of informative columns
    B7  impute              median, or the mode for binary / nominal columns
    B8  recode_binary       1/2 -> 1/0
    B12 standardization (mean / std of the training set), B13 bias column

B5 and B6 use the labels: in cross-validation, fit the pipeline on the training part
of each fold only.

Usage:
    state = fit_preprocess(x_train, y_train, names, config)
    tx_train = apply_preprocess(x_train, names, state)
    tx_test = apply_preprocess(x_test, names, state)
    state["out_names"]  # names of the columns of tx
"""

import numpy as np

from src.feature_types import CONVERTED, DOMAIN_DROP, FEATURE_TYPES
from src.features import add_bias
from src.preprocessing import (
    apply_binary_recode,
    apply_impute,
    apply_standardize,
    codes_to_nan,
    convert_units,
    fill_skip_logic,
    fit_binary_recode,
    fit_impute_by_type,
    fit_impute_median,
    fit_standardize,
    missing_rate_difference,
    one_per_group,
)

# the full pipeline; the ablation variants switch off one step at a time
DEFAULT_CONFIG = dict(
    domain_drop=True,  # B2: drop the columns of DOMAIN_DROP
    convert_units=True,  # B3
    skip_logic_fill=True,  # B4
    max_missing_frac=0.5,  # B5: drop the columns with a larger fraction of NaN ...
    keep_informative=True,  # B5: ... unless their missingness is informative
    missing_indicators=True,  # B6
    min_missing_diff=0.03,  # B5/B6: informative = MICHD differs by >= 3 points ...
    min_missing_rows=500,  # B5/B6: ... between >= 500 NaN and >= 500 answered rows
    impute="by_type",  # B7: "median" or "by_type"
    recode_binary=True,  # B8
)

# pipeline v0 (first submission): codes_to_nan -> drop columns with > 50% NaN ->
# median imputation -> standardization -> bias
V0_CONFIG = dict(
    DEFAULT_CONFIG,
    domain_drop=False,
    convert_units=False,
    skip_logic_fill=False,
    keep_informative=False,
    missing_indicators=False,
    impute="median",
    recode_binary=False,
)


def _clean(x, names, config):
    """B1, B3, B4: the steps without fitted statistics. x is not modified."""
    x = codes_to_nan(x, names)  # a new array: the next steps work in place on it
    if config["convert_units"]:
        convert_units(x, names)
    if config["skip_logic_fill"]:
        fill_skip_logic(x, names)
    return x


def _candidate_columns(names, config):
    """B2, B3: bool mask of the columns that can be features (fixed lists)."""
    drop = set()
    if config["domain_drop"]:
        drop |= set(DOMAIN_DROP)
    if config["convert_units"]:
        drop |= set(CONVERTED)  # the converted version is kept instead
    return np.array([name not in drop for name in names])


def _transform(x, state):
    """B5-B8 on cleaned data: kept columns (imputed, recoded) and missing indicators."""
    indicators = np.isnan(x[:, state["ind_cols"]]).astype(float)
    x = apply_impute(x[:, state["cols"]], state["fill"])
    apply_binary_recode(x, state["recode"])
    return np.hstack([x, indicators])


def fit_preprocess(x_train, y_train, names, config=None):
    """Compute the preprocessing statistics on the training set.

    Args:
        x_train: raw training data, shape=(N, D)
        y_train: training labels, shape=(N,), -1/1 or 0/1. Used only by
            keep_informative and missing_indicators: may be None if both are off.
        names: the D column names
        config: dict with the keys of DEFAULT_CONFIG; None = DEFAULT_CONFIG

    Returns:
        state: dict for apply_preprocess with the config and
            cols: indices of the kept columns, shape=(K,)
            ind_cols: indices of the columns with a missing indicator, shape=(M,)
            fill: imputation values, recode: binary columns to recode (shape=(K,))
            mean, std: standardization statistics, shape=(K + M,)
            out_names: names of the 1 + K + M columns of tx, "bias" first
    """
    config = DEFAULT_CONFIG if config is None else config
    x = _clean(x_train, names, config)
    nan = np.isnan(x)
    candidate = _candidate_columns(names, config)

    # B5 / B6: columns whose missingness changes the MICHD prevalence
    use_labels = config["keep_informative"] or config["missing_indicators"]
    informative = np.zeros(len(names), dtype=bool)
    if use_labels:
        if y_train is None:
            raise ValueError("keep_informative / missing_indicators need y_train")
        diff = missing_rate_difference(nan, y_train, config["min_missing_rows"])
        informative = candidate & (np.abs(diff) >= config["min_missing_diff"])

    # B5: same rule as drop_high_missing_columns, plus the informative exception
    keep = candidate & (nan.mean(axis=0) <= config["max_missing_frac"])
    if config["keep_informative"]:
        keep |= informative
    cols = np.where(keep)[0]

    # B6: largest difference first, so it represents its group
    ind_cols = np.array([], dtype=int)
    if config["missing_indicators"]:
        order = np.argsort(-np.abs(diff), kind="stable")
        ind_cols = one_per_group(nan, order[informative[order]])
    del nan

    # B7, B8 on the kept columns
    x_kept = x[:, cols]
    types = [FEATURE_TYPES.get(names[j]) for j in cols]
    if config["impute"] == "median":
        fill = fit_impute_median(x_kept)
    elif config["impute"] == "by_type":
        fill = fit_impute_by_type(x_kept, types)
    else:
        raise ValueError(f"unknown impute: {config['impute']}")
    # a column without values in the training set (only if max_missing_frac = 1)
    fill[np.isnan(fill)] = 0.0
    recode = np.zeros(len(cols), dtype=bool)
    if config["recode_binary"]:
        recode = fit_binary_recode(x_kept, types)
    del x_kept

    state = dict(config=config, cols=cols, ind_cols=ind_cols, fill=fill, recode=recode)
    # B12: statistics of the transformed training set
    state["mean"], state["std"] = fit_standardize(_transform(x, state))
    state["out_names"] = (
        ["bias"]
        + [str(names[j]) for j in cols]
        + [str(names[j]) + "_isnan" for j in ind_cols]
    )
    return state


def apply_preprocess(x, names, state):
    """Apply the statistics of fit_preprocess to raw data x.

    Args:
        x: raw data, shape=(N, D), same columns as the training set
        names: the D column names
        state: dict returned by fit_preprocess

    Returns:
        tx: shape=(N, len(state["out_names"])), bias column first, no NaN
    """
    x = _transform(_clean(x, names, state["config"]), state)
    x = apply_standardize(x, state["mean"], state["std"])
    return add_bias(x)
