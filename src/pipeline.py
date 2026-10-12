"""Preprocessing pipeline: statistics fitted on the training set only, then applied
unchanged to training, validation and test.

Steps, named B1-B13 in the comments of src/ (B = data cleaning and features), each
one switched on or off by the config for the ablation:
    B1  codes_to_nan        BRFSS special codes -> NaN (don't know) or 0 (none)
    B3  convert_units       one numeric scale per column; raw mixed units dropped
    B4  skip_logic_fill     NaN of the questions that were not asked -> "no" / 0
    B2  domain_drop         drop ids, survey design, label leakage (DOMAIN_DROP)
        drop_redundant      drop the duplicates of other columns (REDUNDANT)
    B5  max_missing_frac    drop columns with too many NaN ...
        keep_informative    ... except those whose missingness is informative
    B6  missing_indicators  one isnan column per group of informative columns
    B7  impute              median, or the mode for binary / nominal columns
    B8  recode_binary       1/2 -> 1/0
    B9  one_hot             nominal and NOT_MONOTONE ordinal columns -> one-hot
        one_hot_large       ... also LARGE_NOMINAL (else they are dropped)
    B10 clip, log_counts    clip numeric columns at train percentiles; log1p of LOG1P
    B11 poly_degree         powers 2..poly_degree of the SQUARED columns
    B12 standardization (mean / std of the training set), B13 bias column

The column lists (src/feature_types.py) come from the EDA. B5 and B6 use the labels:
in cross-validation, fit the pipeline on the training part of each fold only.

Usage:
    state = fit_preprocess(x_train, y_train, names, config)
    tx_train = apply_preprocess(x_train, names, state)
    tx_test = apply_preprocess(x_test, names, state)
    state["out_names"]  # names of the columns of tx
"""

import numpy as np

from src.feature_types import (
    CONTINUOUS,
    CONVERTED,
    COUNT,
    DOMAIN_DROP,
    FEATURE_TYPES,
    LARGE_NOMINAL,
    LOG1P,
    MIXED_UNITS,
    NOMINAL,
    NOT_MONOTONE,
    REDUNDANT,
    SQUARED,
)
from src.features import add_bias, centered_powers, fit_categories, one_hot_encode
from src.preprocessing import (
    apply_binary_recode,
    apply_clip_log,
    apply_impute,
    codes_to_nan,
    convert_units,
    fill_skip_logic,
    fit_binary_recode,
    fit_clip_bounds,
    fit_impute_by_type,
    fit_impute_median,
    fit_standardize,
    missing_rate_difference,
    one_per_group,
)

# the full pipeline; the ablation variants switch off one step at a time
DEFAULT_CONFIG = dict(
    domain_drop=True,  # B2: drop the columns of DOMAIN_DROP
    drop_redundant=True,  # B2: drop the duplicates of REDUNDANT
    convert_units=True,  # B3
    skip_logic_fill=True,  # B4
    max_missing_frac=0.5,  # B5: drop the columns with a larger fraction of NaN ...
    keep_informative=True,  # B5: ... unless their missingness is informative
    missing_indicators=True,  # B6
    min_missing_diff=0.03,  # B5/B6: informative = MICHD differs by >= 3 points ...
    min_missing_rows=500,  # B5/B6: ... between >= 500 NaN and >= 500 answered rows
    impute="by_type",  # B7: "median" or "by_type"
    recode_binary=True,  # B8
    one_hot=True,  # B9
    one_hot_large=True,  # B9: one-hot LARGE_NOMINAL too (False: drop them)
    min_category_rows=500,  # B9: rarer values join the reference value
    clip=True,  # B10: clip count / continuous columns ...
    clip_percentiles=(0.5, 99.5),  # B10: ... at these train percentiles
    log_counts=True,  # B10: log1p of the LOG1P columns
    poly_degree=2,  # B11: powers 2..poly_degree of the SQUARED columns (1 = none)
)

# pipeline v0 (first submission): codes_to_nan -> drop columns with > 50% NaN ->
# median imputation -> standardization -> bias
V0_CONFIG = dict(
    DEFAULT_CONFIG,
    domain_drop=False,
    drop_redundant=False,
    convert_units=False,
    skip_logic_fill=False,
    keep_informative=False,
    missing_indicators=False,
    impute="median",
    recode_binary=False,
    one_hot=False,
    clip=False,
    log_counts=False,
    poly_degree=1,
)

# types clipped by B10
_NUMERIC_TYPES = (COUNT, CONTINUOUS, MIXED_UNITS)


def _clean(x, names, config):
    """B1, B3, B4: the steps without fitted statistics. x is not modified."""
    x = codes_to_nan(x, names)  # a new array: the next steps work in place on it
    if config["convert_units"]:
        convert_units(x, names)
    if config["skip_logic_fill"]:
        fill_skip_logic(x, names)
    return x


def _candidate_columns(names, config):
    """B2, B3, B9: bool mask of the columns that can be features (fixed lists)."""
    drop = set()
    if config["domain_drop"]:
        drop |= set(DOMAIN_DROP)
    if config["drop_redundant"]:
        drop |= set(REDUNDANT)
    if config["convert_units"]:
        drop |= set(CONVERTED)  # the converted version is kept instead
    if config["one_hot"] and not config["one_hot_large"]:
        drop |= set(LARGE_NOMINAL)
    return np.array([name not in drop for name in names])


def _numeric_steps(x, state):
    """B7, B8, B10 on the kept columns. Returns a new array."""
    x = apply_impute(x, state["fill"])
    apply_binary_recode(x, state["recode"])
    apply_clip_log(x, state["low"], state["high"], state["log"])
    return x


def _expand(x, indicators, state):
    """B9, B11: the final columns from the output of _numeric_steps: the columns kept
    as numbers, the one-hot blocks, the powers, the missing indicators."""
    blocks = [x[:, state["plain"]]]
    blocks += [one_hot_encode(x[:, k], cats)[0] for k, cats in state["one_hot"]]
    degree = state["config"]["poly_degree"]
    blocks.append(centered_powers(x[:, state["squared"]], state["center"], degree))
    blocks.append(indicators)
    return np.hstack(blocks)


def _transform(x, state):
    """B5-B11 on cleaned data (output of _clean)."""
    indicators = np.isnan(x[:, state["ind_cols"]]).astype(float)
    return _expand(_numeric_steps(x[:, state["cols"]], state), indicators, state)


def _output_names(names, state):
    """Names of the columns of tx, in the order of _expand, "bias" first."""
    kept = [str(names[j]) for j in state["cols"]]
    out = ["bias"] + [kept[k] for k in state["plain"]]
    for k, cats in state["one_hot"]:
        out += [f"{kept[k]}={c:g}" for c in cats]
    for d in range(2, state["config"]["poly_degree"] + 1):
        out += [f"{kept[k]}^{d}" for k in state["squared"]]
    return out + [str(names[j]) + "_isnan" for j in state["ind_cols"]]


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
            fill: imputation values, recode: binary columns to recode,
            low, high, log: clipping bounds and log1p columns (all shape=(K,))
            plain: kept columns that stay numbers, one_hot: (kept column,
            categories) pairs, squared: kept columns with powers, center: their
            train means after B10
            mean, std: standardization statistics of the columns of tx but bias
            out_names: names of the columns of tx, "bias" first
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

    # B9 decides the types used by B7, B8, B10: one-hot columns are categories
    kept_names = [str(names[j]) for j in cols]
    types = [FEATURE_TYPES.get(name) for name in kept_names]
    one_hot = np.zeros(len(cols), dtype=bool)
    if config["one_hot"]:
        for k, name in enumerate(kept_names):
            one_hot[k] = types[k] == NOMINAL or name in NOT_MONOTONE
        types = [NOMINAL if oh else t for oh, t in zip(one_hot, types)]

    # B7, B8, B10 statistics on the kept columns, before imputation
    x_kept = x[:, cols]
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
    numeric = np.array([t in _NUMERIC_TYPES for t in types], dtype=bool)
    low, high = np.full(len(cols), -np.inf), np.full(len(cols), np.inf)
    if config["clip"]:
        low, high = fit_clip_bounds(x_kept, numeric, config["clip_percentiles"])
    # log1p needs values >= 0: the LOG1P columns are counts and durations
    log = np.zeros(len(cols), dtype=bool)
    if config["log_counts"]:
        for k, name in enumerate(kept_names):
            log[k] = name in LOG1P and not np.any(x_kept[:, k] < 0)

    state = dict(
        config=config,
        cols=cols,
        ind_cols=ind_cols,
        fill=fill,
        recode=recode,
        low=low,
        high=high,
        log=log,
    )
    x_kept = _numeric_steps(x_kept, state)

    # B9 categories and B11 centers, on the imputed, recoded, clipped training data
    state["one_hot"] = [
        (k, fit_categories(x_kept[:, k], config["min_category_rows"]))
        for k in np.flatnonzero(one_hot)
    ]
    state["plain"] = np.flatnonzero(~one_hot)
    squared = [k for k in state["plain"] if kept_names[k] in SQUARED]
    if config["poly_degree"] < 2:
        squared = []
    state["squared"] = np.array(squared, dtype=int)
    state["center"] = x_kept[:, state["squared"]].mean(axis=0)

    # B12: statistics of the transformed training set
    indicators = np.isnan(x[:, ind_cols]).astype(float)
    del x
    state["mean"], state["std"] = fit_standardize(_expand(x_kept, indicators, state))
    state["out_names"] = _output_names(names, state)
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
    # B12 in place, to avoid one more copy of the largest array
    x -= state["mean"]
    x /= state["std"]
    return add_bias(x)
