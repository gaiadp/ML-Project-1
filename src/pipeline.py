"""Preprocessing pipeline: statistics fitted on the training set only, then applied
unchanged to training, validation and test.

Steps, named B1-B13 in the comments of src/ (B = data cleaning and features), each
one switched on or off by the config for the ablation:
    B1  codes_to_nan        BRFSS special codes -> NaN (don't know) or 0 (none)
    B3  convert_units       one numeric scale per column; raw mixed-units columns dropped
    B4  skip_logic_fill     NaN of the questions that were not asked -> "no" / 0
    B10 binning             count columns of BINNED -> bin index, one-hot by B9
    B2  domain_drop         drop ids, survey design (DOMAIN_DROP); the label leakage
                            columns (LABEL_LEAKAGE) are dropped also without it
        drop_redundant      drop the duplicates of other columns (REDUNDANT)
    B5  max_missing_frac    drop columns with too many NaN ...
        keep_informative    ... except those whose missingness is informative
    B6  missing_indicators  one isnan column per group of informative columns
    B7  impute              median, or the mode for binary / nominal columns
    B8  recode_binary       1/2 -> 1/0
    B9  one_hot             nominal (and binned) columns -> one column per frequent value
        one_hot_ordinal     ... also the ordinal columns of ONE_HOT_ORDINAL
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

from src.feature_types import (
    BINNED,
    CONVERTED,
    DOMAIN_DROP,
    FEATURE_TYPES,
    LABEL_LEAKAGE,
    NOMINAL,
    ONE_HOT_ORDINAL,
    REDUNDANT,
)
from src.features import add_bias, fit_one_hot_categories, one_hot_encode
from src.preprocessing import (
    apply_binary_recode,
    apply_impute,
    apply_standardize,
    bin_columns,
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
    drop_redundant=True,  # B2: drop the columns of REDUNDANT
    convert_units=True,  # B3
    skip_logic_fill=True,  # B4
    max_missing_frac=0.5,  # B5: drop the columns with a larger fraction of NaN ...
    keep_informative=True,  # B5: ... unless their missingness is informative
    missing_indicators=True,  # B6
    min_missing_diff=0.03,  # B5/B6: informative = MICHD differs by >= 3 points ...
    min_missing_rows=500,  # B5/B6: ... between >= 500 NaN and >= 500 answered rows
    impute="by_type",  # B7: "median" or "by_type"
    recode_binary=True,  # B8
    one_hot=True,  # B9: one-hot encode the nominal columns ...
    one_hot_min_rows=500,  # ... one column per value with >= 500 training rows
    one_hot_ordinal=True,  # B9: ... and the ordinal columns of ONE_HOT_ORDINAL
    binning=True,  # B10: bins of the columns in BINNED (one-hot if one_hot)
)

# pipeline v0, the baseline: codes_to_nan -> drop columns with > 50% NaN -> median
# imputation -> one-hot of the nominal columns -> standardization -> bias.
# The first submission is V0_CONFIG without one-hot: dict(V0_CONFIG, one_hot=False).
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
    one_hot_ordinal=False,
    binning=False,
)


def _clean(x, names, config):
    """B1, B3, B4, B10: the steps without fitted statistics. x is not modified."""
    x = codes_to_nan(x, names)  # a new array: the next steps work in place on it
    if config["convert_units"]:
        convert_units(x, names)
    if config["skip_logic_fill"]:
        fill_skip_logic(x, names)
    if config["binning"]:
        bin_columns(x, names, BINNED)
    return x


def _candidate_columns(names, config):
    """B2, B3: bool mask of the columns that can be features (fixed lists)."""
    drop = set(LABEL_LEAKAGE)  # with any config
    if config["domain_drop"]:
        drop |= set(DOMAIN_DROP)
    if config["drop_redundant"]:
        drop |= set(REDUNDANT)
    if config["convert_units"]:
        drop |= set(CONVERTED)  # the converted version is kept instead
    return np.array([name not in drop for name in names])


def _transform(x, state):
    """B5-B9 on cleaned data: kept columns (imputed, recoded), then the one-hot
    columns of the nominal ones, then the missing indicators."""
    indicators = np.isnan(x[:, state["ind_cols"]]).astype(float)
    x = apply_impute(x[:, state["cols"]], state["fill"])
    apply_binary_recode(x, state["recode"])
    one_hot = [one_hot_encode(x[:, k], cats)[0] for k, cats in state["one_hot"]]
    encoded = np.array([k for k, _ in state["one_hot"]], dtype=int)
    return np.hstack([np.delete(x, encoded, axis=1), *one_hot, indicators])


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
            one_hot: (k, categories) for each nominal column cols[k]; it is replaced
                by one column per category, named "NAME=value"
            mean, std: standardization statistics, one per column of tx but the bias
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

    # B7, B8 on the kept columns
    x_kept = x[:, cols]
    # binned and ONE_HOT_ORDINAL columns are sets of classes: imputed with their mode
    # and one-hot encoded, as the nominal ones
    as_nominal = set(BINNED) if config["binning"] else set()
    if config["one_hot_ordinal"]:
        as_nominal |= set(ONE_HOT_ORDINAL)
    types = [
        NOMINAL if names[j] in as_nominal else FEATURE_TYPES.get(names[j]) for j in cols
    ]
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

    # B9: categories of each nominal column, on its imputed values
    one_hot = []
    if config["one_hot"]:
        for k, t in enumerate(types):
            if t == NOMINAL:
                v = np.where(np.isnan(x_kept[:, k]), fill[k], x_kept[:, k])
                one_hot.append(
                    (k, fit_one_hot_categories(v, config["one_hot_min_rows"]))
                )
    del x_kept

    state = dict(
        config=config,
        cols=cols,
        ind_cols=ind_cols,
        fill=fill,
        recode=recode,
        one_hot=one_hot,
    )
    # B12: statistics of the transformed training set
    state["mean"], state["std"] = fit_standardize(_transform(x, state))
    encoded = {k for k, _ in one_hot}
    state["out_names"] = (
        ["bias"]
        + [str(names[j]) for k, j in enumerate(cols) if k not in encoded]
        + [f"{names[cols[k]]}={c:g}" for k, cats in one_hot for c in cats]
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
