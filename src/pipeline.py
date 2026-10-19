"""Preprocessing pipeline: statistics fitted on the training set only, then applied
unchanged to training, validation and test.

Steps, named B1-B13 in the comments of src/ (B = data cleaning and features), each
one switched on or off by the config for the ablation:
    B1  codes_to_nan        BRFSS special codes -> NaN (don't know) or 0 (none)
    B3  convert_units       one numeric scale per column; raw mixed-units columns dropped
    B4  skip_logic_fill     NaN of the questions that were not asked -> "no" / 0
    B10 binning             count columns of BINNED -> bin index, one-hot by B9
    B2  domain_drop         drop ids, survey design (DOMAIN_DROP); the label leakage
                            columns (LABEL_LEAKAGE) are dropped also without it ...
        keep_leakage        ... unless this is True: only to measure the leakage
        drop_redundant      drop the duplicates of other columns (REDUNDANT)
        select_features     drop the columns not chosen from the EDA (NOT_SELECTED):
                            True = all its rules, or a tuple of rules ("R1", ...)
    B5  max_missing_frac    drop columns with too many NaN ...
        keep_informative    ... except those whose missingness is informative
    B6  missing_indicators  one isnan column per group of informative columns
    B7  impute              median, or the mode for binary / nominal columns
    B8  recode_binary       1/2 -> 1/0
    B9  one_hot             nominal (and binned) columns -> one column per frequent value
        one_hot_ordinal     ... also the ordinal columns of ONE_HOT_ORDINAL
    B12 standardization (mean / std of the training set)
    B11 poly_degree         powers z^2 .. z^degree of the standardized columns
        interaction_top     products z_i * z_j of the interaction_top columns most
                            correlated with the label (all pairs) ...
        interaction_pairs   ... and of the pairs of output names listed here
        the new columns are standardized with their own training mean / std
    B13 bias column

B5, B6 and interaction_top use the labels: in cross-validation, fit the pipeline on
the training part of each fold only.

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
    NOT_SELECTED,
    SELECTION_RULES,
    ONE_HOT_ORDINAL,
    REDUNDANT,
)
from src.features import (
    add_bias,
    fit_one_hot_categories,
    one_hot_encode,
    pairwise_products,
)
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
    # B2: drop the columns of NOT_SELECTED (V1_CONFIG): True = all the rules R1-R4,
    # or a tuple of the rules to apply, e.g. ("R1", "R2", "R4")
    select_features=False,
    # B2: keep the LABEL_LEAKAGE columns (HAREHAB1). Only to measure what the leakage
    # is worth (experiments/leakage.py): never in a model that is reported as such
    keep_leakage=False,
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
    # B11, off by default: the degree and the interactions are tuned per method
    poly_degree=1,  # powers 2..poly_degree of the columns with > p distinct values
    interaction_top=0,  # all pairs of the k columns most correlated with y
    interaction_pairs=(),  # extra pairs of output names, e.g. (("_AGE80", "_BMI5"),)
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

# pipeline v1: the full pipeline on the 124 columns chosen from the EDA (the others are
# in NOT_SELECTED, src/feature_types.py, with the reason). The selection replaces B5:
# every selected column is kept, also with many NaN (CVDASPRN, 96% NaN: B5 would drop
# it, MICHD is the same when missing, but |r| = 0.26 on its answers). B6 still adds
# the missing indicators.
V1_CONFIG = dict(DEFAULT_CONFIG, select_features=True, max_missing_frac=1.0)


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


def not_selected(rules=True):
    """Columns left out by the feature selection (NOT_SELECTED) under some rules.

    Args:
        rules: True for all the rules (SELECTION_RULES), or a collection of rule
            names, e.g. ("R1", "R2"); every reason in NOT_SELECTED starts with
            its rule, "R1:" .. "R4:"

    Returns:
        list of column names
    """
    rules = SELECTION_RULES if rules is True else tuple(rules)
    unknown = sorted(set(rules) - set(SELECTION_RULES))
    if unknown:
        raise ValueError(
            f"unknown selection rules {unknown}, expected {SELECTION_RULES}"
        )
    return [name for name, reason in NOT_SELECTED.items() if reason[:2] in rules]


def _candidate_columns(names, config):
    """B2, B3: bool mask of the columns that can be features (fixed lists)."""
    drop = set()
    if config["domain_drop"]:
        drop |= set(DOMAIN_DROP)
    if config["drop_redundant"]:
        drop |= set(REDUNDANT)
    if config["select_features"]:
        drop |= set(not_selected(config["select_features"]))
    if config["convert_units"]:
        drop |= set(CONVERTED)  # the converted version is kept instead
    # the label leakage columns are dropped with any config, also without domain_drop;
    # keep_leakage keeps them (also against DOMAIN_DROP), only to measure the leakage.
    # .get: the configs saved before keep_leakage existed do not have the key
    if config.get("keep_leakage", False):
        drop -= set(LABEL_LEAKAGE)
    else:
        drop |= set(LABEL_LEAKAGE)
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


def _fit_expansion(z, y, names, config):
    """B11: which powers and products to add, on the standardized training columns.

    The power p of a column with <= p distinct values is a linear combination of its
    lower powers (z^2 = 1 for a standardized binary column...): only the columns with
    more than p distinct values get z^p.

    Args:
        z: standardized training columns, shape=(N, K), without the bias
        y: training labels, shape=(N,); used only by interaction_top
        names: the K column names
        config: the pipeline config

    Returns:
        powers: list of (p, column indices), p = 2 .. poly_degree
        pairs: list of (i, j) column index pairs, i < j, without duplicates
    """
    # .get: configs saved before B11 existed have none of its keys
    degree = config.get("poly_degree", 1)
    top = config.get("interaction_top", 0)
    named_pairs = config.get("interaction_pairs", ())

    powers = []
    if degree >= 2:
        n_distinct = np.array([len(np.unique(z[:, j])) for j in range(z.shape[1])])
        for p in range(2, degree + 1):
            cols = np.flatnonzero(n_distinct > p)
            if len(cols):
                powers.append((p, cols))

    pairs = []
    if top > 0:
        if y is None:
            raise ValueError("interaction_top needs y_train")
        yc = np.asarray(y, dtype=float) - np.mean(y)
        # z is standardized (constant columns are 0): |z^T yc| ranks as |Pearson r|
        score = np.abs(z.T @ yc)
        best = np.sort(np.argsort(-score, kind="stable")[:top])
        pairs += [(a, b) for k, a in enumerate(best) for b in best[k + 1 :]]
    index = {name: k for k, name in enumerate(names)}
    for a, b in named_pairs:
        for name in (a, b):
            if name not in index:
                raise ValueError(f"interaction_pairs: no output column named {name!r}")
        pairs.append(tuple(sorted((index[a], index[b]))))
    pairs = list(dict.fromkeys((int(a), int(b)) for a, b in pairs if a != b))
    return powers, pairs


def _expand(z, powers, pairs):
    """B11: the power and product columns of z, shape=(N, P), P = 0 if none."""
    new = [z[:, cols] ** p for p, cols in powers]
    if pairs:
        new.append(pairwise_products(z, pairs))
    return np.hstack(new) if new else np.empty((z.shape[0], 0))


def fit_preprocess(x_train, y_train, names, config=None):
    """Compute the preprocessing statistics on the training set.

    Args:
        x_train: raw training data, shape=(N, D)
        y_train: training labels, shape=(N,), -1/1 or 0/1. Used only by
            keep_informative, missing_indicators and interaction_top: may be None
            if they are all off.
        names: the D column names
        config: dict with the keys of DEFAULT_CONFIG; None = DEFAULT_CONFIG

    Returns:
        state: dict for apply_preprocess with the config and
            cols: indices of the kept columns, shape=(K,)
            ind_cols: indices of the columns with a missing indicator, shape=(M,)
            fill: imputation values, recode: binary columns to recode (shape=(K,))
            one_hot: (k, categories) for each nominal column cols[k]; it is replaced
                by one column per category, named "NAME=value"
            mean, std: standardization statistics of the columns before B11
            powers, pairs: B11 columns, see _fit_expansion; named "NAME^p", "A*B"
            expand_mean, expand_std: standardization statistics of the B11 columns
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
    x = _transform(x, state)
    state["mean"], state["std"] = fit_standardize(x)
    encoded = {k for k, _ in one_hot}
    base_names = (
        [str(names[j]) for k, j in enumerate(cols) if k not in encoded]
        + [f"{names[cols[k]]}={c:g}" for k, cats in one_hot for c in cats]
        + [str(names[j]) + "_isnan" for j in ind_cols]
    )

    # B11: powers and products of the standardized columns, standardized again
    z = apply_standardize(x, state["mean"], state["std"])
    del x
    powers, pairs = _fit_expansion(z, y_train, base_names, config)
    state["powers"], state["pairs"] = powers, pairs
    state["expand_mean"], state["expand_std"] = fit_standardize(
        _expand(z, powers, pairs)
    )
    state["out_names"] = (
        ["bias"]
        + base_names
        + [f"{base_names[j]}^{p}" for p, c in powers for j in c]
        + [f"{base_names[a]}*{base_names[b]}" for a, b in pairs]
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
    if state["powers"] or state["pairs"]:
        new = _expand(x, state["powers"], state["pairs"])
        new = apply_standardize(new, state["expand_mean"], state["expand_std"])
        x = np.hstack([x, new])
    return add_bias(x)
