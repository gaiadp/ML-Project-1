"""Preprocessing pipeline: statistics fitted on the training set only, then applied
unchanged to training, validation and test.

Steps: codes_to_nan -> drop columns with too many NaN -> median imputation ->
standardization -> bias column.
"""

from src.features import add_bias
from src.preprocessing import (
    apply_impute,
    apply_standardize,
    codes_to_nan,
    drop_columns,
    drop_high_missing_columns,
    fit_impute_median,
    fit_standardize,
)


def fit_preprocess(x_train, names, max_missing_frac=0.5):
    """Compute the preprocessing statistics on the training set.

    Args:
        x_train: raw training data, shape=(N, D)
        names: the D column names
        max_missing_frac: columns with a larger fraction of NaN are dropped

    Returns:
        dict with keep (bool mask, shape=(D,)), medians, mean, std (shape=(D_kept,))
    """
    x = codes_to_nan(x_train, names)
    keep = drop_high_missing_columns(x, max_missing_frac)
    x = drop_columns(x, keep)
    medians = fit_impute_median(x)
    x = apply_impute(x, medians)
    mean, std = fit_standardize(x)
    return dict(keep=keep, medians=medians, mean=mean, std=std)


def apply_preprocess(x, names, params):
    """Apply the statistics of fit_preprocess to raw data x.

    Args:
        x: raw data, shape=(N, D), same columns as the training set
        names: the D column names
        params: dict returned by fit_preprocess

    Returns:
        tx: shape=(N, 1 + D_kept), bias column first
    """
    x = codes_to_nan(x, names)
    x = drop_columns(x, params["keep"])
    x = apply_impute(x, params["medians"])
    x = apply_standardize(x, params["mean"], params["std"])
    return add_bias(x)
