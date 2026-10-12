"""Tests for the cleaning steps of src/preprocessing.py: BRFSS special codes by
column (codes_to_nan), unit conversion, skip logic, binning, missing indicators, imputation
and binary recoding."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_types import FEATURE_TYPES  # noqa: E402
from src.preprocessing import (  # noqa: E402
    _SKIP_LOGIC,
    apply_binary_recode,
    bin_columns,
    codes_to_nan,
    convert_units,
    fill_skip_logic,
    fit_binary_recode,
    fit_impute_by_type,
    missing_rate_difference,
    one_per_group,
)


def clean_column(name, values):
    """Run codes_to_nan on a single column and return it as a 1D array."""
    x = np.array(values, dtype=float)[:, None]
    return codes_to_nan(x, [name])[:, 0]


def test_same_code_depends_on_the_column():
    # 7 = retired, 9 = refused
    np.testing.assert_array_equal(clean_column("EMPLOY1", [7, 9]), [7, np.nan])
    # 7 = don't know, 9 = refused
    np.testing.assert_array_equal(
        clean_column("GENHLTH", [1, 7, 9]), [1, np.nan, np.nan]
    )
    # 9 = age group 60-64, 14 = don't know
    np.testing.assert_array_equal(clean_column("_AGEG5YR", [9, 14]), [9, np.nan])


def test_none_codes_become_zero():
    # 88 = no days; 7 days is a real answer
    np.testing.assert_array_equal(clean_column("PHYSHLTH", [88, 7, 77]), [0, 7, np.nan])
    # 8 = never has symptoms, below 1 = less than once a week
    np.testing.assert_array_equal(clean_column("ASYMPTOM", [8, 3, 9]), [0, 3, np.nan])
    # 555 = no feet, 888 = never: no checks
    np.testing.assert_array_equal(clean_column("FEETCHK2", [555, 888]), [0, 0])


def test_codes_fixed_with_the_codebook():
    # visits 1-87: 77 is a count, 98 = don't know
    np.testing.assert_array_equal(
        clean_column("ASERVIST", [77, 88, 98]), [77, 0, np.nan]
    )
    # 98 = never heard of the test
    np.testing.assert_array_equal(clean_column("CHKHEMO3", [98, 88]), [np.nan, 0])
    # 97 = "10 or younger", not 97 years
    np.testing.assert_array_equal(clean_column("ASTHMAGE", [97, 45]), [np.nan, 45])
    # 8 = never drives or rides in a car; 6 = unable for other reasons
    np.testing.assert_array_equal(clean_column("SEATBELT", [8, 5]), [np.nan, 5])
    np.testing.assert_array_equal(
        clean_column("VIDFCLT2", [6, 8, 5]), [np.nan, np.nan, 5]
    )
    # 555 = all my life: kept for the unit conversion, not 0
    np.testing.assert_array_equal(clean_column("LONGWTCH", [555, 777]), [555, np.nan])
    # 99 = don't know code 99000 / 1000; real values stop at 98.999
    np.testing.assert_array_equal(
        clean_column("PAFREQ1_", [99, 98.999]), [np.nan, 98.999]
    )


def test_input_is_not_modified():
    x = np.array([[9.0, 88.0]])
    codes_to_nan(x, ["GENHLTH", "PHYSHLTH"])
    np.testing.assert_array_equal(x, [[9.0, 88.0]])


def test_convert_units():
    names = ["IYEAR", "IMONTH", "_AGE80"]
    names += ["BLDSUGAR", "LONGWTCH", "HIVTSTD3", "CHECKUP1"]
    x = np.array(
        [
            # interview Oct 2015; 2 per day; 3 weeks; test Jun 2015; never checked
            [2015, 10, 60, 102, 203, 62015, 8],
            # interview Jan 2016; never; all my life; month unknown; 2 = 1-2 years ago
            [2016, 1, 45, 0, 555, 772015, 2],
            # 1 per year; 2 years; typing error in the year; don't know
            [2015, 3, 30, 401, 402, 80201, np.nan],
        ]
    )
    convert_units(x, names)
    np.testing.assert_allclose(x[:, 3], [730, 0, 1])  # times per year
    np.testing.assert_allclose(x[:, 4], [3 / 52, 45, 2])  # years
    np.testing.assert_allclose(x[:, 5], [4, 6.5, np.nan])  # months before
    np.testing.assert_array_equal(x[:, 6], [5, 2, np.nan])  # never after 4


def test_skip_logic_names_exist():
    # fill_skip_logic skips unknown names silently: a typo would disable a rule
    for gate, fills in _SKIP_LOGIC:
        for name in list(gate) + list(fills):
            assert name in FEATURE_TYPES, name


def test_fill_skip_logic():
    names = ["SMOKE100", "SMOKDAY2", "LASTSMK2", "PHYSHLTH", "MENTHLTH", "POORHLTH"]
    x = np.array(
        [
            # never smoked; no bad day
            [2, np.nan, np.nan, 0, 0, np.nan],
            # smokes every day; 5 bad mental health days
            [1, 1, np.nan, 0, 5, np.nan],
            # gate unknown: nothing is filled; an answer is never replaced
            [np.nan, np.nan, np.nan, 0, 0, 3],
        ]
    )
    fill_skip_logic(x, names)
    np.testing.assert_array_equal(x[:, 1], [3, 1, np.nan])
    np.testing.assert_array_equal(x[:, 2], [8, 0, np.nan])
    np.testing.assert_array_equal(x[:, 5], [0, np.nan, 3])


def test_missing_rate_difference():
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1, -1, -1])
    nan = np.zeros((10, 3), dtype=bool)
    nan[:5, 0] = True  # 4 of 5 NaN rows are positive, 0 of 5 answered rows
    nan[0, 2] = True  # a single NaN row
    diff = missing_rate_difference(nan, y, min_rows=2)
    np.testing.assert_allclose(diff, [0.8, 0.0, 0.0])


def test_one_per_group():
    nan = np.zeros((10, 4), dtype=bool)
    nan[:5, 0] = True
    nan[:5, 1] = True  # same mask as column 0
    nan[5:, 2] = True  # opposite mask: same group
    nan[::2, 3] = True  # another group
    np.testing.assert_array_equal(one_per_group(nan, [1, 0, 2, 3]), [1, 3])


def test_impute_and_recode_by_type():
    # nominal column: the median (3) is a category that nobody chose, the mode is 1
    x = np.array(
        [
            [1, 20.0, 1, 0],
            [1, 25.0, 2, 1],
            [5, 30.0, 2, 1],
            [6, np.nan, np.nan, 1],
            [np.nan, 40.0, 2, np.nan],
        ]
    )
    types = ["nominal", "continuous", "binary", "binary"]
    np.testing.assert_allclose(fit_impute_by_type(x, types), [1, 27.5, 2, 1])
    # SEX-like column 1/2 is recoded, a 0/1 column (e.g. _FRTRESP) is not
    recode = fit_binary_recode(x, types)
    np.testing.assert_array_equal(recode, [False, False, True, False])
    apply_binary_recode(x, recode)
    np.testing.assert_array_equal(x[:, 2], [1, 0, 0, np.nan, 0])
    np.testing.assert_array_equal(x[:, 3], [0, 1, 1, 1, np.nan])


def test_bin_columns_physhlth():
    # raw codes: 88 = none (0 days), 77 = don't know (NaN)
    days = clean_column("PHYSHLTH", [88, 1, 2, 3, 7, 8, 14, 15, 29, 30, 77])
    x = np.column_stack([days, days])
    bin_columns(x, ["PHYSHLTH", "OTHER"], {"PHYSHLTH": (1, 3, 8, 15, 30)})
    np.testing.assert_array_equal(x[:-1, 0], [0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
    assert np.isnan(x[-1, 0])
    np.testing.assert_array_equal(x[:, 1], days)  # not in bins: unchanged
