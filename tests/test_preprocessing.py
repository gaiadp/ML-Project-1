"""Tests for codes_to_nan (src/preprocessing.py): BRFSS special codes by column."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import codes_to_nan  # noqa: E402


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
