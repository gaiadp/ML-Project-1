"""Consistency checks of the column lists in src/feature_types.py (no data needed)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_types import (  # noqa: E402
    CONTINUOUS,
    CONVERTED,
    COUNT,
    DOMAIN_DROP,
    DROP,
    FEATURE_TYPES,
    LARGE_NOMINAL,
    LOG1P,
    MIXED_UNITS,
    NOMINAL,
    NOT_MONOTONE,
    ORDINAL,
    REDUNDANT,
    SQUARED,
    TYPES,
)

# columns of x_train.csv without Id; a key repeated by mistake in the dict literal
# silently replaces the first one, so the dict would be shorter
N_COLUMNS = 321


def test_every_column_has_a_valid_type():
    assert len(FEATURE_TYPES) == N_COLUMNS
    assert set(FEATURE_TYPES.values()) <= set(TYPES)


def test_drop_type_matches_domain_drop():
    dropped = {name for name, t in FEATURE_TYPES.items() if t == DROP}
    assert dropped == set(DOMAIN_DROP)


def test_every_dropped_column_has_a_reason():
    assert all(isinstance(r, str) and r for r in DOMAIN_DROP.values())


def test_eda_lists():
    # names from the EDA: a typo would silently skip a column in the pipeline
    assert all(FEATURE_TYPES[name] != DROP for name in REDUNDANT)
    for kept_name in REDUNDANT.values():
        assert kept_name not in REDUNDANT and FEATURE_TYPES[kept_name] != DROP
    assert all(FEATURE_TYPES[name] == ORDINAL for name in NOT_MONOTONE)
    assert all(FEATURE_TYPES[name] == NOMINAL for name in LARGE_NOMINAL)
    numeric = (COUNT, CONTINUOUS, MIXED_UNITS)
    assert all(FEATURE_TYPES[name] in numeric for name in LOG1P + SQUARED)
    lists = [list(REDUNDANT), NOT_MONOTONE, LARGE_NOMINAL, LOG1P, SQUARED]
    assert all(len(set(names)) == len(names) for names in lists)


def test_converted_columns():
    for raw, converted in CONVERTED.items():
        assert FEATURE_TYPES[raw] == MIXED_UNITS
        assert FEATURE_TYPES[converted] == CONTINUOUS
