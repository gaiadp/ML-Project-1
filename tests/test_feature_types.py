"""Consistency checks of the column lists in src/feature_types.py (no data needed)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_types import (  # noqa: E402
    BINNED,
    CONTINUOUS,
    CONVERTED,
    DOMAIN_DROP,
    DROP,
    FEATURE_TYPES,
    LABEL_LEAKAGE,
    MIXED_UNITS,
    ONE_HOT_ORDINAL,
    ORDINAL,
    REDUNDANT,
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


def test_label_leakage_columns_are_dropped():
    assert set(LABEL_LEAKAGE) <= set(DOMAIN_DROP)
    assert all(DOMAIN_DROP[n].startswith("label leakage") for n in LABEL_LEAKAGE)


def test_every_dropped_column_has_a_reason():
    assert all(isinstance(r, str) and r for r in DOMAIN_DROP.values())


def test_converted_columns():
    for raw, converted in CONVERTED.items():
        assert FEATURE_TYPES[raw] == MIXED_UNITS
        assert FEATURE_TYPES[converted] == CONTINUOUS


def test_redundant_columns():
    """Each dropped duplicate names a kept column that the pipeline uses."""
    for name, (kept, reason) in REDUNDANT.items():
        assert FEATURE_TYPES[name] != DROP and name not in CONVERTED, name
        assert isinstance(reason, str) and reason, name
        assert FEATURE_TYPES[kept] not in (DROP, MIXED_UNITS), kept
        assert kept not in REDUNDANT and kept not in CONVERTED, kept


def test_one_hot_and_binned_columns():
    for name in ONE_HOT_ORDINAL:
        assert FEATURE_TYPES[name] == ORDINAL, name
        assert name not in REDUNDANT, name
    for name, edges in BINNED.items():
        assert FEATURE_TYPES[name] not in (DROP, MIXED_UNITS), name
        assert list(edges) == sorted(set(edges)), name
