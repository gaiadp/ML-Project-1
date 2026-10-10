"""Consistency checks of the column lists in src/feature_types.py (no data needed)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_types import DOMAIN_DROP, DROP, FEATURE_TYPES, TYPES  # noqa: E402

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
