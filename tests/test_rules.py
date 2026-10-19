"""Tests for the known-positive rule (src/rules.py) and experiments/leakage.py."""

import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import run_final  # noqa: E402
from experiments.leakage import VARIANTS  # noqa: E402
from src.rules import apply_known_positives, known_positives  # noqa: E402


def test_known_positives_uses_the_raw_answers():
    names = ["GENHLTH", "HAREHAB1"]
    # yes, no, don't know, refused: all asked; NaN: not asked
    x = np.array([[1, 1], [2, 2], [3, 7], [4, 9], [5, np.nan]], dtype=float)
    np.testing.assert_array_equal(
        known_positives(x, names), [True, True, True, True, False]
    )
    # no leakage column in the data: no known positive
    np.testing.assert_array_equal(known_positives(x[:, :1], names[:1]), False)


def test_apply_known_positives():
    labels = np.array([-1, 1, -1, -1])
    known = np.array([True, True, False, False])
    out = apply_known_positives(labels, known)
    np.testing.assert_array_equal(out, [1, 1, -1, -1])
    np.testing.assert_array_equal(labels, [-1, 1, -1, -1])  # input not modified


def test_leakage_variants_are_valid_options_of_run_final():
    signature = inspect.signature(run_final)
    for variant, (options, label, key) in VARIANTS.items():
        signature.bind("method", "01_baseline", **options)  # raises if unknown
        assert key in ("f1_folds", "f1_rule_folds") and label, variant
