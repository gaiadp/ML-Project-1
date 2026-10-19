"""Known positives: rows whose MICHD label is certain from the questionnaire.

HAREHAB1 ("after your heart attack, did you go to outpatient rehabilitation?") is
asked only to respondents who had a heart attack (CVDINFR4 = 1), which is part of the
definition of MICHD. So every row where it was asked is positive, whatever the answer:
1, 2, and also 7 / 9 (don't know / refused). In the training set: 773 rows, all MICHD.

The models never see this column (LABEL_LEAKAGE in src/feature_types.py): it reveals
the label instead of predicting it. This rule, applied AFTER the decision threshold,
only sets these rows to 1. It is a certain correction of the predictions for this
survey, not part of the predictive model: report the F1 with and without it.
"""

import numpy as np

from src.feature_types import LABEL_LEAKAGE


def known_positives(x, names):
    """Rows whose label is known to be positive: a LABEL_LEAKAGE question was asked.

    Args:
        x: RAW data (before codes_to_nan, which would turn 7 / 9 into NaN),
            shape=(N, D); NaN = the question was not asked
        names: the D column names

    Returns:
        bool array, shape=(N,)
    """
    col = {name: j for j, name in enumerate(names)}
    known = np.zeros(x.shape[0], dtype=bool)
    for name in LABEL_LEAKAGE:
        if name in col:
            known |= ~np.isnan(x[:, col[name]])
    return known


def apply_known_positives(labels, known):
    """Labels in {-1, 1} with the known positives set to 1 (a new array).

    Args:
        labels: predicted labels in {-1, 1}, shape=(N,)
        known: bool array from known_positives, shape=(N,)
    """
    labels = np.array(labels)
    labels[known] = 1
    return labels
