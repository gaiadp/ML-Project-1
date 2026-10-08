"""Evaluation metrics: accuracy, precision, recall, F1-score, confusion matrix.

Labels may be given either in {-1, 1} (as returned by load_csv_data and expected by
create_csv_submission) or in {0, 1} (as used by logistic regression): in both cases the
positive class (MICHD) is the label 1, everything else counts as negative.
"""

import numpy as np


def _is_positive(y):
    """Boolean mask of the positive class (label 1), for labels in {-1, 1} or {0, 1}."""
    return np.asarray(y) == 1


def confusion_counts(y_true, y_pred):
    """Return the four entries of the binary confusion matrix as (tp, fp, fn, tn)."""
    t = _is_positive(y_true)
    p = _is_positive(y_pred)
    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))  # type I error: healthy person flagged as MICHD
    fn = int(np.sum(t & ~p))  # type II error: MICHD case that is missed
    tn = int(np.sum(~t & ~p))
    return tp, fp, fn, tn


def accuracy(y_true, y_pred):
    """Fraction of correct predictions.

    Misleading on this dataset: only ~8.8% of the samples are positive, so always
    predicting "no MICHD" already gives ~91% accuracy.
    """
    return float(np.mean(_is_positive(y_true) == _is_positive(y_pred)))


def precision(y_true, y_pred):
    """TP / (TP + FP): among the people flagged as MICHD, the fraction that really are.

    Defined as 0 when nothing is predicted positive.
    """
    tp, fp, _, _ = confusion_counts(y_true, y_pred)
    return tp / (tp + fp) if tp + fp > 0 else 0.0


def recall(y_true, y_pred):
    """TP / (TP + FN): among the real MICHD cases, the fraction we detect (sensitivity).

    Defined as 0 when there are no positives in y_true.
    """
    tp, _, fn, _ = confusion_counts(y_true, y_pred)
    return tp / (tp + fn) if tp + fn > 0 else 0.0


def fbeta_score(y_true, y_pred, beta=1.0):
    """Weighted harmonic mean of precision and recall.

        F_beta = (1 + beta^2) TP / ((1 + beta^2) TP + beta^2 FN + FP)

    beta = 1 gives F1, where FP and FN weigh the same; beta > 1 (e.g. 2) makes a missed
    MICHD case (FN) beta^2 times as costly as a false alarm (FP).
    Defined as 0 when TP = 0.
    """
    tp, fp, fn, _ = confusion_counts(y_true, y_pred)
    b2 = beta**2
    denom = (1 + b2) * tp + b2 * fn + fp
    return (1 + b2) * tp / denom if tp > 0 else 0.0


def f1_score(y_true, y_pred):
    """F1 = 2 TP / (2 TP + FP + FN), the metric used on AIcrowd. Ignores true negatives."""
    return fbeta_score(y_true, y_pred, beta=1.0)


def predict_labels(scores, threshold):
    """Turn continuous scores into labels in {-1, 1}: positive iff score >= threshold."""
    return np.where(np.asarray(scores) >= threshold, 1, -1)


def best_threshold(scores, y_true, beta=1.0):
    """Decision threshold that maximizes F_beta (default F1) on (scores, y_true).

    scores can be anything where "higher = more likely MICHD": sigmoid probabilities of
    logistic regression, or raw outputs tx @ w of least squares / ridge.
    Use it on a VALIDATION set (or inside each CV fold), never on the test set.

    Every distinct score is a candidate threshold. Sorting the scores in decreasing order,
    predicting positive the first k samples gives TP = cumsum of positives up to k, so all
    candidates are evaluated in O(N log N) instead of O(N^2).

    Returns (threshold, best_fbeta); predict with predict_labels(scores, threshold).
    """
    scores = np.asarray(scores, dtype=float)
    pos = _is_positive(y_true)
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    p = pos[order]

    # TP and FP when the threshold is s[k] (samples 0..k predicted positive)
    tp = np.cumsum(p)
    fp = np.cumsum(~p)
    # With tied scores only the last occurrence is a valid cut: score >= t takes all ties
    last_of_tie = np.r_[s[1:] != s[:-1], True]
    tp, fp, thresholds = tp[last_of_tie], fp[last_of_tie], s[last_of_tie]
    fn = pos.sum() - tp

    b2 = beta**2
    fbeta = (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp)
    k = int(np.argmax(fbeta))
    return float(thresholds[k]), float(fbeta[k])
