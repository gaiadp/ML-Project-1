"""Ensemble: weighted mean of the probabilities of two final models.

    p = a * p_newton + (1 - a) * p_boosting

The two components are the final settings of reg_logistic_newton (degree 2, log,
top 20 interactions: extra/04_final_interactions) and logistic_boosting (04_final).
They are trained on different features and the trees correct a linear model in a
different way, so their errors are partly different and their mean can be better
than each of them.

1. For each component: k-fold CV on the folds of every phase (same k_fold and
   seed), keeping the out-of-fold probability of each training row; then fit on
   the whole training set and predict x_test. Saved in
   results/ensemble/scores_<method>.npz, not computed again if it exists (--force).
2. Honest CV score of the ensemble: for each fold k, the weight a (grid WEIGHTS)
   and the decision threshold are chosen on the out-of-fold probabilities of the
   OTHER folds, then F1 is measured on fold k. The two components alone (a = 1,
   a = 0) are scored the same way, so the three rows of the table compare.
3. Final: a and the threshold chosen on all the out-of-fold probabilities, test
   predictions in submissions/ensemble_newton_boosting.csv, and the same with the
   known-positive rule (HAREHAB1 asked -> 1, src/rules.py) in
   submissions/ensemble_newton_boosting_rule.csv; results in
   results/ensemble/ensemble.json.
~1 h for logistic_boosting (5 folds + the full fit), a few minutes for Newton.

Usage (from the repository root):
    python experiments/ensemble.py
    python experiments/ensemble.py --force     # recompute the component scores
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import RESULTS_DIR, SUBMISSIONS_DIR, load_best  # noqa: E402
from helpers import create_csv_submission  # noqa: E402
from src.cross_validation import cross_validate, stratified_k_indices  # noqa: E402
from src.data import load_data  # noqa: E402
from src.metrics import best_threshold, f1_score, predict_labels  # noqa: E402
from src.models import fit, predict_scores  # noqa: E402
from src.pipeline import apply_preprocess, fit_preprocess  # noqa: E402
from src.rules import apply_known_positives, known_positives  # noqa: E402

# ---- settings ----
# component -> the result file of its final setting (pipeline and fit parameters)
COMPONENTS = {
    "reg_logistic_newton": "extra/04_final_interactions",
    "logistic_boosting": "04_final",
}
WEIGHTS = np.round(np.linspace(0, 1, 21), 2)  # a, the weight of the first component
K_FOLD = 5
SEED = 1  # the folds of every phase
OUT_DIR = os.path.join(RESULTS_DIR, "ensemble")
SUBMISSION = "ensemble_newton_boosting"
# ------------------


def component_scores(method, phase, x, x_test, y, names, force=False):
    """Out-of-fold and test probabilities of a final setting (cached in an .npz)."""
    path = os.path.join(OUT_DIR, f"scores_{method}.npz")
    if os.path.exists(path) and not force:
        print(f"{method}: scores from {os.path.relpath(path, ROOT)}")
        with np.load(path) as data:
            return data["oof"], data["test"]
    best = load_best(method, phase)
    config, params = best["pipeline"], best["fit"]
    print(f"{method}: setting of {phase}\n  fit {params}")
    start = time.time()
    out = cross_validate(x, y, names, config, method, K_FOLD, SEED, **params)
    print(f"  CV F1 {out['f1_mean']:.4f} +- {out['f1_std']:.4f}")
    state = fit_preprocess(x, y, names, config)
    w, _ = fit(method, y, apply_preprocess(x, names, state), seed=SEED, **params)
    test = predict_scores(method, w, apply_preprocess(x_test, names, state))
    print(f"  done in {(time.time() - start) / 60:.0f} min")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(path, oof=out["oof_scores"], test=test)
    return out["oof_scores"], test


def choose(p1, p2, y, weights):
    """Weight a and threshold with the highest F1 of a * p1 + (1 - a) * p2."""
    best = None
    for a in weights:
        threshold, f1 = best_threshold(a * p1 + (1 - a) * p2, y)
        if best is None or f1 > best[2]:
            best = (a, threshold, f1)
    return best


def honest_cv(p1, p2, y, known, weights, k_indices):
    """F1 per fold, with a and the threshold chosen on the other folds."""
    f1, f1_rule, chosen = [], [], []
    for k, val_idx in enumerate(k_indices):
        train_idx = np.concatenate([idx for j, idx in enumerate(k_indices) if j != k])
        a, threshold, _ = choose(p1[train_idx], p2[train_idx], y[train_idx], weights)
        scores = a * p1[val_idx] + (1 - a) * p2[val_idx]
        labels = predict_labels(scores, threshold)
        f1.append(f1_score(y[val_idx], labels))
        f1_rule.append(f1_score(y[val_idx], apply_known_positives(labels, known[val_idx])))
        chosen.append(a)
    return np.array(f1), np.array(f1_rule), np.array(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force", action="store_true", help="recompute the scores of the components"
    )
    args = parser.parse_args()

    x, x_test, y, _, test_ids, names = load_data()
    (m1, ph1), (m2, ph2) = COMPONENTS.items()
    p1, t1 = component_scores(m1, ph1, x, x_test, y, names, args.force)
    p2, t2 = component_scores(m2, ph2, x, x_test, y, names, args.force)
    known = known_positives(x, names)
    known_test = known_positives(x_test, names)
    del x, x_test
    k_indices = stratified_k_indices(y, K_FOLD, SEED)

    rows = {
        f"{m1} alone": [1.0],
        f"{m2} alone": [0.0],
        "ensemble": WEIGHTS,
    }
    print(
        f"\n{K_FOLD}-fold CV, weight and threshold chosen on the other folds\n"
        f"{'model':28s} {'F1':>16s} {'diff':>8s}  better  {'F1 + rule':>9s}  weight a"
    )
    table = {}
    for label, weights in rows.items():
        f1, f1_rule, chosen = honest_cv(p1, p2, y, known, weights, k_indices)
        table[label] = dict(
            f1_mean=f1.mean(),
            f1_std=f1.std(),
            f1_folds=f1,
            f1_rule=f1_rule.mean(),
            f1_rule_folds=f1_rule,
            weight_folds=chosen,
        )
    reference = table[f"{m2} alone"]["f1_folds"]
    for label, row in table.items():
        diff = row["f1_folds"] - reference
        print(
            f"{label:28s} {row['f1_mean']:.4f} +- {row['f1_std']:.4f} "
            f"{diff.mean():+8.4f}  {np.sum(diff > 0)}/{K_FOLD}     "
            f"{row['f1_rule']:9.4f}  {', '.join(f'{a:g}' for a in row['weight_folds'])}"
        )

    a, threshold, f1_oof = choose(p1, p2, y, WEIGHTS)
    y_pred = predict_labels(a * t1 + (1 - a) * t2, threshold)
    path = os.path.join(SUBMISSIONS_DIR, SUBMISSION + ".csv")
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    create_csv_submission(test_ids, y_pred, path)
    print(
        f"\nfinal: a = {a:g} ({m1}), threshold {threshold:.4f}, out-of-fold F1 "
        f"{f1_oof:.4f}\nwrote {path}: {np.mean(y_pred == 1):.2%} predicted positive"
    )
    # the same predictions + the known-positive rule (src/rules.py), a separate file
    y_rule = apply_known_positives(y_pred, known_test)
    path_rule = os.path.join(SUBMISSIONS_DIR, SUBMISSION + "_rule.csv")
    create_csv_submission(test_ids, y_rule, path_rule)
    print(
        f"wrote {path_rule}: known-positive rule, {known_test.sum()} test rows set "
        f"to 1 ({int(np.sum(y_rule != y_pred))} changed), "
        f"{np.mean(y_rule == 1):.2%} predicted positive"
    )

    result = dict(
        date=time.strftime("%Y-%m-%d %H:%M"),
        components=COMPONENTS,
        k_fold=K_FOLD,
        seed=SEED,
        weights=WEIGHTS,
        cv=table,
        final=dict(
            weight=a,
            threshold=threshold,
            f1_oof=f1_oof,
            test_positive_rate=np.mean(y_pred == 1),
            submission=os.path.relpath(path, ROOT),
            f1_rule_cv=table["ensemble"]["f1_rule"],
            test_known_positives=int(known_test.sum()),
            test_positive_rate_rule=np.mean(y_rule == 1),
            submission_rule=os.path.relpath(path_rule, ROOT),
        ),
    )
    with open(os.path.join(OUT_DIR, "ensemble.json"), "w") as f:
        json.dump(result, f, indent=2, default=_plain)
    print(f"saved {os.path.relpath(os.path.join(OUT_DIR, 'ensemble.json'), ROOT)}")


def _plain(v):
    """numpy scalars / arrays -> Python types, for JSON."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v))


if __name__ == "__main__":
    main()
