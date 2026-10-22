"""Screening of hand-made features for reg_logistic_newton, before training them.

Level 1, score test (seconds per feature group). The current model (best setting of
reg_logistic_newton/03_interactions) is fitted once on the whole training set. At its
minimum the residuals y - p are uncorrelated with every column it has; a new group
of columns X_new helps only through what is left in them. The score (Rao /
Lagrange multiplier) test measures it without refitting:

    g = X_new^T (p - y) / N                       gradient of the loss at w_new = 0
    I = H_nn - H_nb H_bb^-1 H_bn                  curvature left after the old columns
    chi2 = N g^T I^-1 g,  ~ chi2(q) if X_new adds nothing (q = its columns)

chi2 / (2 N) is the decrease of the mean logistic loss that refitting with X_new
would give (second-order estimate). Scale of reference: the same test for the top 20
interaction columns against the pipeline without them (02b_transforms), which in CV
gave F1 +0.0016. The test measures the loss, not F1, and only for the linear model.

Level 2, CV with Newton (~4 min per group): the groups significant at level 1
(Bonferroni over the groups) are added to the raw data as extra columns, the pipeline
of 03_interactions is run on them unchanged, and the 5-fold F1 is compared fold by
fold with 03_interactions (same folds and seed). Two side effects of the pipeline:
poly_degree adds the squares of the new columns with > 2 values, and a new column
among the 20 most correlated with the label changes the interaction_top pairs
(flagged by "top20?" at level 1).

Writes results/reg_logistic_newton/extra/feature_screening.json.

Usage (from the repository root):
    python experiments/feature_screening.py
    python experiments/feature_screening.py --level 1     # only the score test
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import load_best, result_path  # noqa: E402
from implementations import sigmoid  # noqa: E402
from src.cross_validation import cross_validate  # noqa: E402
from src.data import load_data  # noqa: E402
from src.extra_implementations.logistic_objective import hessian  # noqa: E402
from src.models import fit  # noqa: E402
from src.pipeline import apply_preprocess, fit_preprocess  # noqa: E402

# ---- settings ----
METHOD = "reg_logistic_newton"
FROM_PHASE = "03_interactions"  # the current model
REFERENCE_PHASE = "02b_transforms"  # the same pipeline without the interactions
ALPHA = 0.05  # level 1: significance of a group, Bonferroni over the groups
K_FOLD = 5
SEED = 1
OUT_PHASE = "extra/feature_screening"
# ------------------


def raw_indicators(x, names):
    """0/1 indicators of the conditions used by the groups, from the RAW codes.

    1 = yes in the BRFSS codebook unless noted; NaN, 7 / 9 (don't know / refused) and
    not asked count as "no".
    """
    col = {name: j for j, name in enumerate(names)}

    def v(name):
        return x[:, col[name]]

    def yes(name, codes=(1,)):
        return np.isin(v(name), codes).astype(float)

    age = v("_AGE80")
    male = yes("SEX")
    physhlth = v("PHYSHLTH")  # days 1-30, 88 = none, 77 / 99 = don't know
    return dict(
        age=age,
        male=male,
        diabetes=yes("DIABETE3"),
        hypertension=yes("_RFHYPE5", (2,)),
        high_chol=yes("TOLDHI2"),
        stroke=yes("CVDSTRK3"),
        kidney=yes("CHCKIDNY"),
        copd=yes("CHCCOPD1"),
        arthritis=yes("HAVARTH3"),
        current_smoker=yes("_SMOKER3", (1, 2)),
        ever_smoker=yes("_SMOKER3", (1, 2, 3)),
        obese=(np.nan_to_num(v("_BMI5")) >= 30).astype(float),
        inactive=yes("_TOTINDA", (2,)),
        # age of risk of the Framingham score: men 55+, women 65+
        old=((male == 1) & (age >= 55) | (male == 0) & (age >= 65)).astype(float),
        diff_walk=yes("DIFFWALK"),
        use_equip=yes("USEEQUIP"),
        limited=yes("QLACTLM2"),
        diff_dress=yes("DIFFDRES"),
        diff_alone=yes("DIFFALON"),
        unable_work=yes("EMPLOY1", (8,)),
        phys_14=((physhlth >= 14) & (physhlth <= 30)).astype(float),
        checkup_year=yes("CHECKUP1"),
        pneumo_vac=yes("PNEUVAC3"),
        flu_shot=yes("FLUSHOT6"),
        aspirin=yes("CVDASPRN"),
        bp_meds=yes("BPMEDS"),
    )


COMORBIDITIES = (
    "diabetes",
    "hypertension",
    "high_chol",
    "stroke",
    "kidney",
    "copd",
    "arthritis",
)
RISK_FACTORS = (
    "old",
    "current_smoker",
    "hypertension",
    "high_chol",
    "diabetes",
    "obese",
    "inactive",
)
LIMITATIONS = (
    "diff_walk",
    "use_equip",
    "limited",
    "diff_dress",
    "diff_alone",
    "unable_work",
    "phys_14",
)
CARE = ("checkup_year", "pneumo_vac", "flu_shot", "aspirin", "bp_meds")
PAIRS = (
    ("age", "male"),
    ("age", "diabetes"),
    ("age", "hypertension"),
    ("age", "high_chol"),
    ("age", "current_smoker"),
    ("age", "ever_smoker"),
    ("age", "stroke"),
    ("male", "ever_smoker"),
    ("male", "diabetes"),
    ("diabetes", "hypertension"),
    ("hypertension", "high_chol"),
    ("obese", "diabetes"),
    ("diabetes", "kidney"),
    ("age", "comorbidities"),
)


def _count(ind, keys):
    return sum(ind[k] for k in keys)


def _at_least(count, levels, prefix):
    """Cumulative indicators count >= k: with the old columns, the same span as a
    one-hot of the count, but each column is the step from k - 1 to k."""
    return {f"{prefix}>={k}": (count >= k).astype(float) for k in levels}


def feature_groups(x, names):
    """Group name -> dict column name -> values, shape=(N,) each."""
    ind = raw_indicators(x, names)
    comorb = _count(ind, COMORBIDITIES)
    ind["comorbidities"] = comorb
    groups = {
        # a sum of columns the model has: the test should give ~0 (sanity check)
        "comorbidity count (number)": {"comorbidities": comorb},
        "comorbidity count (one-hot)": _at_least(comorb, (2, 3, 4), "comorb"),
        "risk factor count (one-hot)": _at_least(
            _count(ind, RISK_FACTORS), (2, 3, 4, 5), "risk"
        ),
        "functional limitation (one-hot)": _at_least(
            _count(ind, LIMITATIONS), (1, 2, 3, 4), "limit"
        ),
        "health care contact (one-hot)": _at_least(
            _count(ind, CARE), (1, 2, 3, 4), "care"
        ),
        "clinical interactions": {
            f"{a}*{b}": _standardize(ind[a]) * _standardize(ind[b]) for a, b in PAIRS
        },
    }
    return groups


def _standardize(v):
    std = v.std()
    return (v - v.mean()) / (std if std > 0 else 1.0)


def score_test(tx, w, lambda_, h_bb, y01, x_new):
    """Score test of the columns x_new (shape=(N, q)) added to the model (tx, w).

    Returns:
        chi2 statistic, degrees of freedom q
    """
    n = tx.shape[0]
    p = sigmoid(tx @ w)
    s = p * (1 - p)
    g = x_new.T @ (p - y01) / n
    h_nb = (x_new * s[:, None]).T @ tx / n
    h_nn = x_new.T @ (x_new * s[:, None]) / n + 2 * lambda_ * np.eye(x_new.shape[1])
    info = h_nn - h_nb @ np.linalg.solve(h_bb, h_nb.T)
    return float(n * g @ np.linalg.lstsq(info, g, rcond=None)[0]), x_new.shape[1]


def chi2_pvalue(stat, df):
    """P(chi2(df) >= stat), Wilson-Hilferty approximation (no scipy)."""
    if stat <= 0:
        return 1.0
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def _corr_with(y01, a):
    """|Pearson r| of each column of a with y01, shape=(a.shape[1],)."""
    a = a - a.mean(axis=0)
    yc = y01 - y01.mean()
    den = np.sqrt((a * a).sum(axis=0) * (yc @ yc))
    return np.abs(a.T @ yc) / np.where(den > 0, den, 1.0)


def level1(x, y, names, best):
    """Score test of every group and of the reference block. Returns the rows."""
    config, params = best["pipeline"], best["fit"]
    lambda_ = params["lambda_"]
    y01 = (np.asarray(y) == 1).astype(float)
    n = len(y01)
    start = time.time()
    state = fit_preprocess(x, y, names, config)
    tx = apply_preprocess(x, names, state)
    out_names = state["out_names"]
    print(f"current model: {tx.shape[1]} columns ({time.time() - start:.0f} s)")

    # reference: the interaction columns against the model without them
    product = np.array(["*" in name for name in out_names])
    tx_ref = tx[:, ~product]
    w_ref, _ = fit(METHOD, y, tx_ref, seed=SEED, **params)
    stat, df = score_test(
        tx_ref, w_ref, lambda_, hessian(tx_ref, w_ref, lambda_), y01, tx[:, product]
    )
    del tx_ref
    rows = [_row("REFERENCE: top 20 interactions", stat, df, n, None)]

    w, _ = fit(METHOD, y, tx, seed=SEED, **params)
    h_bb = hessian(tx, w, lambda_)
    # |r| of the 20th column of the interaction_top ranking (columns before B11)
    base = np.array(["*" not in s and "^" not in s and "(" not in s for s in out_names])
    base[0] = False  # bias
    r_20th = np.sort(_corr_with(y01, tx[:, base]))[-20]

    groups = feature_groups(x, names)
    for label, cols in groups.items():
        x_new = np.column_stack([_standardize(v) for v in cols.values()])
        stat, df = score_test(tx, w, lambda_, h_bb, y01, x_new)
        top20 = bool(_corr_with(y01, x_new).max() > r_20th)
        rows.append(_row(label, stat, df, n, top20, list(cols)))
        if label == "clinical interactions":  # each pair alone too
            for k, name in enumerate(cols):
                stat, df = score_test(tx, w, lambda_, h_bb, y01, x_new[:, [k]])
                rows.append(_row(f"  {name}", stat, df, n, None, [name]))
    del tx

    print(
        f"\nLEVEL 1, score test (N = {n}); delta loss = estimated decrease of the "
        f"mean logistic loss\n{'group':34s} {'cols':>4s} {'chi2':>9s} "
        f"{'chi2/col':>8s} {'p':>9s} {'delta loss':>10s}  top20?"
    )
    for row in rows:
        flag = "" if row["top20"] is None else ("yes" if row["top20"] else "no")
        print(
            f"{row['label']:34s} {row['df']:4d} {row['chi2']:9.1f} "
            f"{row['chi2'] / row['df']:8.1f} {row['p']:9.1e} "
            f"{row['delta_loss']:10.2e}  {flag}"
        )
    return rows, groups


def _row(label, stat, df, n, top20, columns=None):
    return dict(
        label=label,
        df=df,
        chi2=stat,
        p=chi2_pvalue(stat, df),
        delta_loss=stat / (2 * n),
        top20=top20,
        columns=columns,
    )


def level2(x, y, names, best, groups, passed):
    """5-fold CV of the pipeline with each passed group added to the raw data."""
    reference = np.asarray(best["f1_folds"])
    runs = {label: groups[label] for label in passed}
    if len(passed) > 1:
        runs["all the passed groups"] = {
            name: v for label in passed for name, v in groups[label].items()
        }
    print(
        f"\nLEVEL 2, {K_FOLD}-fold CV with Newton; reference {FROM_PHASE} "
        f"F1 {reference.mean():.4f} +- {reference.std():.4f}"
    )
    rows = []
    for label, cols in runs.items():
        new = np.column_stack([_standardize(v) for v in cols.values()])
        # unlisted columns get the special codes 7 / 9 -> NaN (codes_to_nan)
        assert not np.isin(new, (7.0, 9.0)).any()
        new_names = [f"NEW_{name}" for name in cols]
        start = time.time()
        out = cross_validate(
            np.hstack([x, new]),
            y,
            list(names) + new_names,
            best["pipeline"],
            METHOD,
            K_FOLD,
            SEED,
            **best["fit"],
        )
        diff = out["f1"] - reference
        rows.append(
            dict(
                label=label,
                columns=list(cols),
                f1_mean=out["f1_mean"],
                f1_std=out["f1_std"],
                f1_folds=out["f1"],
                diff=float(diff.mean()),
                better=int(np.sum(diff > 0)),
                n_features=int(out["n_features"][0]),
            )
        )
        print(
            f"{label:34s} F1 {out['f1_mean']:.4f} +- {out['f1_std']:.4f}  "
            f"diff {diff.mean():+.4f}  better {np.sum(diff > 0)}/{K_FOLD}  "
            f"{int(out['n_features'][0])} features  "
            f"({(time.time() - start) / 60:.0f} min)"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()

    x, x_test, y, _, _, names = load_data()
    del x_test
    names = list(names)
    best = load_best(METHOD, FROM_PHASE)
    rows1, groups = level1(x, y, names, best)

    group_rows = [r for r in rows1[1:] if r["label"] in groups]
    threshold = ALPHA / len(group_rows)
    passed = [r["label"] for r in group_rows if r["p"] < threshold]
    print(f"\npassed (p < {ALPHA} / {len(group_rows)} = {threshold:.1e}): {passed}")

    rows2 = []
    if args.level == 2 and passed:
        rows2 = level2(x, y, names, best, groups, passed)

    result = dict(
        method=METHOD,
        phase=OUT_PHASE,
        date=time.strftime("%Y-%m-%d %H:%M"),
        from_phase=FROM_PHASE,
        reference_phase=REFERENCE_PHASE,
        k_fold=K_FOLD,
        seed=SEED,
        level1=rows1,
        passed=passed,
        level2=rows2,
    )
    path = result_path(METHOD, OUT_PHASE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=_plain)
    print(f"saved {os.path.relpath(path, ROOT)}")


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v))


if __name__ == "__main__":
    main()
