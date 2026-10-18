"""Shared code of the per-method experiment scripts in experiments/<method>/.

Each phase script only sets its parameters and calls run_grid (phases 1-3) or
run_final (phase 4). Every phase writes results/<method>/<phase>.json; the next phase
reads its best row with load_best, so the numbers never have to be copied by hand.

Phases (one folder per method of implementations.py, same files in each):
    01_baseline.py      pipeline v0 / full / v1 (EDA feature selection), no expansion
    02_degree.py        polynomial degree, on the best pipeline of 01
    03_interactions.py  number of interaction columns, on the best pipeline of 02
    04_final.py         CV of the best setting, fit on the full training set,
                        test predictions in submissions/<method>.csv
"""

import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from helpers import create_csv_submission  # noqa: E402
from src.cross_validation import cross_validate, cross_validate_grid  # noqa: E402
from src.data import load_data  # noqa: E402
from src.metrics import best_threshold, predict_labels  # noqa: E402
from src.models import fit, predict_scores  # noqa: E402
from src.pipeline import apply_preprocess, fit_preprocess  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "results")
SUBMISSIONS_DIR = os.path.join(ROOT, "submissions")


def param_grid(**values):
    """All the combinations of the given parameter values, as a list of dicts.

    param_grid(lambda_=[1e-4, 1e-2], max_iters=[100]) ->
        [{"lambda_": 1e-4, "max_iters": 100}, {"lambda_": 1e-2, "max_iters": 100}]
    param_grid() -> [{}] (one fit with the default parameters)
    """
    keys = list(values)
    return [
        {k: _plain(v) for k, v in zip(keys, combo)}
        for combo in itertools.product(*(values[k] for k in keys))
    ]


def result_path(method, phase):
    return os.path.join(RESULTS_DIR, method, phase + ".json")


def load_result(method, phase):
    """The JSON written by run_grid / run_final for this method and phase."""
    path = result_path(method, phase)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found: run experiments/{method}/{phase}.py"
        )
    with open(path) as f:
        return json.load(f)


def load_best(method, phase):
    """Best row of a phase: dict with "pipeline" (full config), "fit" and the scores."""
    return load_result(method, phase)["best"]


def run_grid(method, phase, pipelines, fit_grid, k_fold=5, seed=1):
    """k-fold CV of every pipeline x every set of fit parameters, on the same folds.

    Prints a table, picks the row with the highest mean validation F1 (among those
    whose scores are finite: a too large gamma makes GD diverge) and writes
    results/<method>/<phase>.json.

    Args:
        method: one of src.models.METHODS
        phase: name of the phase, also the name of the JSON file
        pipelines: dict label -> full preprocessing config
        fit_grid: list of dicts of fit parameters (param_grid)
        k_fold, seed: folds of the CV; seed is also the seed of SGD / undersampling

    Returns:
        the saved dict: settings, rows (one per pipeline x fit), best
    """
    x, x_test, y, _, _, names = load_data()
    del x_test
    print(f"{method}, {phase}: {len(pipelines)} pipelines x {len(fit_grid)} fits")
    rows = []
    for label, config in pipelines.items():
        start = time.time()
        outs = cross_validate_grid(x, y, names, config, method, fit_grid, k_fold, seed)
        seconds = (time.time() - start) / len(fit_grid)
        for params, out in zip(fit_grid, outs):
            row = _row(label, config, params, out, seconds)
            rows.append(row)
            print(_format_row(row))

    finite = [r for r in rows if r["finite"]] or rows
    best = max(finite, key=lambda r: r["f1_mean"])
    print(f"\nbest: {_format_row(best)}")
    result = dict(
        method=method,
        phase=phase,
        date=time.strftime("%Y-%m-%d %H:%M"),
        k_fold=k_fold,
        seed=seed,
        rows=rows,
        best=best,
    )
    _save(result)
    return result


def run_final(
    method, from_phase, pipeline_overrides=None, fit_overrides=None, k_fold=5, seed=1
):
    """Final model of a method: best setting of from_phase, CV score, test predictions.

    The decision threshold is the one that maximizes F1 on the out-of-fold scores of
    the CV (every training row scored by a model that did not see it); the model is
    then refitted on the whole training set and predicts x_test.

    Args:
        method: one of src.models.METHODS
        from_phase: phase whose best row is used (e.g. "03_interactions")
        pipeline_overrides, fit_overrides: dicts that change the best setting
        k_fold, seed: as run_grid

    Returns:
        the saved dict, also written to results/<method>/04_final.json; the
        predictions go to submissions/<method>.csv
    """
    best = load_best(method, from_phase)
    config = dict(best["pipeline"], **(pipeline_overrides or {}))
    params = dict(best["fit"], **(fit_overrides or {}))
    x, x_test, y, _, test_ids, names = load_data()
    print(f"{method}: setting of {from_phase}\n  pipeline {config}\n  fit {params}")

    out = cross_validate(x, y, names, config, method, k_fold, seed, **params)
    threshold, f1_oof = best_threshold(out["oof_scores"], y)
    print(
        f"CV F1 {out['f1_mean']:.4f} +- {out['f1_std']:.4f}; "
        f"out-of-fold threshold {threshold:.4f} (F1 {f1_oof:.4f})"
    )

    state = fit_preprocess(x, y, names, config)
    tx = apply_preprocess(x, names, state)
    del x
    w, loss = fit(method, y, tx, seed=seed, **params)
    del tx
    tx_test = apply_preprocess(x_test, names, state)
    y_pred = predict_labels(predict_scores(method, w, tx_test), threshold)
    path = os.path.join(SUBMISSIONS_DIR, method + ".csv")
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    create_csv_submission(test_ids, y_pred, path)
    print(f"wrote {path}: {np.mean(y_pred == 1):.2%} predicted positive")

    result = dict(
        method=method,
        phase="04_final",
        date=time.strftime("%Y-%m-%d %H:%M"),
        from_phase=from_phase,
        k_fold=k_fold,
        seed=seed,
        best=dict(
            _row("final", config, params, out, None),
            threshold_oof=threshold,
            f1_oof=f1_oof,
            train_loss=loss,
            n_features=len(state["out_names"]),
            test_positive_rate=np.mean(y_pred == 1),
            submission=os.path.relpath(path, ROOT),
        ),
    )
    _save(result)
    return result


def _row(label, config, params, out, seconds):
    return dict(
        label=label,
        pipeline=config,
        fit=params,
        f1_mean=out["f1_mean"],
        f1_std=out["f1_std"],
        f1_folds=out["f1"],
        f1_val_best=float(np.mean(out["f1_val_best"])),
        threshold=float(np.mean(out["threshold"])),
        n_features=int(out["n_features"][0]),
        finite=bool(np.isfinite(out["oof_scores"]).all()),
        seconds=seconds,
    )


def _format_row(row):
    fit_text = ", ".join(f"{k}={v:g}" for k, v in row["fit"].items()) or "-"
    flag = "" if row["finite"] else "  DIVERGED"
    return (
        f"{row['label']:16s} {fit_text:34s} F1 {row['f1_mean']:.4f} +- "
        f"{row['f1_std']:.4f}  (val-best {row['f1_val_best']:.4f})  "
        f"{row['n_features']:5d} features{flag}"
    )


def _plain(v):
    """numpy scalars / arrays -> Python types, for JSON."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, tuple):
        return [_plain(e) for e in v]
    return v


def _save(result):
    path = result_path(result["method"], result["phase"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=_plain)
    print(f"saved {os.path.relpath(path, ROOT)}")
