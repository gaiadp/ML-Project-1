"""Tests for experiments/run_phases.py (no data, no training)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_phases import (  # noqa: E402
    PIPELINE_CONFIGS,
    apply_settings,
    convert_settings,
    load_settings,
    phase_scripts,
    plan,
    select_phases,
)

PHASES = ["01_baseline", "02_degree", "03_interactions", "04_final"]


def test_phase_scripts():
    with_transforms = PHASES[:2] + ["02b_transforms"] + PHASES[2:]
    assert list(phase_scripts("ridge_regression")) == with_transforms
    assert list(phase_scripts("reg_logistic_newton")) == with_transforms
    assert list(phase_scripts("random_forest")) == ["01_baseline", "04_final"]


def test_select_phases():
    scripts = dict.fromkeys(PHASES)
    assert select_phases(scripts, None) == PHASES
    assert select_phases(scripts, ["03", "02"]) == ["02_degree", "03_interactions"]
    # a prefix selects the sub-phases too
    scripts = dict.fromkeys(["02_degree", "02b_transforms", "03_interactions"])
    assert select_phases(scripts, ["02"]) == ["02_degree", "02b_transforms"]
    with pytest.raises(ValueError):
        select_phases(scripts, ["05"])


def test_plan():
    def runs(done, changed=(), force=False):
        return [r for _, r in plan(PHASES, set(done), set(changed), force)]

    # 01 and 02 done: only the missing ones
    assert runs(PHASES[:2]) == [False, False, True, True]
    # all done: nothing, unless forced
    assert runs(PHASES) == [False] * 4
    assert runs(PHASES, force=True) == [True] * 4
    # changed settings of 02: 02 and the phases after it
    assert runs(PHASES, changed=["02_degree"]) == [False, True, True, True]
    # a missing phase in the middle: it and the later ones
    done = ["01_baseline", "03_interactions", "04_final"]
    assert runs(done) == [False, True, True, True]


def test_convert_settings():
    out = convert_settings(
        {
            "PIPELINES": ["v1"],
            "FIT_GRID": {"lambda_": [1e-4, 1e-2], "max_iters": [20]},
            "INTERACTION_PAIRS": [["_AGE80", "SEX"]],
            "K_FOLD": 3,
        }
    )
    assert out["PIPELINES"] == {"v1": PIPELINE_CONFIGS["v1"]}
    assert out["FIT_GRID"] == [
        {"lambda_": 1e-4, "max_iters": 20},
        {"lambda_": 1e-2, "max_iters": 20},
    ]
    assert out["INTERACTION_PAIRS"] == (("_AGE80", "SEX"),) and out["K_FOLD"] == 3
    with pytest.raises(ValueError):
        convert_settings({"PIPELINES": ["v2"]})


def test_settings_of_a_phase_script():
    path = phase_scripts("logistic_boosting")["01_baseline"]
    namespace = load_settings(path)
    assert namespace["METHOD"] == "logistic_boosting" and callable(namespace["main"])
    apply_settings(namespace, {"FIT_GRID": {"n_trees": [400]}}, "01_baseline")
    assert namespace["FIT_GRID"] == [{"n_trees": 400}]
    with pytest.raises(ValueError):
        apply_settings(namespace, {"FIT_GRIDS": {}}, "01_baseline")
