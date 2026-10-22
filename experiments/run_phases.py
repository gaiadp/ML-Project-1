"""Run the phases of one method in order (01 -> 04), skipping the ones already done.

The phases of a method are its scripts experiments/<method>/0*.py (01_baseline,
02_degree, 02b_transforms, 03_interactions, 04_final; 02b only for ridge_regression
and reg_logistic_newton; only 01 and 04 for the extra methods other than
reg_logistic_newton; the extra_*.py scripts are not phases). Each phase reads the
best result of the previous one (experiments/common.py), so they must run in order.
A phase is skipped when its result results/<method>/<phase>.json already exists,
unless:
    --force                  every selected phase runs again
    --config file.json       the file changes some settings of the phase
    a previous phase runs    the later phases depend on it, they run again too
If you edit the settings block of a script by hand, use --force (or --config): the
existing result would be skipped otherwise.

The default settings are the settings blocks of the phase scripts (edit them there).
A JSON file can replace some of them, per phase, without touching the scripts:

    {
      "01_baseline": {"PIPELINES": ["v1"],
                      "FIT_GRID": {"lambda_": [1e-4], "n_trees": [200, 400]}},
      "04_final": {"FIT_OVERRIDES": {"n_trees": 400}}
    }

PIPELINES is a list of names (v0, full, v1), FIT_GRID a dict of lists (every
combination is tried, as experiments.common.param_grid); the other settings (DEGREES,
INTERACTION_TOP, INTERACTION_PAIRS, FROM_PHASE, PIPELINE_OVERRIDES, FIT_OVERRIDES,
K_FOLD, SEED) are written as in the scripts. A phase that runs again overwrites its
result in results/ (git keeps the old one).

Usage (from the repository root):
    python experiments/run_phases.py logistic_boosting            # missing phases only
    python experiments/run_phases.py logistic_boosting --dry-run  # print the plan
    python experiments/run_phases.py ridge_regression --phases 02 03 --force
    python experiments/run_phases.py logistic_boosting --config my_settings.json
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import param_grid, result_path  # noqa: E402
from src.models import ALL_METHODS  # noqa: E402
from src.pipeline import DEFAULT_CONFIG, V0_CONFIG, V1_CONFIG  # noqa: E402

# pipeline names usable in the PIPELINES setting of a config file
PIPELINE_CONFIGS = {"v0": V0_CONFIG, "full": DEFAULT_CONFIG, "v1": V1_CONFIG}


def phase_scripts(method):
    """Phase scripts of a method, in order: dict phase name -> path."""
    # 0[0-9]*: also the sub-phases, e.g. 02b_transforms (sorted after 02_degree)
    pattern = os.path.join(ROOT, "experiments", method, "0[0-9]*_*.py")
    return {os.path.basename(p)[:-3]: p for p in sorted(glob.glob(pattern))}


def load_settings(path):
    """Execute a phase script without running its main(): its global namespace.

    Like runpy.run_path, but returns the namespace itself, the one that main()
    reads: changing a setting in it changes what main() does.
    """
    namespace = {"__name__": "phase_settings", "__file__": path}
    with open(path) as f:
        exec(compile(f.read(), path, "exec"), namespace)
    return namespace


def convert_settings(settings):
    """Settings of a config file -> the Python values used by the phase scripts.

    Args:
        settings: dict setting name -> JSON value, for one phase

    Returns:
        dict with PIPELINES as {name: config}, FIT_GRID as a list of dicts, pairs as
        tuples; the other values unchanged
    """
    out = dict(settings)
    if "PIPELINES" in out:
        unknown = sorted(set(out["PIPELINES"]) - set(PIPELINE_CONFIGS))
        if unknown:
            raise ValueError(
                f"unknown pipelines {unknown}, expected {list(PIPELINE_CONFIGS)}"
            )
        out["PIPELINES"] = {name: PIPELINE_CONFIGS[name] for name in out["PIPELINES"]}
    if "FIT_GRID" in out:
        out["FIT_GRID"] = param_grid(**out["FIT_GRID"])
    if "INTERACTION_PAIRS" in out:
        pairs = out["INTERACTION_PAIRS"]
        out["INTERACTION_PAIRS"] = tuple(tuple(pair) for pair in pairs)
    return out


def apply_settings(namespace, settings, phase):
    """Replace settings of a loaded phase script; unknown names raise ValueError."""
    unknown = sorted(k for k in settings if not (k.isupper() and k in namespace))
    if unknown:
        raise ValueError(f"{phase}: unknown settings {unknown}")
    namespace.update(convert_settings(settings))


def plan(phases, done, changed, force):
    """Which phases run: [(phase, True / False)], in order.

    A phase runs if forced, if its result is missing (not in done), if its settings
    are changed by the config file (in changed), or if a previous phase runs.
    """
    out = []
    previous_runs = False
    for phase in phases:
        runs = force or previous_runs or phase not in done or phase in changed
        out.append((phase, runs))
        previous_runs = previous_runs or runs
    return out


def select_phases(scripts, requested):
    """Phases named in requested (full name or prefix, e.g. "02"), in script order."""
    if not requested:
        return list(scripts)
    selected = [p for p in scripts if any(p.startswith(r) for r in requested)]
    missing = [r for r in requested if not any(p.startswith(r) for p in scripts)]
    if missing:
        raise ValueError(f"no phase {missing}, the phases are {list(scripts)}")
    return selected


def _describe(namespace):
    """Settings of a loaded phase script, short enough to print."""
    settings = {
        k: v
        for k, v in namespace.items()
        if k.isupper() and k != "ROOT" and not k.endswith("_CONFIG")
    }
    if "PIPELINES" in settings:
        settings["PIPELINES"] = list(settings["PIPELINES"])
    return "\n".join(f"    {k} = {v}" for k, v in settings.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("method", choices=ALL_METHODS)
    parser.add_argument(
        "--phases", nargs="+", metavar="PHASE", help="e.g. 02 03 (default: all)"
    )
    parser.add_argument("--config", help="JSON file of settings to replace, per phase")
    parser.add_argument(
        "--force", action="store_true", help="run again also the phases already done"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and the settings only"
    )
    args = parser.parse_args()

    scripts = phase_scripts(args.method)
    try:
        phases = select_phases(scripts, args.phases)
        settings = {}
        if args.config:
            with open(args.config) as f:
                settings = json.load(f)
            unknown = sorted(set(settings) - set(scripts))
            if unknown:
                raise ValueError(f"config file: no phase {unknown} for {args.method}")
        done = {p for p in phases if os.path.exists(result_path(args.method, p))}
        steps = plan(phases, done, set(settings), args.force)
        # load and check the settings of every phase that runs before starting:
        # a wrong name in the config file must not stop the chain after hours
        namespaces = {}
        for phase, runs in steps:
            if runs:
                namespaces[phase] = load_settings(scripts[phase])
                apply_settings(namespaces[phase], settings.get(phase, {}), phase)
    except ValueError as error:
        parser.error(str(error))

    for phase, runs in steps:
        result = os.path.relpath(result_path(args.method, phase), ROOT)
        if not runs:
            print(f"skip {phase}: {result} exists (--force to run it again)")
            continue
        print(f"run {phase} -> {result}\n{_describe(namespaces[phase])}")
        if not args.dry_run:
            namespaces[phase]["main"]()


if __name__ == "__main__":
    main()
