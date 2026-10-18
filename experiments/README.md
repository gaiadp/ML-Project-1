# Experiments

Scripts that produce the results of the report. Contents:

1. [Phases of a method](#phases-of-a-method)
2. [Settings of a phase](#settings-of-a-phase)
3. [Running the chain of phases: `run_phases.py`](#running-the-chain-of-phases-run_phasespy)
4. [One-off runs outside the phases](#one-off-runs-outside-the-phases)
5. [Baseline table: `baselines.py`](#baseline-table-baselinespy)
6. [Notes](#notes)

Other scripts: `eda.py` (exploratory data analysis, tables in `build/eda/`),
`imbalance.py` (class imbalance: threshold vs class weights vs undersampling),
`ablation.py` (ablation study).

## Phases of a method

One folder per method, `experiments/<method>/`, with the same phases in each. Every
phase starts from the best result of the previous one:

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `01_baseline.py` | pipeline v0 vs full pipeline vs v1 (full + feature selection from the EDA, `V1_CONFIG`), no expansion | – | `results/<method>/01_baseline.json` |
| `02_degree.py` | polynomial degree (`poly_degree`) | best of 01 | `results/<method>/02_degree.json` |
| `03_interactions.py` | interaction columns (`interaction_top`, `interaction_pairs`) | best of 02 | `results/<method>/03_interactions.json` |
| `04_final.py` | CV of the best setting, threshold on the out-of-fold scores, fit on all the training set | best of 03 (`FROM_PHASE`) | `results/<method>/04_final.json`, `submissions/<method>.csv` |

Phases 01–03 only compare settings in 5-fold CV (`run_grid` in `common.py`): every
combination of pipeline variant × hyperparameters, the pipeline fitted inside every
fold, the decision threshold chosen on the training part of every fold. Only the
pipeline of the best row passes to the next phase; the hyperparameters are searched
again in every phase (with more columns the best lambda can change). Phase 04
(`run_final`) takes pipeline **and** hyperparameters of the best row of `FROM_PHASE`,
chooses the threshold on the out-of-fold scores of a CV, fits the model on all the
training rows and writes the test predictions.

The 6 methods of `implementations.py` have the four phases. The extra methods of
`src/extra_implementations` (`reg_logistic_adaptive_gd`, `reg_logistic_newton`,
`reg_logistic_sophia`, `reg_logistic_unixgrad`, `random_forest`, `gradient_boosting`,
`logistic_boosting`) have only `01_baseline.py` and `04_final.py` (`FROM_PHASE =
"01_baseline"`): the four optimizers minimize the same loss as `reg_logistic_regression`
(phases 02 / 03 would be the same as there), and the trees build interactions by
themselves. The trees run on pipeline v1 only, since they are slow (minutes per fit).

A single phase can be run with the Run button or `python experiments/<method>/<phase>.py`;
to run the chain, use `run_phases.py` (below).

## Settings of a phase

The parameters of a run are written in the phase scripts themselves, in the block
between the two comment lines (the rest of the script is the same everywhere; the
logic is in `common.py` and `src/`). Example, `reg_logistic_regression/02_degree.py`:

```python
# ---- settings of the experiment ----
METHOD = "reg_logistic_regression"
FROM_PHASE = "01_baseline"
DEGREES = (1, 2, 3)
FIT_GRID = param_grid(lambda_=[1e-6, 1e-4, 1e-2], gamma=[0.3], max_iters=[500])
K_FOLD = 5
SEED = 1
# -------------------------------------
```

| Setting | Phases | Meaning |
|---|---|---|
| `METHOD` | all | name of the method (do not change) |
| `PIPELINES` | 01 | pipelines to compare, e.g. `{"v1": V1_CONFIG}` |
| `FIT_GRID` | 01, 02, 03 | hyperparameter grid: every combination is tried, e.g. `param_grid(lambda_=[1e-4, 1e-2], gamma=[0.3])` |
| `FROM_PHASE` | 02, 03, 04 | phase whose best result is the starting point; point it to an earlier phase to skip one |
| `DEGREES` | 02 | polynomial degrees to try, e.g. `(1, 2, 3)` |
| `INTERACTION_TOP` | 03 | numbers of most correlated columns whose products are added, e.g. `(0, 10, 20, 30)` |
| `INTERACTION_PAIRS` | 03 | pairs of columns chosen by hand, e.g. `(("_AGE80", "_BMI5"),)` |
| `PIPELINE_OVERRIDES` | 04 | changes to the best pipeline, e.g. `dict(select_features=False)` |
| `FIT_OVERRIDES` | 04 | changes to the best hyperparameters, e.g. `dict(n_trees=400)` |
| `K_FOLD`, `SEED` | all | 5 folds, seed 1 (also of SGD and of the trees): keep them in every phase and method, so that all the methods are compared on the same folds |

`FIT_GRID` can also contain the options of the methods, e.g. `pos_weight=[3.2]` or
`neg_per_pos=[3]` for the class imbalance, `n_trees=[200]` for the trees.

## Running the chain of phases: `run_phases.py`

`run_phases.py <method>` runs the phases of a method in order and **skips the phases
whose result already exists** in `results/<method>/<phase>.json`, so the training is
not repeated. It has no parameters of its own: it uses the settings of the phase
scripts, and a JSON file can replace some of them.

### When a phase runs

A phase runs if its result is missing, or with `--force`, or if the `--config` file
changes its settings, or if a previous phase runs (the later phases depend on it).
Otherwise it is skipped. With four phases:

| Situation | 01 | 02 | 03 | 04 |
|---|---|---|---|---|
| only the result of 01 exists | skip | run | run | run |
| all the results exist | skip | skip | skip | skip |
| all the results exist, `--force` | run | run | run | run |
| all the results exist, the config file changes 02 | skip | run | run | run |
| only 02 is missing | skip | run | run | run |

Before anything starts, the settings of every phase that will run are loaded and
checked: a wrong name in the config file stops at once, not after hours of phase 01.

### Commands

From the repository root, with the conda environment active:

```bash
# all the options
python experiments/run_phases.py --help

# the plan only: which phases would run or be skipped, with which settings (no training)
python experiments/run_phases.py logistic_boosting --dry-run

# the missing phases: for logistic_boosting 01 exists, so only 04 (final model + submission)
python experiments/run_phases.py logistic_boosting

# reg_logistic_regression: 01 exists, so 02, 03 and 04
python experiments/run_phases.py reg_logistic_regression

# only phase 02 (03 and 04 are not touched)
python experiments/run_phases.py reg_logistic_regression --phases 02

# all the phases again, also the ones already done
python experiments/run_phases.py ridge_regression --force

# with settings from a JSON file, keeping the output in a log
python experiments/run_phases.py logistic_boosting \
    --config experiments/configs/logistic_boosting_more_trees.json 2>&1 | tee -a run.log
```

`--phases` takes full phase names or prefixes (`02`, `02_degree`). Example of
`--dry-run` output:

```
skip 01_baseline: results/logistic_boosting/01_baseline.json exists (--force to run it again)
run 04_final -> results/logistic_boosting/04_final.json
    METHOD = logistic_boosting
    FROM_PHASE = 01_baseline
    PIPELINE_OVERRIDES = {}
    FIT_OVERRIDES = {}
    K_FOLD = 5
    SEED = 1
```

### Config files

A config file is a JSON file for **one method** (the method is given on the command
line). The keys are phase names; inside each one, only the settings to replace, the
others stay as in the script. The ready-made examples below are in
[`configs/`](configs/); a new file can go anywhere, its path is given to `--config`.

- `PIPELINES`: a list of names among `"v0"`, `"full"`, `"v1"`.
- `FIT_GRID`: a dict of lists; every combination is tried, as `param_grid`.
- The other settings: as in the scripts, with lists instead of tuples.
- JSON syntax: lowercase `true` / `false`, no comments, numbers like `1e-4` are fine.

**`FIT_GRID` replaces the whole grid**: the hyperparameters it does not mention get the
defaults of `src.models.fit` (`gamma=0.1`, `max_iters=100`), so for the logistic
methods write `"gamma"` and `"max_iters"` too. **`FIT_OVERRIDES` and
`PIPELINE_OVERRIDES` of phase 04 change only the keys they mention**: the rest comes
from the best row of `FROM_PHASE`.

Example 1, `configs/logistic_boosting_more_trees.json`: more trees for the hybrid
model. Phase 01 runs again with this grid, then phase 04 (it depends on 01).

```json
{
  "01_baseline": {
    "FIT_GRID": {"lambda_": [1e-4], "max_iters": [20], "n_trees": [400],
                 "max_depth": [3, 4], "learning_rate": [0.05]}
  }
}
```

Example 2, `configs/logistic_boosting_final_400_trees.json`: only the final model, with
one value forced. Phase 01 is skipped (its result exists); phase 04 starts from its
best row, with 400 trees.

```json
{
  "04_final": {"FIT_OVERRIDES": {"n_trees": 400}}
}
```

Example 3, `configs/ridge_regression_degree_2.json`: degrees 1–2 and a finer lambda grid
for ridge (ridge only uses `lambda_`, so the grid needs nothing else).

```json
{
  "02_degree": {
    "DEGREES": [1, 2],
    "FIT_GRID": {"lambda_": [1e-5, 1e-4, 1e-3]}
  }
}
```

Example 4, `configs/reg_logistic_regression_pairs.json`: interactions with pairs chosen
by hand (the lambda grid of the script is kept).

```json
{
  "03_interactions": {
    "INTERACTION_TOP": [0, 10],
    "INTERACTION_PAIRS": [["_AGE80", "SEX"], ["_AGE80", "_BMI5"]]
  }
}
```

Example 5: one pipeline only in phase 01, and the final model without the feature
selection (written inline here, not in `configs/`).

```json
{
  "01_baseline": {"PIPELINES": ["v1"]},
  "04_final": {"PIPELINE_OVERRIDES": {"select_features": false}}
}
```

### Results

- Phases 01–03: `results/<method>/<phase>.json`, with every row of the grid and the best
  one. Every row stores the pipeline and the hyperparameters it used, so a run with a
  config file stays reproducible from its JSON.
- Phase 04: `results/<method>/04_final.json` and the predictions
  `submissions/<method>.csv` (gitignored), ready for AIcrowd.

### Things to remember

- After editing the settings block of a script by hand, use `--force` (or
  `--phases <phase> --force`): the existing result would be skipped otherwise.
- A phase that runs again overwrites its JSON in `results/` (git keeps the old version
  until you commit the new one).
- With `--phases`, the settings of the config file for the phases not selected are ignored.
- Changing `K_FOLD` or `SEED` changes the folds: the results can no longer be compared
  fold by fold with the other methods.

## One-off runs outside the phases

A grid that is not one of the phases and should not replace its result (e.g. a quick
check of a finer lambda grid) goes to `results/<method>/extra/<name>.json`: give
`run_grid` a phase name starting with `extra/`, so the files read by the next phases
and by `baselines.py` (`results/<method>/0*.json`) do not change:

```bash
python -c "
import runpy
from experiments.common import param_grid, run_grid
s = runpy.run_path('experiments/logistic_boosting/01_baseline.py')
run_grid(s['METHOD'], 'extra/01_more_trees', {'v1': s['V1_CONFIG']},
         param_grid(lambda_=[1e-4], max_iters=[20], n_trees=[400], max_depth=[3, 4],
                    learning_rate=[0.05]), 5, 1)
"
```

`load_best` reads these files too: `FROM_PHASE = "extra/01_more_trees"` in a phase
(or in a config file) starts from them.

## Baseline table: `baselines.py`

`baselines.py` puts phase 01 of every method in one table (best F1 per pipeline, best
setting), next to trivial rules computed on the same folds (everyone MICHD, best rule
on the age group, on age group x sex, on GENHLTH). There is no separate tuning script:
the hyperparameters are the grids of the phases, the threshold is chosen in every fold.

```bash
python experiments/baselines.py            # from results/*/01_baseline.json (seconds)
python experiments/baselines.py --rerun    # run phase 01 of every method again first (hours)
python experiments/baselines.py --rerun ridge_regression least_squares   # only some methods
```

It writes `results/baselines.md` and `results/baselines.tex` (a LaTeX tabular for the
report). `--rerun` runs the `01_baseline.py` scripts as they are (all their pipelines).

## Notes

- F1 is measured on the validation part of each fold with the threshold chosen on the
  training part (`src/cross_validation.py`); `val-best` in the tables is the optimistic
  F1 with the threshold chosen on the validation part itself. Report `f1_mean`.
- The polynomial and interaction columns are pipeline step B11 (`src/pipeline.py`):
  they are fitted inside each fold, and the interaction columns chosen by
  `interaction_top` use the labels of the training part only.
- On the full pipeline, ~317 columns become ~400 with degree 2 and ~480 with degree 3;
  `interaction_top = k` adds k (k - 1) / 2 columns. With 260k training rows per fold,
  ~600 columns already take ~1.3 GB per matrix.
