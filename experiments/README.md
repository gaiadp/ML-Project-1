# Experiments

Scripts that produce the results of the report. Contents:

1. [Phases of a method](#phases-of-a-method)
2. [Settings of a phase](#settings-of-a-phase)
3. [Running the chain of phases: `run_phases.py`](#running-the-chain-of-phases-run_phasespy)
4. [One-off runs outside the phases](#one-off-runs-outside-the-phases)
5. [Extra scripts: interactions, ensemble, feature screening](#extra-scripts-interactions-ensemble-feature-screening)
6. [Baseline table: `baselines.py`](#baseline-table-baselinespy)
7. [Ablation study: `ablation.py`](#ablation-study-ablationpy)
8. [HAREHAB1, the label leakage column: `leakage.py`](#harehab1-the-label-leakage-column-leakagepy)
9. [Notes](#notes)

**Official phases and extra scripts.** The phases are the scripts
`experiments/<method>/0*_*.py`; `run_phases.py` runs them in order and each writes
`results/<method>/<phase>.json`. The extra scripts (`<method>/extra_*.py`,
`ensemble.py`, `feature_screening.py`) are run by hand, are never run by
`run_phases.py`, and write only to `results/<method>/extra/` or `results/ensemble/`:
they never overwrite a phase file such as `04_final.json`.

**Which model is submitted:** the ensemble of `ensemble.py`, 0.35 x
`reg_logistic_newton` (`extra/04_final_interactions`) + 0.65 x model A, without
`HAREHAB1` and without the known-positive rule (`submissions/ensemble_newton_boosting.csv`,
AIcrowd F1 0.446), reproduced by `run.py`. Here "model A" is `logistic_boosting` phase
04 (`results/logistic_boosting/04_final.json`), the second component of the ensemble
(alone: `run_model_a.py`) and the reference of `ablation.py` and `leakage.py`
(`FINAL_METHOD`, `METHOD`); the ensemble of `ensemble.py` is described below.

Every method folder has a short `README.md`; the files of `results/` (what writes them,
cost, reproducibility) are listed in [results/README.md](../results/README.md).

Other scripts: `eda.py` (exploratory data analysis, tables and figures in `build/eda/`,
two summary figures in `report/figures/`, not used by the final report),
`imbalance.py` (class imbalance: threshold vs class weights vs undersampling; its
table goes only to the gitignored `build/imbalance.txt`, so its results are not in the
repository: run it again to see them; not used by the report),
`smoke_test.py` (~30 s check on 20k real rows before a long run: every pipeline
variant of the ablation, model A and the models of the ablation train and predict;
writes nothing).

The exploratory analysis (`eda.py`, analyses A1–A9), its outputs and the column lists it
led to are described in detail in [EDA.md](EDA.md).

## Phases of a method

One folder per method, `experiments/<method>/`, with the same phases in each. Every
phase starts from the best result of the previous one:

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `01_baseline.py` | pipeline v0 vs full pipeline vs v1 (full + feature selection from the EDA, `V1_CONFIG`), no expansion (the extra methods compare fewer pipelines, see below) | – | `results/<method>/01_baseline.json` |
| `02_degree.py` | polynomial degree (`poly_degree`) | best of 01 | `results/<method>/02_degree.json` |
| `02b_transforms.py` | only `reg_logistic_newton` and `ridge_regression`: sqrt / log of the continuous and count columns (`extra_transforms`, see [src/README.md](../src/README.md)), after a printed correlation check (`check_transforms`) | best of 02 | `results/<method>/02b_transforms.json` |
| `03_interactions.py` | interaction columns (`interaction_top`, `interaction_pairs`) | best of 02 (of 02b for `reg_logistic_newton`) | `results/<method>/03_interactions.json` |
| `04_final.py` | CV of the best setting, threshold on the out-of-fold scores, fit on all the training set | best of 03 (`FROM_PHASE`) | `results/<method>/04_final.json`, `submissions/<method>.csv` |

Phases 01–03 (and 02b) only compare settings in 5-fold CV (`run_grid` in `common.py`): every
combination of pipeline variant × hyperparameters, the pipeline fitted inside every
fold, the decision threshold chosen on the training part of every fold. Only the
pipeline of the best row passes to the next phase; the hyperparameters are searched
again in every phase (with more columns the best lambda can change). Phase 04
(`run_final`) takes pipeline **and** hyperparameters of the best row of `FROM_PHASE`,
chooses the threshold on the out-of-fold scores of a CV, fits the model on all the
training rows and writes the test predictions.

Which phases each method has:

- The 6 methods of `implementations.py`: 01, 02, 03, 04; `ridge_regression` also
  02b. Its `03_interactions.py` starts from `02_degree` (`FROM_PHASE`), not from
  `02b_transforms`.
- `reg_logistic_newton`: 01, 02, 02b, 03, 04. Its `04_final.py` keeps
  `FROM_PHASE = "01_baseline"`, so `04_final.json` is the model on pipeline v1; the
  final model with the expansions of 02-03 is the extra script
  `extra_final_interactions.py` (below).
- The other extra methods of `src/extra_implementations` (`reg_logistic_adaptive_gd`,
  `reg_logistic_sophia`, `reg_logistic_unixgrad`, `random_forest`,
  `gradient_boosting`, `logistic_boosting`): only `01_baseline.py` and `04_final.py`
  (`FROM_PHASE = "01_baseline"`). The optimizers minimize the same loss as
  `reg_logistic_regression`, and the trees build interactions by themselves (for
  `logistic_boosting` the interactions are tried by an extra script, below).

The trees run on pipeline v1 only, since they are slow (minutes per fit); the four
optimizers compare full and v1 (their saved `01_baseline.json` have only v1 rows: see
[results/README.md](../results/README.md)).

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
| `FIT_GRID` | 01, 02, 02b, 03 | hyperparameter grid: every combination is tried, e.g. `param_grid(lambda_=[1e-4, 1e-2], gamma=[0.3])` |
| `FROM_PHASE` | 02, 02b, 03, 04 | phase whose best result is the starting point; point it to an earlier phase to skip one |
| `DEGREES` | 02, 02b | polynomial degrees to try, e.g. `(1, 2, 3)`; in 02b `None` = only the degree of the best row of 02 |
| `TRANSFORMS` | 02b | label -> `extra_transforms` to compare, e.g. `{"none": (), "log": ("log",)}` |
| `CHECK` | 02b | `True`: print `check_transforms` before the CV |
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
scripts, and a JSON file can replace some of them. The phases are the scripts that
match `0[0-9]*_*.py`, in name order, so `02b_transforms` runs between `02_degree` and
`03_interactions`; the `extra_*.py` scripts are not phases.

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

# the missing phases: for logistic_boosting 01 and 04 exist, so everything is skipped
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

`--phases` takes full phase names or prefixes (`02`, `02_degree`); the prefix `02`
also selects `02b_transforms`. Example of
`--dry-run` output, for `python experiments/run_phases.py logistic_boosting --phases 04
--force --dry-run`:

```
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

Example 2, `configs/logistic_boosting_final_400_trees.json`: only phase 04, with
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

Example 5: one pipeline only in phase 01, and phase 04 without the feature
selection (written inline here, not in `configs/`).

```json
{
  "01_baseline": {"PIPELINES": ["v1"]},
  "04_final": {"PIPELINE_OVERRIDES": {"select_features": false}}
}
```

### Results

- Phases 01–03 and 02b: `results/<method>/<phase>.json`, with every row of the grid and the best
  one. Every row stores the pipeline and the hyperparameters it used, so a run with a
  config file stays reproducible from its JSON.
- Phase 04: `results/<method>/04_final.json` and the predictions
  `submissions/<method>.csv` (tracked by git, see [results/README.md](../results/README.md)),
  ready for AIcrowd.

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
(`results/<method>/<phase>.json`) and by `baselines.py` (`01_baseline.json`) do not
change:

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

## Extra scripts: interactions, ensemble, feature screening

Scripts outside the phases: `run_phases.py` does not run them and they do not touch
the phase files. Their results, in tables, are [below](#results-of-the-extra-scripts);
the files, costs and reproducibility are in [results/README.md](../results/README.md).

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `reg_logistic_newton/extra_final_interactions.py` | `run_final` with the best row of `03_interactions` (degree 2, log, top 20 interactions): the Newton final model with the expansions, next to `04_final.json` (v1) | `results/reg_logistic_newton/03_interactions.json` | `results/reg_logistic_newton/extra/04_final_interactions.json`, `submissions/reg_logistic_newton_interactions.csv` |
| `logistic_boosting/extra_interactions.py` | 1. CV grid: the pipeline of `01_baseline` + top 20 interactions (`INTERACTION_TOP = (20,)`), `n_trees` 200 / 400, compared fold by fold with the rows of `01_baseline` with the same fit; 2. `run_final` with the best row of step 1. A step whose file exists is skipped (`--force`) | `results/logistic_boosting/01_baseline.json` | `results/logistic_boosting/extra/03_interactions.json` and `extra/04_final_interactions.json`, `submissions/logistic_boosting_interactions.csv` |
| `ensemble.py` | `p = a * p_newton + (1 - a) * p_boosting` of the two settings of `COMPONENTS`. CV: for every fold, `a` (grid `WEIGHTS`, 0 to 1 by 0.05) and the threshold are chosen on the out-of-fold probabilities of the other folds; each component alone (`a = 1`, `a = 0`) is scored the same way. Final: `a` and threshold on all the out-of-fold probabilities; one submission without and one with the known-positive rule | `results/reg_logistic_newton/extra/04_final_interactions.json`, `results/logistic_boosting/04_final.json`, the cache `results/ensemble/scores_*.npz` | `results/ensemble/ensemble.json`, the cache if missing, `submissions/ensemble_newton_boosting.csv`, `submissions/ensemble_newton_boosting_rule.csv` |
| `feature_screening.py` | groups of hand-made features for Newton (comorbidity, risk factor, functional limitation and health care contact counts, clinical interactions), added as extra raw columns. Level 1: score test against the best row of `03_interactions` (reference: the top 20 interactions against `02b_transforms`); level 2: 5-fold CV of the groups that pass (Bonferroni over the groups) | `results/reg_logistic_newton/03_interactions.json`, `02b_transforms.json` | `results/reg_logistic_newton/extra/feature_screening.json` |

```bash
python experiments/reg_logistic_newton/extra_final_interactions.py
python experiments/logistic_boosting/extra_interactions.py    # --force: both steps again
python experiments/ensemble.py                                 # from the cache, no training
python experiments/ensemble.py --force                         # train the two components again
python experiments/feature_screening.py                        # levels 1 and 2
python experiments/feature_screening.py --level 1              # only the score test
```

Things to know:

- **Order.** `ensemble.py` needs `reg_logistic_newton/extra/04_final_interactions.json`
  (so `extra_final_interactions.py` after phase 03) and `logistic_boosting/04_final.json`.
  `feature_screening.py` needs phases 02b and 03 of `reg_logistic_newton`.
- **The ensemble cache.** `results/ensemble/scores_<method>.npz` hold the out-of-fold
  and test probabilities of each component; with them `ensemble.py` only chooses the
  weight and the threshold and writes the two submissions, without training (in a
  few seconds, see the ensemble results below). The cache is not invalidated by itself: after a change of a
  component's result file, use `--force`.
- **`--level 1`** of `feature_screening.py` rewrites
  `extra/feature_screening.json` with an empty `level2`.
- **Which model is submitted** (model A, the ensemble, with or without the
  known-positive rule): the ensemble, without the rule
  (`submissions/ensemble_newton_boosting.csv`), reproduced by `run.py`.

### Results of the extra scripts

The extra scripts work on the two methods with the highest F1, `reg_logistic_newton`
and `logistic_boosting`. The tables are written by hand from the JSON files named in
each part (no script writes them). Every F1 is a 5-fold CV with the same folds as every
other file (`k_fold = 5`, `seed = 1`): "folds better" compares the row with its
reference fold by fold. The standard deviation between folds is about 0.002, so a
smaller difference is noise unless it holds in every fold. `HAREHAB1` is never a
feature; the column "F1 + rule" adds the known-positive rule after the threshold
(`src/rules.py`), only to measure the leakage.

#### 1. Newton: degree, transforms, interactions

Phases 02 -> 02b -> 03 of `reg_logistic_newton`
(`results/reg_logistic_newton/02_degree.json`, `02b_transforms.json`,
`03_interactions.json`), `max_iters = 20`, lambda in {1e-6, 1e-4, 1e-2} (best for every
variant). Each phase starts from the best pipeline of the previous one (in bold); the
reference of a phase is its first row.

| Phase | Variant | F1 | Δ F1 | folds better | lambda | columns | F1 + rule |
|---|---|---|---|---|---|---|---|
| 02 degree | 1 (pipeline v1) | 0.4323 ± 0.0020 | reference | – | 1e-4 | 334 | 0.4382 |
| | **2** | 0.4323 ± 0.0022 | +0.0000 | 3/5 | 1e-4 | 390 | 0.4384 |
| | 3 | 0.4321 ± 0.0017 | -0.0002 | 1/5 | 1e-6 | 444 | 0.4379 |
| 02b transforms (degree 2) | none | 0.4323 ± 0.0022 | reference | – | 1e-4 | 390 | 0.4384 |
| | sqrt | 0.4322 ± 0.0015 | -0.0001 | 3/5 | 1e-4 | 427 | 0.4382 |
| | **log** | 0.4324 ± 0.0018 | +0.0001 | 2/5 | 1e-4 | 427 | 0.4385 |
| | sqrt + log | 0.4319 ± 0.0015 | -0.0004 | 2/5 | 1e-4 | 464 | 0.4379 |
| 03 interactions (degree 2, log) | top 0 | 0.4324 ± 0.0018 | reference | – | 1e-4 | 427 | 0.4385 |
| | top 10 | 0.4336 ± 0.0012 | +0.0012 | 4/5 | 1e-4 | 472 | 0.4398 |
| | **top 20** | 0.4339 ± 0.0016 | +0.0015 | 5/5 | 1e-6 | 617 | 0.4398 |
| | top 30 | 0.4336 ± 0.0027 | +0.0012 | 3/5 | 1e-4 | 862 | 0.4396 |

"top k": the products of all the pairs of the k columns most correlated with the
label, chosen on the training part of each fold (k (k - 1) / 2 products: 190 for
k = 20). Columns: in fold 0.

- Powers and the sqrt / log transforms change F1 by at most 0.0004: the log-odds of
  most columns are already close to linear after the encoding ([EDA.md](EDA.md),
  analysis A6).
- The interactions add +0.0015 in all 5 folds; 30 columns do not improve on 20.
- lambda = 1e-6 is the smallest value of the grid (row "top 20"; with 1e-4 the F1 is
  also 0.4339).

Final model with these expansions
(`results/reg_logistic_newton/extra/04_final_interactions.json`,
`reg_logistic_newton/extra_final_interactions.py`; `04_final.json` stays the one of
pipeline v1): CV 0.4339 ± 0.0016, out-of-fold threshold 0.196 (out-of-fold F1 0.4344),
619 columns on the whole training set, 14.70% predicted positive on the test set. It is
the first component of the submitted ensemble (part 3).

#### 2. Logistic boosting with interactions

`results/logistic_boosting/extra/03_interactions.json` and
`extra/04_final_interactions.json` (`logistic_boosting/extra_interactions.py`):
pipeline v1 + top 20 interactions, `lambda_ = 1e-4`, `max_iters = 20`, `max_depth = 3`,
`learning_rate = 0.05`. Reference: the rows of `logistic_boosting/01_baseline.json`
with the same fit, without interactions.

| Trees | With interactions | Without (`01_baseline`) | Δ F1 | folds better | columns |
|---|---|---|---|---|---|
| 200 | 0.4330 ± 0.0017 | 0.4335 ± 0.0022 | -0.0006 | 1/5 | 524 |
| 400 | 0.4333 ± 0.0025 | 0.4338 ± 0.0020 | -0.0005 | 1/5 | 524 |

Final model with interactions (`extra/04_final_interactions.json`): CV 0.4333 ±
0.0025, out-of-fold F1 0.4347, 526 columns, 14.36% predicted positive. Not used: the
interactions do not help the boosted trees, which already split on combinations of
columns. Monotone transforms (sqrt, log) were not tried for the trees: a split on
log x divides the rows as a split on x does.

#### 3. Ensemble Newton + logistic boosting

`results/ensemble/ensemble.json` (`ensemble.py`):
`p = a p_newton + (1 - a) p_boosting`, with `p_newton` from part 1 (final model) and
`p_boosting` from `logistic_boosting/04_final.json` (model A). The two components are
the two methods with the highest CV F1, and they model the interactions differently
(fixed products in a linear model, trees on the residuals): if they err on different
people, averaging the probabilities cancels part of the errors (not checked directly).
In each fold, `a` (grid 0, 0.05, ..., 1) and the threshold are chosen on the
out-of-fold probabilities of the other 4 folds, then F1 is measured on the remaining
fold. Each component alone is scored with the same protocol (`a` fixed to 1 or 0), so
the rows are comparable. Reference: logistic boosting alone.

| Model | F1 | Δ F1 | folds better | `a` per fold | F1 + rule |
|---|---|---|---|---|---|
| Newton (interactions) alone | 0.4336 ± 0.0015 | -0.0001 | 2/5 | 1 | 0.4390 ± 0.0018 |
| Logistic boosting alone | 0.4338 ± 0.0021 | reference | – | 0 | 0.4399 ± 0.0026 |
| **Ensemble** | **0.4356 ± 0.0020** | **+0.0018** | **5/5** | 0.4, 0.4, 0.35, 0.35, 0.35 | 0.4414 ± 0.0021 |

- The ensemble is better in all 5 folds, and the weight chosen on the other folds is
  stable (0.35 to 0.40 for Newton).
- Final (`final` in the JSON): `a = 0.35`, threshold 0.2062 (0.20618416941358658),
  out-of-fold F1 0.4360, 13.60% predicted positive:
  `submissions/ensemble_newton_boosting.csv`, the submitted file, reproduced by
  `run.py`. With the rule: `submissions/ensemble_newton_boosting_rule.csv`
  (`HAREHAB1` was asked in 238 test rows; 13.67% predicted positive).
- The out-of-fold and test probabilities of both components are cached in
  `results/ensemble/scores_*.npz` (see [results/README.md](../results/README.md)): with
  them `ensemble.py` writes both files in a few seconds. A regeneration from the cache
  gave a file identical byte for byte to the first one (checked once by the group).

#### 4. Feature screening: hand-made features

`results/reg_logistic_newton/extra/feature_screening.json` (`feature_screening.py`).
Groups of features built from the raw codes (indicator 1 = yes; missing, "don't know",
"refused" and question not asked count as no), added as extra raw columns to the
pipeline of the best row of `03_interactions` (degree 2, log, top 20 interactions).

| Group | Columns |
|---|---|
| comorbidity count (number) | number of conditions among diabetes, hypertension, high cholesterol, stroke, kidney disease, COPD, arthritis |
| comorbidity count (one-hot) | the same count as indicators >= 2, >= 3, >= 4 |
| risk factor count (one-hot) | age at risk (men 55+, women 65+), current smoker, hypertension, high cholesterol, diabetes, obesity (BMI >= 30), inactivity: indicators >= 2 ... >= 5 |
| functional limitation (one-hot) | `DIFFWALK`, `USEEQUIP`, `QLACTLM2`, `DIFFDRES`, `DIFFALON`, unable to work, >= 14 days of poor physical health: indicators >= 1 ... >= 4 |
| health care contact (one-hot) | check-up in the last year, pneumonia vaccine, flu vaccine, daily aspirin, blood pressure medication: indicators >= 1 ... >= 4 |
| clinical interactions | 14 products: age x sex, age x diabetes, age x hypertension, age x cholesterol, age x current / ever smoker, age x stroke, sex x ever smoker, sex x diabetes, diabetes x hypertension, hypertension x cholesterol, obesity x diabetes, diabetes x kidney, age x comorbidity count |

**Level 1, score test** (no refit). The current model is fitted once; at its minimum
the residuals `y - p` are uncorrelated with every column of the model, and the score
(Lagrange multiplier) test measures how much of what is left a new group explains:

    g = X_new^T (p - y) / N                 gradient of the loss in the new weights, at 0
    I = H_nn - H_nb H_bb^-1 H_bn            curvature left after the columns already in
    chi2 = N g^T I^-1 g  ~  chi2(df)        if the group adds nothing

`chi2 / (2N)` estimates the decrease of the mean logistic loss after a refit
("Δ loss"). Reference of scale: the same test for the top 20 interactions against the
model without them (`02b_transforms`), which are worth +0.0015 in F1. A group passes
with p < 0.05 / 6 (Bonferroni over the 6 groups). "In top 20": the group's columns
enter the 20 columns most correlated with the label, so they change the interactions
chosen by `interaction_top`.

| Group | df | chi2 | p | Δ loss | in top 20 | passes |
|---|---|---|---|---|---|---|
| reference: top 20 interactions | 190 | 1152.9 | 7e-129 | 1.8e-3 | – | – |
| comorbidity count (number) | 1 | 0.0 | 0.82 | 6e-8 | yes | no |
| comorbidity count (one-hot) | 3 | 17.7 | 6e-4 | 2.7e-5 | yes | yes |
| risk factor count (one-hot) | 4 | 18.2 | 1e-3 | 2.8e-5 | yes | yes |
| functional limitation (one-hot) | 4 | 21.8 | 3e-4 | 3.3e-5 | yes | yes |
| health care contact (one-hot) | 4 | 6.7 | 0.15 | 1.0e-5 | yes | no |
| clinical interactions | 14 | 54.8 | 1e-6 | 8.4e-5 | no | yes |

- The count as a number gives chi2 = 0: it is a sum of columns the model already has,
  and a logistic regression already uses weighted sums. A new feature can help a
  linear model only if it is non-linear in the existing columns.
- Among the clinical interactions only age x sex adds much (chi2 34.6), then
  age x current smoker (10.1) and age x ever smoker (8.7); 8 of the 14 products have
  chi2 < 0.1.
- The groups that pass are 20 to 65 times smaller in Δ loss than the reference: with
  328,135 rows even tiny effects are significant.

**Level 2, CV** of the groups that pass, on the same folds as the reference row
(`03_interactions`, top 20: 0.4339 ± 0.0016).

| Group | F1 | Δ F1 | folds better | columns |
|---|---|---|---|---|
| comorbidity count (one-hot) | 0.4330 ± 0.0019 | -0.0009 | 2/5 | 620 |
| risk factor count (one-hot) | 0.4335 ± 0.0021 | -0.0004 | 2/5 | 621 |
| functional limitation (one-hot) | 0.4328 ± 0.0017 | -0.0011 | 2/5 | 621 |
| clinical interactions | 0.4327 ± 0.0009 | -0.0012 | 1/5 | 645 |
| all the groups that pass | 0.4326 ± 0.0018 | -0.0013 | 1/5 | 656 |

No group improves F1: a decrease of the loss of order 3e-5 does not move the ranking
of the rows around the threshold. The clinical interactions do not enter the top 20
and do not improve either, so the change of the automatic interactions does not
explain the result. Level 3 (boosting, ensemble) was not run.

#### Summary

| Experiment | Result |
|---|---|
| Newton: powers, sqrt / log | no gain (at most +0.0001) |
| Newton: top 20 interactions | +0.0015, 5/5 folds |
| Logistic boosting: top 20 interactions | -0.0005, 1/5 folds |
| Hand-made clinical features | -0.0004 to -0.0013 |
| Ensemble Newton (interactions) + logistic boosting | +0.0018 over logistic boosting, 5/5 folds: the submitted model |

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

## Ablation study: `ablation.py`

How much every choice contributes to model A (`FINAL_METHOD = "logistic_boosting"`,
phase 04; the scripts call it "final", which is not a decision on the submission):
starting from model A, one thing is removed at a time, on the same folds as all the other experiments. Two
tables:

| Table | What is removed | Model | Hyperparameters |
|---|---|---|---|
| Data preparation | one pipeline step at a time: each rule R1–R4 of the feature selection, the whole selection, the selection replaced by the NaN rule B5 (= full pipeline), domain drop, duplicate drop, unit conversion, skip-logic fill, missing indicators, one-hot, one-hot of the ordinals, binning, imputation by type | Newton (`reg_logistic_newton`) | lambda searched again for every variant (`DATA_LAMBDAS`) |
| Model | the trees on the residuals; then also Newton's exact minimum (plain GD); the tuned threshold (0.5 instead) | model A (`logistic_boosting`) | fixed, those of model A |

Why two models: removing columns can move the best lambda, so it is searched again
for every pipeline variant, which is affordable with Newton (~20 s per fit: 90 s per
5-fold CV for the row "final" of `results/reg_logistic_newton/ablation_data.json`) and
not with the hybrid (~44 min per 5-fold CV, the mean over the grid of
`results/logistic_boosting/01_baseline.json`). The hybrid is Newton plus a small correction by
trees, so the steps of the pipeline weigh the same on both. In the model table the
removed components already had their hyperparameters chosen in CV.

Model A is read from `results/<FINAL_METHOD>/04_final.json`: the pipeline and
the hyperparameters that phase 04 used for the submission, `FIT_OVERRIDES` /
`PIPELINE_OVERRIDES` included. So **run phase 04 first** (`run_phases.py <method>`);
without it the ablation stops and says so. The ablation does not change the model: it
measures what every part of it is worth, for the report. The other settings are the
block at the top of the script.
Every row gives the mean F1, the difference from the final row and in how many
folds the variant is better (paired on the same folds).

```bash
python experiments/ablation.py                # missing parts, then the tables
python experiments/ablation.py --parts data   # only the data-preparation table (~1 h)
python experiments/ablation.py --parts model  # only the model table (~7 min: the final
                                              # row is reused from the CV of phase 04)
python experiments/ablation.py --force        # compute everything again
```

The CV results go to `results/reg_logistic_newton/ablation_data.json`,
`results/logistic_boosting/ablation_model.json`,
`results/reg_logistic_regression/ablation_model.json`; a part whose results exist is
not computed again (use `--force` after changing model A or the settings).
The tables go to `results/ablation.md` and `results/ablation.tex` (LaTeX tabular).

## HAREHAB1, the label leakage column: `leakage.py`

`HAREHAB1` ("after your heart attack, did you go to rehabilitation?") is asked only to
people who had a heart attack, which is part of the definition of MICHD: the 773
training rows where it was asked (answers 1, 2, 7, 9) are all positive. Every model
drops it (`LABEL_LEAKAGE`). `leakage.py` trains two variants of model A (phase 04 of
`METHOD = "logistic_boosting"`), with its setting and folds, without touching it:

| Model | What | Result | Submission |
|---|---|---|---|
| A | phase 04, no HAREHAB1 | `results/<method>/04_final.json` | `submissions/<method>.csv` |
| B (`harehab1`) | HAREHAB1 as a feature (`keep_leakage=True` in the pipeline) | `results/<method>/extra/04_final_harehab1.json` | `submissions/<method>_harehab1.csv` |
| C (`rule`) | model A, then the known-positive rule (`src/rules.py`): rows where HAREHAB1 was asked set to 1 after the threshold | `results/<method>/extra/04_final_rule.json` | `submissions/<method>_rule.csv` |

B learns both sides of the leakage (asked → heart attack; in the states with the
module, not asked → no heart attack, 4.1% MICHD instead of 8.8%); C only applies the
certain side. Every CV now also saves `f1_rule` (F1 after the rule, same model and
threshold), which gives the CV number of C. A variant already computed is skipped
(`--force`). At the end the script prints A, B and C on the same folds; `ablation.py`
adds B and C as rows of its model table. The variants take the best row of
`FROM_PHASE = "01_baseline"` (settings block of `leakage.py`), without the
`PIPELINE_OVERRIDES` / `FIT_OVERRIDES` of `04_final.py`: this is the setting of model A
for `logistic_boosting` and `reg_logistic_newton`; for a method whose phase 04 starts
from another phase, change `FROM_PHASE` first.

```bash
python experiments/leakage.py                                  # B and C of logistic_boosting (~2 h)
python experiments/leakage.py --variants rule                  # only C
python experiments/leakage.py --method reg_logistic_newton     # the same for Newton (minutes)
```

In the report: A is a predictive model; B measures the leakage; C is a declared
correction of certain labels, not part of the model. The ensemble has its own rule
variant (`ensemble.py`, above).

## Notes

- F1 is measured on the validation part of each fold with the threshold chosen on the
  training part (`src/cross_validation.py`); `val-best` in the tables is the optimistic
  F1 with the threshold chosen on the validation part itself. Report `f1_mean`.
  `f1_default` is the F1 with the natural threshold of the method (0.5 for
  probabilities, 0 for the linear methods), without tuning: used by the ablation.
- The polynomial and interaction columns are pipeline step B11 (`src/pipeline.py`):
  they are fitted inside each fold, and the interaction columns chosen by
  `interaction_top` use the labels of the training part only.
- The models see 321 columns with v0, 415 with the full pipeline and 334 with v1 (bias
  included; `n_features` in `results/*/01_baseline.json`). Degree p adds the power p of
  every standardized column with more than p distinct values; `interaction_top = k`
  adds k (k - 1) / 2 columns. For `reg_logistic_newton` on v1 (fold 0): 390 columns
  with degree 2, 444 with degree 3 (`02_degree.json`), 427 with degree 2 + log
  (`02b_transforms.json`), 617 with degree 2 + log + top 20 (`03_interactions.json`). With 260k training rows per fold, ~600 columns already take ~1.3 GB per
  matrix.
