# ML Project 1 — MICHD / Heart Attack Risk Prediction

CS-433 Machine Learning, Fall 2026 — Class Project 1 (EPFL).
Binary classification of coronary heart disease (MICHD: heart attack or coronary heart
disease) from the answers of the BRFSS 2015 health survey: 328,135 training rows,
109,379 test rows, 321 columns, 8.8% positives; the metric is F1.
The submitted model is an **ensemble**: the weighted mean (weight 0.35 / 0.65) of the
probabilities of a Newton logistic regression with expanded features and of a hybrid
model (Newton logistic regression + boosted trees on its residuals), without the
label-leakage column `HAREHAB1`.


Contents:

1. [Results](#results)
2. [Reproducing the submission](#reproducing-the-submission)
3. [Reproducibility details](#reproducibility-details)
4. [Data preparation, features and cross-validation](#data-preparation-features-and-cross-validation)
5. [Repository structure](#repository-structure)
6. [Running the experiments](#running-the-experiments)
7. [Tests](#tests)
8. [Limitations](#limitations)

## Results

F1 in 5-fold stratified cross-validation (seed 1, the same folds for every model) and
F1 on AIcrowd. The AIcrowd scores are those reported by the group from the competition
platform; they are not stored in the repository.

| Submission | File | CV F1 | AIcrowd F1 |
|---|---|---|---|
| **Ensemble (submitted)**: 0.35 x Newton (degree 2, log, top 20 interactions) + 0.65 x model A, no `HAREHAB1` | `submissions/ensemble_newton_boosting.csv` | 0.4356 ± 0.0020 | 0.446 |
| Model A: `logistic_boosting` (Newton logistic regression + boosted trees), no `HAREHAB1` | `submissions/logistic_boosting.csv` | 0.4338 ± 0.0020 | 0.445 |
| Ensemble + known-positive rule (`HAREHAB1` asked -> 1, after the threshold) | `submissions/ensemble_newton_boosting_rule.csv` | 0.4414 ± 0.0021 | 0.449 |
| Model B: model A with `HAREHAB1` as a feature | `submissions/logistic_boosting_harehab1.csv` | 0.4413 ± 0.0032 | 0.452 |
| All the methods against trivial rules; the ablation study | [results/baselines.md](results/baselines.md), [results/ablation.md](results/ablation.md) | – | – |

Sources of the CV numbers: `results/ensemble/ensemble.json` (ensemble rows: the weight
and the threshold of each fold are chosen on the other folds) and
`results/logistic_boosting/04_final.json`, `results/logistic_boosting/extra/04_final_harehab1.json`
(threshold chosen on the training part of each fold).

**Why the best online score (0.452) is not the submitted model.** `HAREHAB1` ("after
your heart attack, did you go to rehabilitation?") is asked only to people who had a
heart attack, which is part of the definition of MICHD: the 773 training rows where it
was asked are all positive. The column reveals the label instead of predicting it, so
model B and the known-positive rule are a measure of this leakage, not a predictive
model, and the submitted model never uses the column. Among the models without it, the
ensemble has the highest CV F1 (+0.0018 over model A, better in 5 of 5 folds) and the
higher AIcrowd score (0.446 against 0.445). The project description asks to mention the
test performance of the submission when it is worse than the best one on the
competition system: the report gives both. Details:
[experiments/EDA.md](experiments/EDA.md) (section 6) and
[experiments/README.md](experiments/README.md) (`leakage.py`).

## Reproducing the submission

```bash
pip install -r requirements.txt
python run.py
```

`run.py` reproduces `submissions/ensemble_newton_boosting.csv`, the submitted
ensemble. It trains both components from scratch on the whole training set, with the
settings of `results/reg_logistic_newton/extra/04_final_interactions.json` and
`results/logistic_boosting/04_final.json` written as constants (no cross-validation,
no cached probabilities), averages their test probabilities with the weight a = 0.35
and applies the threshold of `results/ensemble/ensemble.json` (`final`). `HAREHAB1` is
not used and the known-positive rule is not applied. It writes
`submissions/submission.csv` (gitignored) and prints, at every step, the elapsed time and the values
expected from the JSON files (619 and 336 columns, 13.605% predicted positive). Its
running time, measured once on the machine of one of us, is 689 s (about 11.5 minutes):
23 s for the data and the preprocessing of the first component, 18 s to fit it, and about
645 s for the second component (preprocessing and 400 trees on 336 columns). To check the output, compare it with the reference file: its SHA-256, as written
by `helpers.create_csv_submission` (CRLF line ends), is

```
0cf54c57231a8dfda95aafd6c46e0291bdde67e777e533dc988a16a4cacc1c91  submissions/ensemble_newton_boosting.csv
```

(`Get-FileHash <file> -Algorithm SHA256` in PowerShell, `sha256sum <file>` on
Linux / macOS). A clone with LF line ends (git `core.autocrlf=false`) has the same
content with another hash, `fc4043f04ec69ad04a47620e247542a466793422347df0019cfe7551d2787ef1`.
To compare the predictions regardless of the line ends:
`git diff --no-index --ignore-cr-at-eol <output file> submissions/ensemble_newton_boosting.csv`
(no output = the same predictions).

The data are read from `dataset/x_train.csv`, `y_train.csv`, `x_test.csv` (not in git:
download them from AIcrowd) and cached in `build/data.npz` on the first run
(`src/data.py`).

The other files:

| File | Command | Notes |
|---|---|---|
| both ensemble files | `python experiments/ensemble.py` | Uses the cached out-of-fold and test probabilities of the two components (`results/ensemble/scores_*.npz`): no training. `--force` trains the components again. Also rewrites `results/ensemble/ensemble.json` |
| model A | `python experiments/run_model_a.py` | Fits model A, the second component of the ensemble, alone, with the constants of `results/logistic_boosting/04_final.json` and writes `submissions/submission_model_a.csv`. `python experiments/logistic_boosting/04_final.py` also runs the CV and writes `submissions/logistic_boosting.csv` and `04_final.json` |
| model B and the rule on model A | `python experiments/leakage.py --force` | Writes `submissions/logistic_boosting_harehab1.csv` and `logistic_boosting_rule.csv`; without `--force` the variants whose results exist are skipped |

The training code of each component: `experiments/reg_logistic_newton/` (phases 01-03,
then `extra_final_interactions.py`) and `experiments/logistic_boosting/` (phases 01 and
04). Running times of the experiments, where measured: [results/README.md](results/README.md).

## Reproducibility details

- **Folds and seed.** 5 stratified folds, seed 1 (`stratified_k_indices` in
  `src/cross_validation.py`), the same in every experiment. The seed also fixes the
  rows, columns and bins of the trees.
- **Preprocessing.** Fitted on the training part of each fold only, then applied to
  the validation part (and, for the submission, fitted on the whole training set and
  applied to the test set).
- **Decision threshold.** In the CV, the threshold that maximizes F1 on the training
  scores of the fold, F1 measured on its validation part. For a submission
  (`run_final` in `experiments/common.py`), the threshold that maximizes F1 on the
  out-of-fold scores of the CV. For the ensemble, the weight `a` (grid 0, 0.05, ..., 1)
  and the threshold maximize F1 on the out-of-fold probabilities of the two components.
- **The submitted ensemble** (`results/ensemble/ensemble.json`): `a = 0.35` (Newton),
  threshold `0.20618416941358658`, out-of-fold F1 0.4360.
  - Newton component (`results/reg_logistic_newton/extra/04_final_interactions.json`):
    `reg_logistic_newton`, `lambda_=1e-6`, `max_iters=20` (Newton steps, tolerance
    `tol=1e-10`, the default); pipeline `V1_CONFIG` with `poly_degree=2`,
    `extra_transforms=("log",)`, `interaction_top=20`: 619 columns on the whole
    training set.
  - Model A component (`results/logistic_boosting/04_final.json`):
    `logistic_boosting`, `lambda_=1e-4`, `max_iters=20`, `n_trees=400`, `max_depth=3`,
    `learning_rate=0.05`, and the defaults of
    `src/extra_implementations/logistic_boosting.py`, not saved in the JSON:
    `min_samples_leaf=100`, `reg_lambda=1.0`, `subsample=0.5`, `colsample=0.5`,
    `n_bins=32`; pipeline `V1_CONFIG`: 336 columns on the whole training set.
- **Libraries.** Python 3.9 with the versions of the graders' environment, pinned in
  [requirements.txt](requirements.txt): `numpy==1.23.1`, `matplotlib==3.5.2`,
  `seaborn==0.11.2` (plots only), and for development `pytest==7.1.2`, `black==22.6.0`,
  `pytest-mock==3.7.0`, `gitpython==3.1.18`. The conda environment of the course
  (`grading_tests/environment.yml`) has the same versions.
- **Rules of the project.** Only NumPy, the standard library and matplotlib / seaborn
  for the plots; the 6 methods of [implementations.py](implementations.py) keep the
  signatures required by the grader.

## Data preparation, features and cross-validation

The steps are described once, in [src/README.md](src/README.md) (modules and the
`fit_preprocess` / `apply_preprocess` flow), in the docstring of
[src/pipeline.py](src/pipeline.py) (steps B1-B13 and their config keys) and in
[experiments/EDA.md](experiments/EDA.md) (the analyses A1-A9 and the column lists they
led to). In short:

- **Cleaning** (no fitted statistics): BRFSS special codes to NaN or 0 column by column,
  one unit per column, skip-logic answers filled, `PHYSHLTH` binned.
- **Columns:** fixed lists from the codebook and the EDA (`src/feature_types.py`):
  identifiers and survey design dropped, duplicates dropped, the feature selection of
  pipeline v1 (124 of the 321 columns kept), `HAREHAB1` always dropped.
- **Fitted on the training rows:** missing indicators for the columns whose missingness
  changes the MICHD rate, imputation by type, binary recoding, one-hot of the nominal
  (and some ordinal) columns, standardization, bias.
- **Feature generation** (step B11, used by the Newton component): squares,
  `log(1 + x - min)` of the continuous and count columns, and the products of the 20
  columns most correlated with the label (chosen on the training part of each fold).
  Hand-made clinical features were tested and did not improve F1
  (`experiments/feature_screening.py`, `results/reg_logistic_newton/extra/feature_screening.json`;
  tables in [experiments/README.md](experiments/README.md#results-of-the-extra-scripts)).
- **Cross-validation:** `src/cross_validation.py` (`cross_validate_grid`), see the
  reproducibility details above; the experiment phases are in
  [experiments/README.md](experiments/README.md).

## Repository structure

```
README.md                 # this file
implementations.py        # the 6 required methods (step 2), signatures fixed by the grader
helpers.py                # course-provided load_csv_data / create_csv_submission
run.py                    # reproduces the submitted csv (the ensemble)
requirements.txt          # pinned versions of the graders' environment
src/                      # reusable code; README.md describes every module
  data.py                 # load_data: dataset/*.csv -> cache build/data.npz
  feature_types.py        # column lists from the codebook and the EDA
  preprocessing.py        # cleaning steps, imputation, standardization
  features.py             # one-hot, polynomial and interaction helpers
  pipeline.py             # fit_preprocess / apply_preprocess (steps B1-B13), V0/V1 configs
  rules.py                # known-positive rule (HAREHAB1 asked -> 1), not part of the model
  models.py               # fit / predict interface of all the methods, labels in {-1, 1}
  metrics.py              # F1, best threshold, ...
  cross_validation.py     # stratified folds, CV with the preprocessing inside each fold
  plots.py                # plotting helpers of the EDA
  extra_implementations/  # Newton, adaptive GD, Sophia, UniXGrad, random forest,
                          # gradient boosting, logistic_boosting (NumPy only)
experiments/              # scripts that produce results/; README.md explains them
  EDA.md                  # the exploratory analysis A1-A9 and the column choices
  eda.py                  # the EDA (tables in build/eda/, figures in report/figures/)
  common.py               # run_grid / run_final, shared by the phase scripts
  run_phases.py           # runs the phases of a method in order, skipping those done
  configs/                # example JSON settings for run_phases.py --config
  <method>/               # 13 folders: README.md, phases 01_baseline ... 04_final,
                          # extra_*.py scripts (not phases)
  baselines.py            # all the methods against trivial rules -> results/baselines.*
  ablation.py             # ablation study -> results/ablation.*
  leakage.py              # HAREHAB1: as a feature (model B) or as a rule after model A
  ensemble.py             # the submitted ensemble -> results/ensemble/, submissions/
  feature_screening.py    # hand-made clinical features: score test, then CV
  imbalance.py            # class imbalance: threshold vs class weights vs undersampling
  smoke_test.py           # ~30 s check on real data before the long runs
  run_model_a.py          # model A alone -> submissions/submission_model_a.csv
results/                  # JSON of every phase and extra run, tables; README.md lists
                          # every file, the script that writes it and its cost
  ensemble/               # ensemble.json and the cache scores_*.npz
tests/                    # our tests; README.md describes each file
report/                   # LaTeX report and figures; README.md explains how to compile
submissions/              # the csv uploaded to AIcrowd (tracked in git; run.py's output is not)
dataset/                  # x_train.csv, y_train.csv, x_test.csv (gitignored)
build/                    # data cache and EDA outputs (gitignored)
```

## Running the experiments

Every command runs from the repository root. Details, settings and the JSON config
files: [experiments/README.md](experiments/README.md); the result files:
[results/README.md](results/README.md).

Each method has its phases in `experiments/<method>/` (01 baseline, 02 degree, 02b
transforms for `ridge_regression` and `reg_logistic_newton`, 03 interactions, 04 final
model; only 01 and 04 for most of the extra methods), every phase starting from the best
result of the previous one. `run_phases.py` runs them in order, skipping the phases
whose result is already in `results/`:

```bash
python experiments/run_phases.py <method> --dry-run   # plan only
python experiments/run_phases.py <method>             # missing phases
python experiments/run_phases.py <method> --force     # all the phases again
```

Baseline table of all the methods (5-fold CV, same folds) against trivial rules,
written to `results/baselines.md` and `results/baselines.tex`:

```bash
python experiments/baselines.py            # reads results/*/01_baseline.json
python experiments/baselines.py --rerun    # first runs phase 01 of every method again (hours)
python experiments/baselines.py --rerun ridge_regression least_squares   # only some methods
```

Ablation study (one pipeline step or model component removed at a time), written to
`results/ablation.md` and `results/ablation.tex`; the HAREHAB1 variants:

```bash
python experiments/ablation.py                # missing parts, then the tables
python experiments/ablation.py --parts data   # only the data-preparation table
python experiments/leakage.py                 # model B and the rule on model A
```

Extra scripts (not phases, they never overwrite `04_final.json`):

```bash
python experiments/reg_logistic_newton/extra_final_interactions.py
python experiments/logistic_boosting/extra_interactions.py
python experiments/ensemble.py
python experiments/feature_screening.py
```

## Tests

```bash
pytest tests/        # our 142 tests, small fake data, no dataset needed
black --check .      # formatting (black 22.6.0)
```

Each test file is described in [tests/README.md](tests/README.md).

Official tests of the course: clone the course repository
(https://github.com/epfml/ML_course) and run, from its folder
`projects/project1/grading_tests`, in the conda environment of its `environment.yml`:

```bash
pytest --github_link <GITHUB-REPO-URL> .
```

## Limitations

- **Column lists chosen with the labels.** The feature selection of pipeline v1 and the
  other EDA lists were chosen looking at MICHD on the whole training set, then used as
  fixed lists in every fold: the CV score of v1 is slightly optimistic. The ablation
  shows no measurable gain from the selection (0.4321 without it, 0.4323 with it); the
  size of the optimism is not measured ([experiments/EDA.md](experiments/EDA.md),
  section 5).
- **Data ablation with Newton.** The data-preparation ablation uses Newton's logistic
  regression (lambda searched again for each variant), not the submitted ensemble or
  model A: the hybrid is too slow to tune again for every variant.
- **Lambda at the edge of the grid.** The Newton component uses `lambda_=1e-6`, the
  smallest value of the grid {1e-6, 1e-4, 1e-2} of `03_interactions`; smaller values
  were not tried.
- **`HAREHAB1`.** Excluded from the submitted model (label leakage, see Results); with
  it the online score is higher (0.452), and the report gives both numbers.
- **CV and AIcrowd.** The CV and AIcrowd scores are different estimates (another
  sample, a threshold that transfers differently): compare CV with CV.
