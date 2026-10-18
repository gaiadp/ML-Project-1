# ML Project 1 — MICHD / Heart Attack Risk Prediction

CS-433 Machine Learning, Fall 2026 — Class Project 1 (EPFL).
Binary classification of coronary heart disease (MICHD) risk from BRFSS lifestyle/clinical survey data.
Competition: https://www.aicrowd.com/challenges/epfl-machine-learning-project-1

**Deadline: Thursday Oct 29th, 2026, 16:00.**

## Rules recap

- Only `numpy`, the Python standard library, and matplotlib/seaborn (visualization only).
  No pandas, scikit-learn, PyTorch, TensorFlow, etc.
- The 6 methods in [implementations.py](implementations.py) must keep their exact
  signatures — they are imported directly by the auto-grader.
- `run.py` must reproduce the `.csv` of our best AIcrowd submission.

## Environment

Python 3.9 with the versions of the graders' environment, pinned in
[requirements.txt](requirements.txt): `pip install -r requirements.txt`, or the conda
environment of the course (`grading_tests/environment.yml`, which has the same versions).

## Repository structure

```
implementations.py   # The 6 required methods (Step 2) — do not rename/move
helpers.py            # Course-provided load_csv_data / create_csv_submission
run.py                # Reproduces our best submission end-to-end -> submissions/
src/                  # Our own reusable pipeline code
  preprocessing.py     # Cleaning, missing values, standardization
  features.py           # Feature engineering (encoding, polynomial expansion, ...)
  models.py             # Common fit / predict interface: 6 base methods + extra methods
  extra_implementations/ # Extra methods, NumPy only: adaptive GD, Newton, Sophia,
                         # UniXGrad (same loss as reg_logistic_regression),
                         # random forest and gradient boosting (histogram trees),
                         # logistic regression + boosted trees on its residuals
  cross_validation.py   # Train/val split, k-fold CV, grid search
  metrics.py             # Accuracy, F1, confusion matrix, ...
  plots.py                # Shared plotting helpers
experiments/          # Scripts producing the results discussed in the report
  eda.py                # Exploratory data analysis
  common.py, <method>/  # Per-method phases: 01 baseline (pipelines v0 / full / v1,
                         # hyperparameter grid), 02 degree, 03 interactions, 04 final
  run_phases.py          # Runs the phases of a method in order, skipping those done
  baselines.py           # Table of all the methods vs trivial rules, from results/
  ablation.py             # Ablation study
results/               # JSON of every phase (results/<method>/<phase>.json) and tables;
                         # one-off runs outside the phases in results/<method>/extra/
USAGE_NOTES.md         # Pitfalls of the data and of the code (Italian)
tests/                 # Our sanity tests (official grading_tests run separately, see below)
report/                # LaTeX report (2 pages + 1 page references, from the course template)
  figures/               # Figures exported from experiments/, referenced by report.tex
dataset/               # x_train.csv, y_train.csv, x_test.csv (gitignored)
submissions/           # Generated prediction .csv files (gitignored)
```

## Running the experiments

Each method has its phases in `experiments/<method>/` (01 baseline, 02 degree,
03 interactions, 04 final model and `submissions/<method>.csv`), every phase starting
from the best result of the previous one. To run them in order, skipping the phases
whose result is already in `results/` (details and the JSON config file in
[experiments/README.md](experiments/README.md)):

```bash
python experiments/run_phases.py <method> --dry-run   # plan only
python experiments/run_phases.py <method>             # missing phases
python experiments/run_phases.py <method> --force     # all the phases again
```

## Baseline comparison

The hyperparameters are tuned inside the phases (grids of `experiments/<method>/0*.py`,
decision threshold chosen in every CV fold). The comparison table of all the methods
(5-fold CV, same folds) against trivial rules, written to `results/baselines.md` and
`results/baselines.tex`:

```bash
python experiments/baselines.py            # reads results/*/01_baseline.json (seconds)
python experiments/baselines.py --rerun    # first runs phase 01 of every method again (hours)
python experiments/baselines.py --rerun ridge_regression least_squares   # only some methods
```
