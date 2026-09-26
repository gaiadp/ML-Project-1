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

## Setup

1. Create an AIcrowd account with your epfl.ch email and download the dataset from the
   competition page.
2. Place `x_train.csv`, `y_train.csv`, `x_test.csv` in [data/](data/) (untracked by git, see
   [.gitignore](.gitignore)).
3. `pip install -r requirements.txt`

## Repository structure

```
implementations.py   # The 6 required methods (Step 2) — do not rename/move
helpers.py            # Course-provided load_csv_data / create_csv_submission
run.py                # Reproduces our best submission end-to-end -> submissions/
src/                  # Our own reusable pipeline code
  preprocessing.py     # Cleaning, missing values, standardization
  features.py           # Feature engineering (encoding, polynomial expansion, ...)
  models.py             # Our extensions of the 6 base methods
  cross_validation.py   # Train/val split, k-fold CV, grid search
  metrics.py             # Accuracy, F1, confusion matrix, ...
  plots.py                # Shared plotting helpers
experiments/          # Scripts producing the results discussed in the report
  eda.py                # Exploratory data analysis
  baselines.py           # CV comparison of the 6 base methods
  tuning.py               # Hyperparameter search
  ablation.py             # Ablation study
notebooks/            # Scratch/exploration notebooks (not graded, keep experiments/ authoritative)
tests/                 # Our sanity tests (official grading_tests run separately, see below)
report/                # LaTeX report (2 pages + 1 page references, from the course template)
  figures/               # Figures exported from experiments/, referenced by report.tex
data/                  # x_train.csv, y_train.csv, x_test.csv (gitignored)
submissions/           # Generated prediction .csv files (gitignored)
```

## Team & task split

| Person | Area | Files owned |
|---|---|---|
| TBD | Data & EDA | `src/preprocessing.py`, `experiments/eda.py` |
| TBD | Modeling & tuning | `src/models.py`, `src/cross_validation.py`, `experiments/tuning.py`, `experiments/ablation.py` |
| TBD | Methods, pipeline & report | `implementations.py`, `src/features.py`, `run.py`, `report/` |

Everyone reviews everyone's PRs. `implementations.py` is graded verbatim — changes to it
need a second pair of eyes before merging.

## Suggested workflow

1. Work on a branch per feature/person, open a PR into `main`, get one review before merging.
2. Keep `main` always able to run `python run.py` end-to-end.
3. Run the official grading tests before each submission:
   https://github.com/epfml/ML_course/tree/main/projects/project1/grading_tests
4. Log experiment results (method, hyperparameters, CV score) somewhere shared (e.g. a table
   in `experiments/`) so the ablation study and report are easy to write up.
