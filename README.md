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
dataset/               # x_train.csv, y_train.csv, x_test.csv (gitignored)
submissions/           # Generated prediction .csv files (gitignored)
```