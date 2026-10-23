# ridge_regression

Least squares with the penalty lambda ||w||^2 (`implementations.ridge_regression`). In the CV the whole lambda grid costs one SVD per fold (`ridge_weights_svd` in `src/cross_validation.py`).

- **Phases** (run by `run_phases.py`): `01_baseline.py` (pipelines v0, full, v1), `02_degree.py` (`poly_degree`), `02b_transforms.py` (`extra_transforms` none / sqrt / log / sqrt+log, after the printed `check_transforms`), `03_interactions.py` (`interaction_top`, `interaction_pairs`), `04_final.py` (`FROM_PHASE = "03_interactions"`). `03_interactions.py` has `FROM_PHASE = "02_degree"`, so it does not use the result of 02b.
- **Grid** (`FIT_GRID` in the settings block of the scripts): `lambda_`.
- **Results** in `results/ridge_regression/`: `01_baseline.json` (phases 02-04 not run yet). Phase 04 also writes `submissions/ridge_regression.csv` (not gitignored: commit it if it goes to AIcrowd).
- **Run**: `python experiments/run_phases.py ridge_regression` (the missing phases; `--dry-run` shows the plan), or one phase with `python experiments/ridge_regression/01_baseline.py`.

Settings, config files and the other scripts: [experiments/README.md](../README.md).
