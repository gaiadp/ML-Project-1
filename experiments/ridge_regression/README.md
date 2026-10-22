# ridge_regression

Least squares with the penalty lambda ||w||^2 (`implementations.ridge_regression`). In the CV the whole lambda grid costs one SVD per fold (`ridge_weights_svd` in `src/cross_validation.py`).

- **Phases**: `01_baseline.py` (pipelines v0, full, v1), `02_degree.py` (`poly_degree`), `03_interactions.py` (`interaction_top`, `interaction_pairs`), `04_final.py` (`FROM_PHASE = "03_interactions"`); 02 and 03 start from the pipeline of the best row of the previous phase, 04 from its pipeline and hyperparameters.
- **Grid** (`FIT_GRID` in the settings block of the scripts): `lambda_`.
- **Results** in `results/ridge_regression/`: `01_baseline.json` (phases 02-04 not run yet). Phase 04 also writes `submissions/ridge_regression.csv` (not gitignored: commit it if it goes to AIcrowd).
- **Run**: `python experiments/run_phases.py ridge_regression` (the missing phases; `--dry-run` shows the plan), or one phase with `python experiments/ridge_regression/01_baseline.py`.

Settings, config files and the other scripts: [experiments/README.md](../README.md).
