# reg_logistic_newton

The objective of `reg_logistic_regression`, minimized by damped Newton / IRLS (`src/extra_implementations/newton.py`): the exact minimum, no step size. It is also the model of the data-preparation ablation (`experiments/ablation.py`).

- **Phases** (run by `run_phases.py`): `01_baseline.py` (pipelines full and v1), `02_degree.py` (`poly_degree` 1-3), `02b_transforms.py` (`extra_transforms` none / sqrt / log / sqrt+log, after the printed `check_transforms`), `03_interactions.py` (`interaction_top` 0-30, from the best of 02b), `04_final.py`. `04_final.py` keeps `FROM_PHASE = "01_baseline"`: `04_final.json` is the model on v1 and does not use phases 02-03.
- **Extra script** (not a phase): `extra_final_interactions.py`, the final model with the best row of `03_interactions` (degree 2, log, top 20 interactions) -> `extra/04_final_interactions.json` and `submissions/reg_logistic_newton_interactions.csv`. It is one of the two components of `experiments/ensemble.py`; `experiments/feature_screening.py` also starts from `03_interactions` and writes `extra/feature_screening.json`.
- **Grid** (`FIT_GRID` in the settings block of the scripts): `lambda_`, `max_iters`.
- **Results** in `results/reg_logistic_newton/`: `01_baseline.json` (only v1 rows: see [results/README.md](../../results/README.md)), `02_degree.json`, `02b_transforms.json`, `03_interactions.json`, `04_final.json`, `ablation_data.json` (ablation), `extra/01_lambda.json` (finer lambda grid, no script), `extra/04_final_interactions.json`, `extra/feature_screening.json`. Phase 04 also writes `submissions/reg_logistic_newton.csv`.
- **Run**: `python experiments/run_phases.py reg_logistic_newton` (the missing phases; `--dry-run` shows the plan), or one script, e.g. `python experiments/reg_logistic_newton/extra_final_interactions.py`.

Settings, config files and the other scripts: [experiments/README.md](../README.md).
