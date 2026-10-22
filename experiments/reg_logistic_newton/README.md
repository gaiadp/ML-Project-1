# reg_logistic_newton

The objective of `reg_logistic_regression`, minimized by damped Newton / IRLS (`src/extra_implementations/newton.py`): the exact minimum, no step size. It is also the model of the data-preparation ablation (`experiments/ablation.py`).

- **Phases**: `01_baseline.py` (pipelines full and v1) and `04_final.py` (`FROM_PHASE = "01_baseline"`); no 02 / 03: the objective is the one of `reg_logistic_regression`.
- **Grid** (`FIT_GRID` in the settings block of the scripts): `lambda_`, `max_iters`.
- **Results** in `results/reg_logistic_newton/`: `01_baseline.json` (only v1 rows: see [results/README.md](../../results/README.md)), `04_final.json`, `ablation_data.json` (ablation), `extra/01_lambda.json` (finer lambda grid, no script). Phase 04 also writes `submissions/reg_logistic_newton.csv` (not gitignored: commit it if it goes to AIcrowd).
- **Run**: `python experiments/run_phases.py reg_logistic_newton` (the missing phases; `--dry-run` shows the plan), or one phase with `python experiments/reg_logistic_newton/01_baseline.py`.

Settings, config files and the other scripts: [experiments/README.md](../README.md).
