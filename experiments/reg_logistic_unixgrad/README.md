# reg_logistic_unixgrad

The objective of `reg_logistic_regression`, minimized by UniXGrad, an adaptive accelerated extra-gradient method on the ball of radius `radius` (`src/extra_implementations/unixgrad.py`); `gamma` is not used.

- **Phases**: `01_baseline.py` (pipelines full and v1) and `04_final.py` (`FROM_PHASE = "01_baseline"`); no 02 / 03: the objective is the one of `reg_logistic_regression`.
- **Grid** (`FIT_GRID` in the settings block of the scripts): `lambda_`, `max_iters`, `radius`.
- **Results** in `results/reg_logistic_unixgrad/`: `01_baseline.json` (only v1 rows: see [results/README.md](../../results/README.md)) (phase 04 not run yet). Phase 04 also writes `submissions/reg_logistic_unixgrad.csv` (not gitignored: commit it if it goes to AIcrowd).
- **Run**: `python experiments/run_phases.py reg_logistic_unixgrad` (the missing phases; `--dry-run` shows the plan), or one phase with `python experiments/reg_logistic_unixgrad/01_baseline.py`.

Settings, config files and the other scripts: [experiments/README.md](../README.md).
