# reg_logistic_sophia

The objective of `reg_logistic_regression`, minimized by Sophia: clipped diagonal Newton steps on random minibatches (`src/extra_implementations/sophia.py`).

- **Phases**: `01_baseline.py` (pipelines full and v1) and `04_final.py` (`FROM_PHASE = "01_baseline"`); no 02 / 03: the objective is the one of `reg_logistic_regression`.
- **Grid** (`FIT_GRID` in the settings block of the scripts): `lambda_`, `gamma` (largest move of a coordinate per step), `max_iters`, `batch_size`.
- **Results** in `results/reg_logistic_sophia/`: `01_baseline.json` (only v1 rows: see [results/README.md](../../results/README.md)) (phase 04 not run yet). Phase 04 also writes `submissions/reg_logistic_sophia.csv` (not gitignored: commit it if it goes to AIcrowd).
- **Run**: `python experiments/run_phases.py reg_logistic_sophia` (the missing phases; `--dry-run` shows the plan), or one phase with `python experiments/reg_logistic_sophia/01_baseline.py`.

Settings, config files and the other scripts: [experiments/README.md](../README.md).
