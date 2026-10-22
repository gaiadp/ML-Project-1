# reg_logistic_regression

Logistic regression with the penalty lambda ||w||^2, by gradient descent (`implementations.reg_logistic_regression`). Its best phase-01 setting is also the "plain GD" row of the model ablation (`experiments/ablation.py`).

- **Phases**: `01_baseline.py` (pipelines v0, full, v1), `02_degree.py` (`poly_degree`), `03_interactions.py` (`interaction_top`, `interaction_pairs`), `04_final.py` (`FROM_PHASE = "03_interactions"`); 02 and 03 start from the pipeline of the best row of the previous phase, 04 from its pipeline and hyperparameters.
- **Grid** (`FIT_GRID` in the settings block of the scripts): `lambda_`, `gamma`, `max_iters` (options `pos_weight`, `neg_per_pos`).
- **Results** in `results/reg_logistic_regression/`: `01_baseline.json` (phases 02-04 not run yet), `ablation_model.json` (ablation). Phase 04 also writes `submissions/reg_logistic_regression.csv` (not gitignored: commit it if it goes to AIcrowd).
- **Run**: `python experiments/run_phases.py reg_logistic_regression` (the missing phases; `--dry-run` shows the plan), or one phase with `python experiments/reg_logistic_regression/01_baseline.py`.

Settings, config files and the other scripts: [experiments/README.md](../README.md).
