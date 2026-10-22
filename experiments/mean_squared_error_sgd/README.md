# mean_squared_error_sgd

The same MSE regression by stochastic gradient descent, one sample per step (`implementations.mean_squared_error_sgd`): `max_iters` is the number of samples seen, `SEED` fixes their order.

- **Phases**: `01_baseline.py` (pipelines v0, full, v1), `02_degree.py` (`poly_degree`), `03_interactions.py` (`interaction_top`, `interaction_pairs`), `04_final.py` (`FROM_PHASE = "03_interactions"`); 02 and 03 start from the pipeline of the best row of the previous phase, 04 from its pipeline and hyperparameters.
- **Grid** (`FIT_GRID` in the settings block of the scripts): `gamma`, `max_iters`.
- **Results** in `results/mean_squared_error_sgd/`: `01_baseline.json` (phases 02-04 not run yet). Phase 04 also writes `submissions/mean_squared_error_sgd.csv` (not gitignored: commit it if it goes to AIcrowd).
- **Run**: `python experiments/run_phases.py mean_squared_error_sgd` (the missing phases; `--dry-run` shows the plan), or one phase with `python experiments/mean_squared_error_sgd/01_baseline.py`.

Settings, config files and the other scripts: [experiments/README.md](../README.md).
