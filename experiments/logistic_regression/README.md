# logistic_regression

Logistic regression by gradient descent (`implementations.logistic_regression`); `src/models.py` converts the labels to {0, 1}, the scores are probabilities.

- **Phases**: `01_baseline.py` (pipelines v0, full, v1), `02_degree.py` (`poly_degree`), `03_interactions.py` (`interaction_top`, `interaction_pairs`), `04_final.py` (`FROM_PHASE = "03_interactions"`); 02 and 03 start from the pipeline of the best row of the previous phase, 04 from its pipeline and hyperparameters.
- **Grid** (`FIT_GRID` in the settings block of the scripts): `gamma`, `max_iters` (options `pos_weight`, `neg_per_pos` for the class imbalance).
- **Results** in `results/logistic_regression/`: `01_baseline.json` (phases 02-04 not run yet). Phase 04 also writes `submissions/logistic_regression.csv` (not gitignored: commit it if it goes to AIcrowd).
- **Run**: `python experiments/run_phases.py logistic_regression` (the missing phases; `--dry-run` shows the plan), or one phase with `python experiments/logistic_regression/01_baseline.py`.

Settings, config files and the other scripts: [experiments/README.md](../README.md).
