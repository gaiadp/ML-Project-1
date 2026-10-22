# least_squares

Least squares by the normal equations (`implementations.least_squares`: minimum-norm solution through the eigendecomposition of X^T X). No hyperparameters: only the pipeline and the threshold change.

- **Phases**: `01_baseline.py` (pipelines v0, full, v1), `02_degree.py` (`poly_degree`), `03_interactions.py` (`interaction_top`, `interaction_pairs`), `04_final.py` (`FROM_PHASE = "03_interactions"`); 02 and 03 start from the pipeline of the best row of the previous phase, 04 from its pipeline and hyperparameters.
- **Results** in `results/least_squares/`: `01_baseline.json` (phases 02-04 not run yet). Phase 04 also writes `submissions/least_squares.csv` (not gitignored: commit it if it goes to AIcrowd).
- **Run**: `python experiments/run_phases.py least_squares` (the missing phases; `--dry-run` shows the plan), or one phase with `python experiments/least_squares/01_baseline.py`.

Settings, config files and the other scripts: [experiments/README.md](../README.md).
