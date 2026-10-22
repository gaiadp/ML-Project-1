# gradient_boosting

Gradient boosting of histogram trees on the log-loss (`src/extra_implementations/gradient_boosting.py`, `trees.py`), started from the log-odds of the training prevalence.

- **Phases**: `01_baseline.py` (pipeline v1 only: the trees are slow) and `04_final.py` (`FROM_PHASE = "01_baseline"`); no 02 / 03: the trees build interactions by themselves.
- **Grid** (`FIT_GRID` in the settings block of the scripts): `n_trees`, `learning_rate`, `max_depth`.
- **Results** in `results/gradient_boosting/`: `01_baseline.json` (phase 04 not run yet). Phase 04 also writes `submissions/gradient_boosting.csv` (not gitignored: commit it if it goes to AIcrowd).
- **Run**: `python experiments/run_phases.py gradient_boosting` (the missing phases; `--dry-run` shows the plan), or one phase with `python experiments/gradient_boosting/01_baseline.py`.

Settings, config files and the other scripts: [experiments/README.md](../README.md).
