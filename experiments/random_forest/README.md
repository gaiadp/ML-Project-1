# random_forest

Random forest of histogram trees (`src/extra_implementations/random_forest.py`, `trees.py`): the score is the mean over the trees of the fraction of MICHD in the leaf.

- **Phases**: `01_baseline.py` (pipeline v1 only: the trees are slow) and `04_final.py` (`FROM_PHASE = "01_baseline"`); no 02 / 03: the trees build interactions by themselves.
- **Grid** (`FIT_GRID` in the settings block of the scripts): `n_trees`, `max_depth`, `min_samples_leaf`.
- **Results** in `results/random_forest/`: `01_baseline.json` (phase 04 not run yet). Phase 04 also writes `submissions/random_forest.csv` (not gitignored: commit it if it goes to AIcrowd).
- **Run**: `python experiments/run_phases.py random_forest` (the missing phases; `--dry-run` shows the plan), or one phase with `python experiments/random_forest/01_baseline.py`.

Settings, config files and the other scripts: [experiments/README.md](../README.md).
