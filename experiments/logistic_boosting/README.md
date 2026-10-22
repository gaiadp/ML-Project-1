# logistic_boosting

The final model: Newton's logistic regression plus gradient boosting of histogram trees on its residuals (`src/extra_implementations/logistic_boosting.py`), so the trees only learn what the linear model misses.

- **Phases**: `01_baseline.py` (pipeline v1 only: the trees are slow) and `04_final.py` (`FROM_PHASE = "01_baseline"`); no 02 / 03: the trees build interactions by themselves.
- **Grid** (`FIT_GRID` in the settings block of the scripts): `lambda_`, `max_iters` (Newton part), `n_trees`, `max_depth`, `learning_rate`.
- **Results** in `results/logistic_boosting/`: `01_baseline.json`, `04_final.json`, `ablation_model.json` (ablation), `extra/04_final_harehab1.json` and `extra/04_final_rule.json` (`experiments/leakage.py`). Phase 04 also writes `submissions/logistic_boosting.csv` (not gitignored: commit it if it goes to AIcrowd).
- **Run**: `python experiments/run_phases.py logistic_boosting` (the missing phases; `--dry-run` shows the plan), or one phase with `python experiments/logistic_boosting/01_baseline.py`.

Settings, config files and the other scripts: [experiments/README.md](../README.md).
