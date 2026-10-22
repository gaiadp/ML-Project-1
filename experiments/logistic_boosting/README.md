# logistic_boosting

Newton's logistic regression plus gradient boosting of histogram trees on its residuals (`src/extra_implementations/logistic_boosting.py`), so the trees only learn what the linear model misses. Its phase 04 is model A: the model of `run.py` and the reference of `experiments/ablation.py` and `experiments/leakage.py`. Which model is submitted: [FINAL MODEL: to be decided].

- **Phases** (run by `run_phases.py`): `01_baseline.py` (pipeline v1 only: the trees are slow) and `04_final.py` (`FROM_PHASE = "01_baseline"`).
- **Extra script** (not a phase): `extra_interactions.py`, the pipeline of `01_baseline` + top 20 interactions (CV grid, then final model) -> `extra/03_interactions.json`, `extra/04_final_interactions.json` and `submissions/logistic_boosting_interactions.csv`; `04_final.json` is not touched. `04_final.json` is one of the two components of `experiments/ensemble.py`.
- **Grid** (`FIT_GRID` in the settings block of the scripts): `lambda_`, `max_iters` (Newton part), `n_trees`, `max_depth`, `learning_rate`.
- **Results** in `results/logistic_boosting/`: `01_baseline.json`, `04_final.json`, `ablation_model.json` (ablation), `extra/04_final_harehab1.json` and `extra/04_final_rule.json` (`experiments/leakage.py`), `extra/03_interactions.json` and `extra/04_final_interactions.json` (`extra_interactions.py`). Phase 04 also writes `submissions/logistic_boosting.csv`.
- **Run**: `python experiments/run_phases.py logistic_boosting` (the missing phases; `--dry-run` shows the plan), or `python experiments/logistic_boosting/extra_interactions.py` (`--force` to compute its steps again).

Settings, config files and the other scripts: [experiments/README.md](../README.md).
