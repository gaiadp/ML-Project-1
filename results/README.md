# Results

Every number of the report comes from the files in this folder. They are written by
the scripts of `experiments/` (see [experiments/README.md](../experiments/README.md))
and committed; a script that runs again overwrites its file.

All the cross-validations use the same 5 stratified folds, seed 1
(`stratified_k_indices` in `src/cross_validation.py`), so every file can be compared
fold by fold with the others. The seed also fixes SGD, the undersampling, Sophia's
minibatches and the trees, so a run with the same code, data and settings should give
the same numbers; the files have not been re-run to check it.

**Cost** is the sum of the `seconds` field of the rows (wall-clock time of the 5-fold
CV of every setting, preprocessing included, on the machine that ran it). Phase 04
and the leakage variants do not save it: "not measured".

## Files

| File | Contents | Produced by | Reproducible | Cost |
|---|---|---|---|---|
| `<method>/01_baseline.json` (13 methods) | Phase 01: one row per pipeline x setting of `FIT_GRID`, and the best row | `python experiments/<method>/01_baseline.py` (`run_grid`), or `python experiments/run_phases.py <method>` | Yes for the 6 methods of `implementations.py`, `random_forest`, `gradient_boosting`, `logistic_boosting`. **Not as they are** for the four optimizers (see the notes) | see the table below |
| `logistic_boosting/04_final.json` | Phase 04 of the final model: CV of the best row of `01_baseline`, out-of-fold threshold, fit on the whole training set | `python experiments/logistic_boosting/04_final.py` (`run_final`) | Yes; also writes `submissions/logistic_boosting.csv` | not measured |
| `reg_logistic_newton/04_final.json` | Phase 04 of Newton's logistic regression, from its `01_baseline` | `python experiments/reg_logistic_newton/04_final.py` (`run_final`) | Yes; also writes `submissions/reg_logistic_newton.csv` | not measured |
| `logistic_boosting/extra/04_final_harehab1.json` | Model B: the final model with `HAREHAB1` as a feature (`keep_leakage=True`) | `python experiments/leakage.py --variants harehab1` (`run_final` with `phase="extra/04_final_harehab1"`) | Yes; also writes `submissions/logistic_boosting_harehab1.csv` | not measured |
| `logistic_boosting/extra/04_final_rule.json` | Model C: the final model + known-positive rule (`f1_rule`) | `python experiments/leakage.py --variants rule` | Yes; also writes `submissions/logistic_boosting_rule.csv` | not measured |
| `reg_logistic_newton/ablation_data.json` | Data-preparation ablation: 16 pipeline variants x lambda in {3e-5, 1e-4, 1e-3}, Newton | `python experiments/ablation.py --parts data` | Yes (reads the pipeline of `logistic_boosting/04_final.json`) | 3858 s (64 min) |
| `logistic_boosting/ablation_model.json` | Model ablation: the final setting with `n_trees = 0` (the final row is reused from `04_final.json`) | `python experiments/ablation.py --parts model` | Yes | 158 s |
| `reg_logistic_regression/ablation_model.json` | Model ablation, row "plain GD": best setting of `reg_logistic_regression/01_baseline.json` on the final pipeline | `python experiments/ablation.py --parts model` | Yes | 287 s |
| `reg_logistic_newton/extra/01_lambda.json` | Finer lambda grid for Newton: lambda in {3e-5, 1e-4, 3e-4, 1e-3}, `max_iters=20`, pipeline v1 | **No dedicated script**: a one-off `run_grid` from the command line (see the notes) | Only with the command below | 335 s |
| `baselines.md`, `baselines.tex` | Best F1 of phase 01 per method and pipeline, next to trivial rules computed on the same folds | `python experiments/baselines.py` (reads `*/01_baseline.json`; `--rerun` runs phase 01 again first) | Yes, from the JSON files | not measured (no training) |
| `ablation.md`, `ablation.tex` | The two ablation tables (data preparation, model), with the leakage rows B and C | `python experiments/ablation.py` (reads the ablation JSON, `logistic_boosting/04_final.json` and the leakage variants; computes only the missing parts) | Yes, from the JSON files | not measured when every part is saved |

Cost of phase 01 per method (`results/<method>/01_baseline.json`):

| Method | Rows | Pipelines | Cost |
|---|---|---|---|
| `mean_squared_error_gd` | 9 | v0, full, v1 | 3820 s (64 min) |
| `mean_squared_error_sgd` | 9 | v0, full, v1 | 279 s |
| `least_squares` | 3 | v0, full, v1 | 277 s |
| `ridge_regression` | 21 | v0, full, v1 | 344 s |
| `logistic_regression` | 9 | v0, full, v1 | 2520 s (42 min) |
| `reg_logistic_regression` | 9 | v0, full, v1 | 2386 s (40 min) |
| `reg_logistic_adaptive_gd` | 2 | v1 | 462 s |
| `reg_logistic_newton` | 3 | v1 | 271 s |
| `reg_logistic_sophia` | 2 | v1 | 133 s |
| `reg_logistic_unixgrad` | 2 | v1 | 441 s |
| `random_forest` | 2 | v1 | 1538 s (26 min) |
| `gradient_boosting` | 2 | v1 | 4054 s (68 min) |
| `logistic_boosting` | 4 | v1 | 10458 s (2.9 h) |

## Notes

- **Four optimizers, phase 01.** `reg_logistic_adaptive_gd`, `reg_logistic_newton`,
  `reg_logistic_sophia` and `reg_logistic_unixgrad` have
  `PIPELINES = {"full": DEFAULT_CONFIG, "v1": V1_CONFIG}` in their `01_baseline.py`,
  but their saved files have only v1 rows: running the script again adds the `full`
  rows (and the best row may change). To get the same rows, run only phase 01 with a
  config file `{"01_baseline": {"PIPELINES": ["v1"]}}`:
  `python experiments/run_phases.py <method> --phases 01 --config <file>` (see
  experiments/README.md).
- **`reg_logistic_newton/extra/01_lambda.json`** has no script. Its rows (pipeline,
  grid, folds) correspond to this command, from the repository root:

  ```bash
  python -c "
  import runpy
  from experiments.common import param_grid, run_grid
  s = runpy.run_path('experiments/reg_logistic_newton/01_baseline.py')
  run_grid(s['METHOD'], 'extra/01_lambda', {'v1': s['V1_CONFIG']},
           param_grid(lambda_=[3e-5, 1e-4, 3e-4, 1e-3], max_iters=[20]), 5, 1)
  "
  ```

  Its `phase` field is `"01_lambda"`, not `"extra/01_lambda"`: the file was saved
  under another name and moved to `extra/` by hand. The command above writes
  `"extra/01_lambda"`.
- **Older format.** Files written before a field existed lack it: `f1_rule` is only in
  the two leakage variants, `f1_default` / `f1_default_folds` only in the files of
  `logistic_boosting`, `reg_logistic_newton/04_final.json` and the ablation files, and
  `known_positive_rule` only in the leakage variants. A re-run with the current code
  adds them. The pipeline configs saved before `keep_leakage` existed do not have the
  key; `src/pipeline.py` reads it as `False`.
- **Order of the runs.** `ablation.py` and `leakage.py` start from
  `logistic_boosting/04_final.json`: run phase 04 first. After a new phase 04, run them
  with `--force`.

## Format of the JSON files

- `run_grid` (phases 01-03, `ablation_*`, `extra/01_*`): `method`, `phase`, `date`,
  `k_fold`, `seed`, `rows` (one per pipeline x setting) and `best` (the row with the
  highest `f1_mean` among those that did not diverge).
- `run_final` (phase 04 and its variants): the same without `rows`, plus `from_phase`,
  `known_positive_rule`; `best` adds `threshold_oof`, `f1_oof`, `train_loss`,
  `n_features` (on the whole training set), `test_positive_rate`, `submission`.
- A row: `label` (pipeline), `pipeline` (the full config), `fit` (the parameters of
  `src.models.fit`), `f1_mean`, `f1_std`, `f1_folds` (F1 on the validation part of
  every fold, threshold from its training part: report `f1_mean`),
  `f1_val_best` (threshold chosen on the validation part: optimistic), `f1_default`,
  `f1_rule`, `threshold`, `n_features` (columns in fold 0), `finite`, `seconds`.

## Outside this folder

- `submissions/*.csv`: the predictions written by `run_final` and `run.py`. They are
  **tracked by git** (no longer in `.gitignore`): the files uploaded to AIcrowd stay in
  the repository, so `run.py` can be checked against them. A new run overwrites the
  file of its method: check `git diff --stat submissions/` before committing, and
  commit a new file only if it goes to AIcrowd. The `submission` field of a phase 04
  JSON says which file it wrote.
- `build/`: gitignored. `build/data.npz` (data cache, `src/data.py`),
  `build/imbalance.txt` (`experiments/imbalance.py`), `build/eda/` (`experiments/eda.py`).
