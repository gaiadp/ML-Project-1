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
CV of every setting, preprocessing included, on the machine that ran it). Phase 04,
the `extra/04_final_*` files, the ensemble and the feature screening do not save it:
"not measured", with the estimate of the script's docstring when it gives one.

**Official phases and extra scripts.** `run_phases.py` runs only the phase scripts
`experiments/<method>/0*_*.py` (`01_baseline`, `02_degree`, `02b_transforms`,
`03_interactions`, `04_final`), which write `results/<method>/<phase>.json`. The
`extra_*.py` scripts, `ensemble.py` and `feature_screening.py` are run by hand and
write to `results/<method>/extra/` or `results/ensemble/`: they never overwrite a
phase file such as `04_final.json`.

**Which model is submitted:** the ensemble (`ensemble/ensemble.json`, `final`: 0.35 x
`reg_logistic_newton` + 0.65 x model A, without the known-positive rule), reproduced by
`run.py`. Below, "model A" is `logistic_boosting` phase 04 (`04_final.json`), the
second component of the ensemble and the reference (`FINAL_METHOD`) of `ablation.py`
and `leakage.py`; the ensemble is described in its own rows.

## Files

| File | Contents | Produced by | Reproducible | Cost |
|---|---|---|---|---|
| `<method>/01_baseline.json` (13 methods) | Phase 01: one row per pipeline x setting of `FIT_GRID`, and the best row | `python experiments/<method>/01_baseline.py` (`run_grid`), or `python experiments/run_phases.py <method>` | Yes for the 6 methods of `implementations.py`, `random_forest`, `gradient_boosting`, `logistic_boosting`. **Not as they are** for the four optimizers (see the notes) | see the table below |
| `logistic_boosting/04_final.json` | Phase 04 of `logistic_boosting` (model A): CV of the best row of `01_baseline`, out-of-fold threshold, fit on the whole training set | `python experiments/logistic_boosting/04_final.py` (`run_final`) | Yes; also writes `submissions/logistic_boosting.csv` | not measured |
| `reg_logistic_newton/04_final.json` | Phase 04 of Newton's logistic regression, from its `01_baseline` | `python experiments/reg_logistic_newton/04_final.py` (`run_final`) | Yes; also writes `submissions/reg_logistic_newton.csv` | not measured |
| `logistic_boosting/extra/04_final_harehab1.json` | Model B: model A with `HAREHAB1` as a feature (`keep_leakage=True`) | `python experiments/leakage.py --variants harehab1` (`run_final` with `phase="extra/04_final_harehab1"`) | Yes; also writes `submissions/logistic_boosting_harehab1.csv` | not measured |
| `logistic_boosting/extra/04_final_rule.json` | Model C: model A + known-positive rule (`f1_rule`) | `python experiments/leakage.py --variants rule` | Yes; also writes `submissions/logistic_boosting_rule.csv` | not measured |
| `reg_logistic_newton/ablation_data.json` | Data-preparation ablation: 16 pipeline variants x lambda in {3e-5, 1e-4, 1e-3}, Newton | `python experiments/ablation.py --parts data` | Yes (reads the pipeline of `logistic_boosting/04_final.json`) | 3858 s (64 min) |
| `logistic_boosting/ablation_model.json` | Model ablation: the setting of model A with `n_trees = 0` (the row of model A is reused from `04_final.json`) | `python experiments/ablation.py --parts model` | Yes | 158 s |
| `reg_logistic_regression/ablation_model.json` | Model ablation, row "plain GD": best setting of `reg_logistic_regression/01_baseline.json` on the pipeline of model A | `python experiments/ablation.py --parts model` | Yes | 287 s |
| `reg_logistic_newton/extra/01_lambda.json` | Finer lambda grid for Newton: lambda in {3e-5, 1e-4, 3e-4, 1e-3}, `max_iters=20`, pipeline v1 | **No dedicated script**: a one-off `run_grid` from the command line (see the notes) | Only with the command below | 335 s |
| `reg_logistic_newton/02_degree.json` | Phase 02: `poly_degree` 1, 2, 3 on the best pipeline of `01_baseline` (v1) x lambda in {1e-6, 1e-4, 1e-2} | `python experiments/reg_logistic_newton/02_degree.py`, or `run_phases.py` | Yes | 1213 s (20 min) |
| `reg_logistic_newton/02b_transforms.json` | Phase 02b: `extra_transforms` none / sqrt / log / sqrt+log on the best pipeline of `02_degree` x the same lambdas | `python experiments/reg_logistic_newton/02b_transforms.py`, or `run_phases.py` (also prints `check_transforms`, no file) | Yes | 818 s (14 min) |
| `reg_logistic_newton/03_interactions.json` | Phase 03: `interaction_top` 0, 10, 20, 30 on the best pipeline of `02b_transforms` x the same lambdas | `python experiments/reg_logistic_newton/03_interactions.py`, or `run_phases.py` | Yes | 1120 s (19 min) |
| `reg_logistic_newton/extra/04_final_interactions.json` | Extra final model of Newton with the best row of `03_interactions` (degree 2, log, top 20); `04_final.json` stays the one from `01_baseline` | `python experiments/reg_logistic_newton/extra_final_interactions.py` (`run_final` with `phase="extra/04_final_interactions"`) | Yes; also writes `submissions/reg_logistic_newton_interactions.csv` (not in the repository) | not measured |
| `reg_logistic_newton/extra/feature_screening.json` | Hand-made features: level 1 score test of 6 groups against the best row of `03_interactions` (reference: the top 20 interactions against `02b_transforms`), level 2 5-fold CV of the groups that pass | `python experiments/feature_screening.py` (`--level 1`: only the score test, and the file is rewritten with an empty `level2`) | Yes | not measured (docstring: seconds per group at level 1, ~4 min per group at level 2) |
| `logistic_boosting/extra/03_interactions.json` | Step 1 of `extra_interactions.py`: pipeline of `01_baseline` + top 20 interactions, `n_trees` 200 and 400, `lambda_=1e-4`, `max_iters=20`, `max_depth=3`, `learning_rate=0.05` | `python experiments/logistic_boosting/extra_interactions.py` | Yes | 7025 s (1.95 h) |
| `logistic_boosting/extra/04_final_interactions.json` | Step 2 of `extra_interactions.py`: final model with the best row of step 1; `04_final.json` is not touched | `python experiments/logistic_boosting/extra_interactions.py` (a step whose file exists is skipped, `--force`) | Yes; also writes `submissions/logistic_boosting_interactions.csv` (not in the repository) | not measured (docstring: "~3-3.5 h in all" for both steps) |
| `ensemble/ensemble.json` | Ensemble `p = a p_newton + (1 - a) p_boosting` of `reg_logistic_newton` `extra/04_final_interactions` and `logistic_boosting` `04_final`: CV of each component alone and of the ensemble (weight `a` in {0, 0.05, ..., 1} and threshold chosen on the other folds), final weight and threshold | `python experiments/ensemble.py` | Yes; also writes `submissions/ensemble_newton_boosting.csv` (the submitted file) and `submissions/ensemble_newton_boosting_rule.csv` | from the cache: not measured; without it: not measured (docstring: "~1 h for logistic_boosting (5 folds + the full fit), a few minutes for Newton") |
| `ensemble/scores_reg_logistic_newton.npz`, `ensemble/scores_logistic_boosting.npz` | Cache of the ensemble: `oof` (328135 out-of-fold probabilities) and `test` (109379 test probabilities) of each component | `python experiments/ensemble.py`, only when the file is missing or with `--force` | Yes | included in the ensemble row |
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

## The ensemble cache

`results/ensemble/scores_<method>.npz` store, for each component of the ensemble, the
out-of-fold probability of every training row (`oof`) and the probability of every
test row (`test`), from the CV and the full fit of its setting. When both files exist,
`python experiments/ensemble.py` only loads the data, chooses the weight and the
threshold and writes the two ensemble submissions: it regenerates them in a few seconds
without training anything. The results of the extra scripts
([experiments/README.md](../experiments/README.md#results-of-the-extra-scripts)) report
that the regenerated submission is identical byte for byte to the first one (not re-checked
here). The cache is not invalidated by itself: after a change of
`logistic_boosting/04_final.json` or `reg_logistic_newton/extra/04_final_interactions.json`,
run `python experiments/ensemble.py --force`, which trains the two components again.

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
- **Older format.** Files written before a field existed lack it. `f1_rule` is only in
  the leakage variants and in the files dated from 2026-10-21 (`reg_logistic_newton`
  02 / 02b / 03 and the `extra/*_interactions` files); `f1_default` /
  `f1_default_folds` also in the `logistic_boosting` files,
  `reg_logistic_newton/04_final.json` and the ablation files; `known_positive_rule`
  only in the `extra/04_final_*` files. A re-run with the current code adds them. The
  pipeline configs saved before `keep_leakage` or `extra_transforms` existed do not
  have the key (`extra_transforms` is missing also in `reg_logistic_newton/02_degree.json`);
  `src/pipeline.py` reads them as `False` and `()`.
- **Newton: phases 02-03 and phase 04.** `reg_logistic_newton/04_final.py` has
  `FROM_PHASE = "01_baseline"`, so `04_final.json` does not use phases 02-03; the final
  model with their expansions is `extra/04_final_interactions.json`.
- **`ridge_regression`** has a `02b_transforms.py` phase, but no result of phases
  02-04 yet, and its `03_interactions.py` starts from `02_degree`, not from
  `02b_transforms`.
- **Order of the runs.** `ablation.py` and `leakage.py` start from
  `logistic_boosting/04_final.json`: run phase 04 first. After a new phase 04, run them
  with `--force`.

## Format of the JSON files

- `run_grid` (phases 01-03 and 02b, `ablation_*`, `extra/01_lambda`,
  `logistic_boosting/extra/03_interactions`): `method`, `phase`, `date`,
  `k_fold`, `seed`, `rows` (one per pipeline x setting) and `best` (the row with the
  highest `f1_mean` among those that did not diverge).
- `run_final` (phase 04 and its variants): the same without `rows`, plus `from_phase`,
  `known_positive_rule`; `best` adds `threshold_oof`, `f1_oof`, `train_loss`,
  `n_features` (on the whole training set), `test_positive_rate`, `submission`.
- `ensemble.json`: `date`, `components` (method -> result file of its setting),
  `k_fold`, `seed`, `weights` (the grid of `a`), `cv` (per model: `f1_mean`, `f1_std`,
  `f1_folds`, `f1_rule`, `f1_rule_folds`, `weight_folds`) and `final` (`weight`,
  `threshold`, `f1_oof`, `test_positive_rate`, `submission`, `f1_rule_cv`,
  `test_known_positives`, `test_positive_rate_rule`, `submission_rule`).
- `feature_screening.json`: `method`, `phase`, `date`, `from_phase`,
  `reference_phase`, `k_fold`, `seed`, `level1` (per group: `df`, `chi2`, `p`,
  `delta_loss`, `top20`), `passed` and `level2` (per group: `f1_mean`, `f1_std`,
  `f1_folds`, `diff`, `better`, `n_features`).
- A row: `label` (pipeline), `pipeline` (the full config), `fit` (the parameters of
  `src.models.fit`), `f1_mean`, `f1_std`, `f1_folds` (F1 on the validation part of
  every fold, threshold from its training part: report `f1_mean`),
  `f1_val_best` (threshold chosen on the validation part: optimistic), `f1_default`,
  `f1_rule`, `threshold`, `n_features` (columns in fold 0), `finite`, `seconds`.

## Outside this folder

- `submissions/*.csv`: the predictions written by `run_final`, `ensemble.py` and
  `run.py`. They are
  **tracked by git** (no longer in `.gitignore`): the files uploaded to AIcrowd stay in
  the repository, so `run.py` can be checked against them. A new run overwrites the
  file of its method: check `git diff --stat submissions/` before committing, and
  commit a new file only if it goes to AIcrowd. The `submission` field of a phase 04
  JSON (`submission`, `submission_rule` in `ensemble.json`) says which file it wrote.
  The csv of the `extra/04_final_interactions` files are not in the repository; the
  outputs of `run.py` (`submission.csv`) and `experiments/run_model_a.py`
  (`submission_model_a.csv`) are gitignored: compare them with the tracked files.
- `build/`: gitignored. `build/data.npz` (data cache, `src/data.py`),
  `build/imbalance.txt` (`experiments/imbalance.py`), `build/eda/` (`experiments/eda.py`).
