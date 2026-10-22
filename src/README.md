# src

Reusable code shared by `run.py`, `experiments/` and `tests/`. NumPy only, plus
matplotlib / seaborn in `plots.py`. The 6 required methods stay in
`implementations.py` at the repository root, and `helpers.py` (from the course) reads
the CSV files and writes the submissions.

Contents:

1. [Modules](#modules)
2. [The option `extra_transforms`](#the-option-extra_transforms)
3. [How they connect](#how-they-connect)
4. [Principles](#principles)

## Modules

| Module | Contents |
|---|---|
| `data.py` | `load_data()` returns `(x_train, x_test, y_train, train_ids, test_ids, feature_names)`. The first call runs `build_cache()`: it reads `dataset/*.csv` with `helpers.load_csv_data`, checks that train and test have the same columns and that `y_train.csv` is in the row order of `x_train.csv`, and saves `build/data.npz` (uncompressed, gitignored). `python src/data.py` builds the cache. |
| `feature_types.py` | Column lists from the BRFSS 2015 codebook, data only: `DOMAIN_DROP` (columns that are not features, with the reason), `LABEL_LEAKAGE` (`HAREHAB1`), `FEATURE_TYPES` (one of `TYPES` for each of the 321 columns), `BINNED`, `ONE_HOT_ORDINAL`, `REDUNDANT`, `CONVERTED`, and `NOT_SELECTED` with `SELECTION_RULES` (`R1`-`R4`), the feature selection of `V1_CONFIG`. |
| `preprocessing.py` | The cleaning steps as functions: `codes_to_nan` (BRFSS special codes, assigned column by column), `convert_units`, `fill_skip_logic`, `bin_columns`, `missing_rate_difference` and `one_per_group` (informative missing values), and the fit / apply pairs `fit_impute_median` or `fit_impute_by_type` / `apply_impute`, `fit_binary_recode` / `apply_binary_recode`, `fit_standardize` / `apply_standardize`. |
| `features.py` | `add_bias`, `one_hot_encode`, `fit_one_hot_categories`, `build_poly`, `pairwise_products`. |
| `pipeline.py` | Chains the steps B1-B13 (listed in its docstring). `fit_preprocess(x_train, y_train, names, config)` returns a `state` dict with every fitted statistic; `apply_preprocess(x, names, state)` applies it to raw data and returns `tx`, bias column first (`state["out_names"]` are the column names). Configs: `DEFAULT_CONFIG` (the full pipeline, `full` in the results), `V0_CONFIG` (the baseline: special codes, columns with > 50% NaN dropped, median imputation, one-hot, standardization), `V1_CONFIG` (the full pipeline on the 124 columns kept by the EDA selection). `not_selected(rules)` lists the columns removed by some selection rules. Step B11 also has the option `extra_transforms` (below). |
| `rules.py` | Known-positive rule, not part of the model: `known_positives(x, names)` finds, on the raw data, the rows where a `LABEL_LEAKAGE` question was asked; `apply_known_positives(labels, known)` sets them to 1 after the threshold. |
| `models.py` | One interface for every method: `fit(method, y, tx, ...)`, `predict_scores`, `predict`, `default_threshold`. Method lists: `METHODS` (the 6 of `implementations.py`), `LOGISTIC_OPTIMIZERS`, `TREE_METHODS`, `EXTRA_METHODS`, `ALL_METHODS`. Class imbalance options of `fit`: `pos_weight` (`weighted_logistic_regression`, logistic methods only) and `neg_per_pos` (`undersample_indices`). Label conversion: `to_01`, `to_pm1`. |
| `metrics.py` | `confusion_counts`, `accuracy`, `precision`, `recall`, `fbeta_score`, `f1_score`, `predict_labels(scores, threshold)` and `best_threshold(scores, y)` (the threshold that maximizes F1, every distinct score tried in O(N log N)). Labels in {-1, 1} or {0, 1}: the positive class is 1. |
| `cross_validation.py` | `stratified_k_indices`, `stratified_split`, `cross_validate` and `cross_validate_grid` (k-fold CV of preprocessing + model, see below), `ridge_weights_svd` and `cross_validation_ridge` (a whole lambda grid from one SVD per fold). |
| `plots.py` | matplotlib helpers for `experiments/eda.py`: `plot_missing_and_balance`, `plot_ranking`, `plot_mi_vs_corr`, `plot_dependence_grid`, `plot_corr_heatmap`. |
| `extra_implementations/` | Methods beyond the 6, NumPy only, labels in {0, 1}, called through `models.py`. Optimizers of the `reg_logistic_regression` objective (`logistic_objective.py`): `reg_logistic_adaptive_gd` (`adaptive_gd.py`), `reg_logistic_newton` (`newton.py`), `reg_logistic_sophia` (`sophia.py`), `reg_logistic_unixgrad` (`unixgrad.py`). Histogram trees (`trees.py`): `random_forest`, `gradient_boosting`, and the hybrid `logistic_boosting` (Newton logistic regression plus boosted trees on its residuals), each with a `*_predict_proba` function. |

## The option `extra_transforms`

Part of step B11 of `src/pipeline.py`, off by default (`extra_transforms=()` in
`DEFAULT_CONFIG`, so `V0_CONFIG` and `V1_CONFIG` too). With `("sqrt",)`, `("log",)` or
`("sqrt", "log")`, `fit_preprocess` adds new columns next to the original ones:

- **Which columns:** the kept columns of type `CONTINUOUS` or `COUNT` with more than
  `TRANSFORM_MIN_DISTINCT = 10` distinct values in the training set. The one-hot
  columns (nominal, binned, `ONE_HOT_ORDINAL`) and the missing indicators are left out.
- **Which values:** `TRANSFORMS = {"sqrt": np.sqrt, "log": np.log1p}` applied to
  `x - min`, where `min` is the training minimum of the column (`transform_shift`), on
  the imputed values before the standardization B12; values below the training
  minimum are clipped at 0. So the columns are `sqrt(x - min)` and `log(1 + x - min)`,
  named `"sqrt(NAME)"`, `"log(NAME)"`.
- **Standardization:** with the powers and products of B11, with their own training
  mean / std (`expand_mean`, `expand_std`).
- **State:** `transforms`, `transform_cols`, `transform_shift`; `apply_preprocess`
  reuses them, nothing is refitted. An unknown name raises `ValueError`.

It is used by the phases `02b_transforms` of `reg_logistic_newton` and
`ridge_regression`; before their CV, `experiments.common.check_transforms` prints a
correlation check of the transforms on the whole training set (a printout only: it
chooses nothing, the CV decides).

## How they connect

```
dataset/*.csv --helpers.load_csv_data--> data.load_data()   (cache build/data.npz)
    x: raw data, NaN = missing;  y in {-1, 1};  names
        |
pipeline.fit_preprocess(x_train, y_train, names, config) -> state
pipeline.apply_preprocess(x, names, state) -> tx      (training, validation, test alike)
        |
models.fit(method, y, tx, **params) -> w  (a model dict for the trees)
models.predict_scores(method, w, tx) -> scores
        |
metrics.best_threshold(scores, y) -> threshold
metrics.predict_labels(scores, threshold) -> labels in {-1, 1}
        |
[rules.apply_known_positives]   only in the declared "rule" variant
helpers.create_csv_submission(test_ids, labels, path)
```

Where this chain runs:

- `cross_validation.cross_validate_grid` (phases 01-03, and the CV of phase 04): for
  every fold of `stratified_k_indices`, `fit_preprocess` on the training part, `fit`,
  threshold on the training scores, F1 on the validation part. It also returns
  `f1_val_best` (threshold chosen on the validation part: optimistic), `f1_default`
  (`default_threshold`: 0.5 for probabilities, 0 for the linear methods), `f1_rule`
  (after `apply_known_positives`) and the out-of-fold scores.
- `experiments/common.py`, `run_final` (phase 04): threshold on the out-of-fold
  scores of the CV, then `fit_preprocess` and `fit` on the whole training set and
  predictions for `x_test` in `submissions/<method>.csv`.
- `run.py`: model A (`logistic_boosting`) with the setting and the out-of-fold
  threshold of `results/logistic_boosting/04_final.json`, written as constants:
  `fit_preprocess` and `fit` on the whole training set, no CV, predictions in
  `submissions/submission.csv`.
- `experiments/ensemble.py`: the out-of-fold and test probabilities of two final
  settings (`cross_validate`, then a full fit), averaged with a weight chosen on the
  out-of-fold probabilities, like the threshold.

## Principles

1. **Preprocessing is fitted on training rows only.** Every statistic (kept columns,
   informative missing values, imputation values, one-hot categories, mean / std,
   interaction columns) comes from `fit_preprocess` on the training part. In CV it
   runs inside every fold (`_preprocess_fold`), never on the whole training set
   before the split: B5, B6 and `interaction_top` use the labels. `apply_preprocess`
   never refits. Tests: `test_preprocessing_never_sees_the_validation_rows`,
   `test_pipeline_uses_training_statistics_only`.
2. **Labels are in {-1, 1} outside `models.py`.** `load_data`, the pipeline, the CV,
   the metrics, the rules and the submissions use {-1, 1} (MICHD = 1). `fit` converts
   to {0, 1} with `to_01` for the logistic methods and every extra method, and
   `predict` returns {-1, 1}: call the methods through `models.py`, not
   `implementations.py` or `extra_implementations/` directly. (`experiments/eda.py`
   builds its own 0/1 indicator `y01` for statistics and plots.)
3. **The threshold is never chosen on the rows it is scored on.** In the CV it comes
   from the training scores of the fold; in phase 04 from the out-of-fold scores.
   `f1_val_best` is only a comparison. `run.py` uses the out-of-fold threshold saved
   by phase 04; `ensemble.py` chooses weight and threshold on the other folds in its
   CV.
4. **`HAREHAB1` never enters a reported model.** `LABEL_LEAKAGE` is dropped by every
   config, also without `domain_drop`; `keep_leakage=True` is used only by
   `experiments/leakage.py` to measure the leakage. The known-positive rule works on
   the raw data, after the threshold, and is reported separately.
5. **Same folds everywhere.** `stratified_k_indices(y, 5, seed=1)` in every
   experiment; the same seed goes to SGD, the undersampling, Sophia's minibatches
   and the trees.
6. **Saved configs must keep loading.** Phases reuse the pipeline config stored in
   `results/*.json` (`load_best`). Keys added later are read with
   `config.get(key, default)` (`keep_leakage`, `poly_degree`, `interaction_top`,
   `interaction_pairs`, `extra_transforms`), so older results still work; a new key
   should do the same.
