# Tests

Our sanity tests: 142 tests in 13 files (`pytest --collect-only`). They use small fake
data and need neither `dataset/` nor the cache `build/data.npz` (`test_data.py` writes
its fake CSV files to a temporary folder). Every file puts the repository root on
`sys.path`, so `src`, `experiments`, `implementations` and `helpers` import from any
working directory; run them from the repository root.

## Running

```bash
pytest tests/                               # all the tests
pytest tests/test_pipeline.py               # one file
pytest tests/test_models.py -k undersample  # tests whose name matches
pytest --collect-only -q tests/             # list the tests without running them
```

`requirements.txt` pins `pytest==7.1.2`. The official grading tests of the course are
separate (https://github.com/epfml/ML_course/tree/main/projects/project1/grading_tests);
as the docstring of `test_implementations.py` says, they run with
`pytest --github_link <repo-url> .` and need `pytest-mock` and `gitpython`, also
pinned in `requirements.txt`.

## What each file covers

| File | Tests | Covers |
|---|---|---|
| `test_implementations.py` | 24 | The 6 methods of `implementations.py`: GD converges to `least_squares`, ridge with lambda = 0 equals least squares, regularized logistic with lambda = 0 equals logistic; gradients against finite differences; losses against reference formulas and without the penalty term; output shapes; the logistic helpers against lab ex05 and the public values; no overflow for large scores; `initial_w` not modified; the loss decreases. |
| `test_models.py` | 27 | `src/models.py`: label conversion, {0, 1} labels for the logistic methods and {-1, 1} for the linear ones, reproducible SGD seed, every method (the 6 and the extra ones) classifies separable data, probabilities as scores, class weights (`weighted_logistic_regression`, its gradient, `pos_weight` only for logistic methods), undersampling, errors for unknown methods and options. |
| `test_extra_implementations.py` | 22 | `src/extra_implementations`: Hessian against finite differences, Newton equals a long gradient descent, every optimizer reaches the minimum and follows the conventions of `reg_logistic_regression`, adaptive GD decreases the objective, Sophia's clipping and seed, UniXGrad stays in its ball; histogram bins and best split, the trees learn XOR, trees without splits predict the prevalence, boosting loss decreases with more trees, `logistic_boosting` without trees equals Newton and learns what the linear model misses. |
| `test_pipeline.py` | 18 | `stratified_split` and `src/pipeline.py`: statistics from the training set only, `V0_CONFIG` with and without one-hot, the full pipeline, one-hot, binning, one-hot of the ordinals, duplicate drop, `HAREHAB1` dropped also without `domain_drop` and kept only with `keep_leakage`, the `V1_CONFIG` selection and selection by rule, labels needed only by the steps that use them, polynomial and interaction expansion (B11) and its errors, every step can be switched off. |
| `test_preprocessing.py` | 11 | `src/preprocessing.py`: special codes depend on the column, "none" codes become 0, codes checked against the codebook, input not modified, unit conversion, skip logic (and that its column names exist), missing-rate difference, one indicator per group, imputation and binary recoding by type, binning of `PHYSHLTH`. |
| `test_feature_types.py` | 8 | Consistency of the lists in `src/feature_types.py`: 321 columns with a valid type, `DROP` type and `DOMAIN_DROP` agree, label leakage dropped, every dropped column has a reason, `CONVERTED`, `REDUNDANT`, one-hot and binned columns, `NOT_SELECTED` only removes columns the other lists keep. |
| `test_cross_validation.py` | 7 | `src/cross_validation.py`: stratified folds, the preprocessing never sees the validation rows, reproducible CV with SGD, `cross_validate_grid` equals `cross_validate`, `cross_validation_ridge` equals `ridge_regression`, `f1_default` uses the natural threshold, `f1_rule` sets the known positives. |
| `test_eda.py` | 7 | Statistics helpers of `experiments/eda.py`: mutual information, discretization, determination, pairwise correlation, F1 of the best rule, log-odds fit of a linear column, train / test shift distance. |
| `test_run_phases.py` | 5 | `experiments/run_phases.py`, no data and no training: phase scripts of a method, `--phases` selection, the run / skip plan, conversion of the config-file settings, loading the settings of a phase script. |
| `test_ablation.py` | 4 | `experiments/ablation.py` on fake rows and data: the data-preparation variants, every variant runs through the pipeline, the data and model tables. |
| `test_baselines.py` | 4 | `experiments/baselines.py`: keys of the trivial rules, prevalence scores, the trivial baselines, best row per pipeline. |
| `test_rules.py` | 3 | `src/rules.py` and `experiments/leakage.py`: known positives from the raw answers, `apply_known_positives`, the leakage variants are valid options of `run_final`. |
| `test_data.py` | 2 | `src/data.py`: the cache round trip (written, then reloaded without the CSV files) and the check of the label row order. |
