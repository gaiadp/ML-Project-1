# Interactions, ensemble and feature screening

Results of the extra scripts (not phases) on the two methods with the highest F1,
`reg_logistic_newton` and `logistic_boosting`. Written by hand from the JSON files
named in each section (no script writes this file). Every F1 is a 5-fold CV with the
same folds as every other file (`k_fold = 5`, `seed = 1`): "folds better" compares the
row with its reference fold by fold. The standard deviation between folds is about
0.002, so a smaller difference is noise unless it holds in every fold. `HAREHAB1` is
never a feature; the column "F1 + rule" adds the known-positive rule after the
threshold (`src/rules.py`), only to measure the leakage.

## 1. Newton: degree, transforms, interactions

Phases 02 -> 02b -> 03 of `reg_logistic_newton`
(`reg_logistic_newton/02_degree.json`, `02b_transforms.json`, `03_interactions.json`),
`max_iters = 20`, lambda in {1e-6, 1e-4, 1e-2} (best for every variant). Each phase
starts from the best pipeline of the previous one (in bold); the reference of a
phase is its first row.

| Phase | Variant | F1 | Δ F1 | folds better | lambda | columns | F1 + rule |
|---|---|---|---|---|---|---|---|
| 02 degree | 1 (pipeline v1) | 0.4323 ± 0.0020 | reference | – | 1e-4 | 334 | 0.4382 |
| | **2** | 0.4323 ± 0.0022 | +0.0000 | 3/5 | 1e-4 | 390 | 0.4384 |
| | 3 | 0.4321 ± 0.0017 | -0.0002 | 1/5 | 1e-6 | 444 | 0.4379 |
| 02b transforms (degree 2) | none | 0.4323 ± 0.0022 | reference | – | 1e-4 | 390 | 0.4384 |
| | sqrt | 0.4322 ± 0.0015 | -0.0001 | 3/5 | 1e-4 | 427 | 0.4382 |
| | **log** | 0.4324 ± 0.0018 | +0.0001 | 2/5 | 1e-4 | 427 | 0.4385 |
| | sqrt + log | 0.4319 ± 0.0015 | -0.0004 | 2/5 | 1e-4 | 464 | 0.4379 |
| 03 interactions (degree 2, log) | top 0 | 0.4324 ± 0.0018 | reference | – | 1e-4 | 427 | 0.4385 |
| | top 10 | 0.4336 ± 0.0012 | +0.0012 | 4/5 | 1e-4 | 472 | 0.4398 |
| | **top 20** | 0.4339 ± 0.0016 | +0.0015 | 5/5 | 1e-6 | 617 | 0.4398 |
| | top 30 | 0.4336 ± 0.0027 | +0.0012 | 3/5 | 1e-4 | 862 | 0.4396 |

"top k": the products of all the pairs of the k columns most correlated with the
label, chosen on the training part of each fold (k (k - 1) / 2 products: 190 for
k = 20). Columns: in fold 0.

- Powers and the sqrt / log transforms change F1 by at most 0.0004: the log-odds of
  most columns are already close to linear after the encoding (EDA, analysis A6).
- The interactions add +0.0015 in all 5 folds; 30 columns do not improve on 20.
- lambda = 1e-6 is the smallest value of the grid (row "top 20"; with 1e-4 the F1 is
  also 0.4339).

Final model with these expansions (`reg_logistic_newton/extra/04_final_interactions.json`,
`experiments/reg_logistic_newton/extra_final_interactions.py`; `04_final.json` stays
the one of pipeline v1): CV 0.4339 ± 0.0016, out-of-fold threshold 0.196 (out-of-fold
F1 0.4344), 619 columns on the whole training set, 14.70% predicted positive on the
test set. It is the first component of the submitted ensemble (section 3).

## 2. Logistic boosting with interactions

`logistic_boosting/extra/03_interactions.json` and `extra/04_final_interactions.json`
(`experiments/logistic_boosting/extra_interactions.py`): pipeline v1 + top 20
interactions, `lambda_ = 1e-4`, `max_iters = 20`, `max_depth = 3`,
`learning_rate = 0.05`. Reference: the rows of `logistic_boosting/01_baseline.json`
with the same fit, without interactions.

| Trees | With interactions | Without (`01_baseline`) | Δ F1 | folds better | columns |
|---|---|---|---|---|---|
| 200 | 0.4330 ± 0.0017 | 0.4335 ± 0.0022 | -0.0006 | 1/5 | 524 |
| 400 | 0.4333 ± 0.0025 | 0.4338 ± 0.0020 | -0.0005 | 1/5 | 524 |

Final model with interactions (`extra/04_final_interactions.json`): CV 0.4333 ±
0.0025, out-of-fold F1 0.4347, 526 columns, 14.36% predicted positive. Not used: the
interactions do not help the boosted trees, which already split on combinations of
columns. Monotone transforms (sqrt, log) were not tried for the trees: a split on
log x divides the rows as a split on x does.

## 3. Ensemble Newton + logistic boosting

`ensemble/ensemble.json` (`experiments/ensemble.py`):
`p = a p_newton + (1 - a) p_boosting`, with `p_newton` from section 1 (final model)
and `p_boosting` from `logistic_boosting/04_final.json` (model A). In each fold, `a`
(grid 0, 0.05, ..., 1) and the threshold are chosen on the out-of-fold probabilities
of the other 4 folds, then F1 is measured on the remaining fold. Each component alone
is scored with the same protocol (`a` fixed to 1 or 0), so the rows are comparable.
Reference: logistic boosting alone.

| Model | F1 | Δ F1 | folds better | `a` per fold | F1 + rule |
|---|---|---|---|---|---|
| Newton (interactions) alone | 0.4336 ± 0.0015 | -0.0001 | 2/5 | 1 | 0.4390 ± 0.0018 |
| Logistic boosting alone | 0.4338 ± 0.0021 | reference | – | 0 | 0.4399 ± 0.0026 |
| **Ensemble** | **0.4356 ± 0.0020** | **+0.0018** | **5/5** | 0.4, 0.4, 0.35, 0.35, 0.35 | 0.4414 ± 0.0021 |

- The ensemble is better in all 5 folds, and the weight chosen on the other folds is
  stable (0.35 to 0.40 for Newton).
- Final (`final` in the JSON): `a = 0.35`, threshold 0.2062 (0.20618416941358658),
  out-of-fold F1 0.4360, 13.60% predicted positive:
  `submissions/ensemble_newton_boosting.csv`, the submitted file, reproduced by
  `run.py`. With the rule: `submissions/ensemble_newton_boosting_rule.csv`
  (`HAREHAB1` was asked in 238 test rows; 13.67% predicted positive).
- The out-of-fold and test probabilities of both components are cached in
  `ensemble/scores_*.npz` (see [README.md](README.md)): with them `ensemble.py` writes
  both files in a few seconds. A regeneration from the cache gave a file identical
  byte for byte to the first one (checked once by the group).

## 4. Feature screening: hand-made features

`reg_logistic_newton/extra/feature_screening.json` (`experiments/feature_screening.py`).
Groups of features built from the raw codes (indicator 1 = yes; missing, "don't know",
"refused" and question not asked count as no), added as extra raw columns to the
pipeline of the best row of `03_interactions` (degree 2, log, top 20 interactions).

| Group | Columns |
|---|---|
| comorbidity count (number) | number of conditions among diabetes, hypertension, high cholesterol, stroke, kidney disease, COPD, arthritis |
| comorbidity count (one-hot) | the same count as indicators >= 2, >= 3, >= 4 |
| risk factor count (one-hot) | age at risk (men 55+, women 65+), current smoker, hypertension, high cholesterol, diabetes, obesity (BMI >= 30), inactivity: indicators >= 2 ... >= 5 |
| functional limitation (one-hot) | `DIFFWALK`, `USEEQUIP`, `QLACTLM2`, `DIFFDRES`, `DIFFALON`, unable to work, >= 14 days of poor physical health: indicators >= 1 ... >= 4 |
| health care contact (one-hot) | check-up in the last year, pneumonia vaccine, flu vaccine, daily aspirin, blood pressure medication: indicators >= 1 ... >= 4 |
| clinical interactions | 14 products: age x sex, age x diabetes, age x hypertension, age x cholesterol, age x current / ever smoker, age x stroke, sex x ever smoker, sex x diabetes, diabetes x hypertension, hypertension x cholesterol, obesity x diabetes, diabetes x kidney, age x comorbidity count |

**Level 1, score test** (no refit). The current model is fitted once; at its minimum
the residuals `y - p` are uncorrelated with every column of the model, and the score
(Lagrange multiplier) test measures how much of what is left a new group explains:

    g = X_new^T (p - y) / N                 gradient of the loss in the new weights, at 0
    I = H_nn - H_nb H_bb^-1 H_bn            curvature left after the columns already in
    chi2 = N g^T I^-1 g  ~  chi2(df)        if the group adds nothing

`chi2 / (2N)` estimates the decrease of the mean logistic loss after a refit
("Δ loss"). Reference of scale: the same test for the top 20 interactions against the
model without them (`02b_transforms`), which are worth +0.0015 in F1. A group passes
with p < 0.05 / 6 (Bonferroni over the 6 groups). "In top 20": the group's columns
enter the 20 columns most correlated with the label, so they change the interactions
chosen by `interaction_top`.

| Group | df | chi2 | p | Δ loss | in top 20 | passes |
|---|---|---|---|---|---|---|
| reference: top 20 interactions | 190 | 1152.9 | 7e-129 | 1.8e-3 | – | – |
| comorbidity count (number) | 1 | 0.0 | 0.82 | 6e-8 | yes | no |
| comorbidity count (one-hot) | 3 | 17.7 | 6e-4 | 2.7e-5 | yes | yes |
| risk factor count (one-hot) | 4 | 18.2 | 1e-3 | 2.8e-5 | yes | yes |
| functional limitation (one-hot) | 4 | 21.8 | 3e-4 | 3.3e-5 | yes | yes |
| health care contact (one-hot) | 4 | 6.7 | 0.15 | 1.0e-5 | yes | no |
| clinical interactions | 14 | 54.8 | 1e-6 | 8.4e-5 | no | yes |

- The count as a number gives chi2 = 0: it is a sum of columns the model already has,
  and a logistic regression already uses weighted sums. A new feature can help a
  linear model only if it is non-linear in the existing columns.
- Among the clinical interactions only age x sex adds much (chi2 34.6), then
  age x current smoker (10.1) and age x ever smoker (8.7); 8 of the 14 products have
  chi2 < 0.1.
- The groups that pass are 20 to 65 times smaller in Δ loss than the reference: with
  328,135 rows even tiny effects are significant.

**Level 2, CV** of the groups that pass, on the same folds as the reference row
(`03_interactions`, top 20: 0.4339 ± 0.0016).

| Group | F1 | Δ F1 | folds better | columns |
|---|---|---|---|---|
| comorbidity count (one-hot) | 0.4330 ± 0.0019 | -0.0009 | 2/5 | 620 |
| risk factor count (one-hot) | 0.4335 ± 0.0021 | -0.0004 | 2/5 | 621 |
| functional limitation (one-hot) | 0.4328 ± 0.0017 | -0.0011 | 2/5 | 621 |
| clinical interactions | 0.4327 ± 0.0009 | -0.0012 | 1/5 | 645 |
| all the groups that pass | 0.4326 ± 0.0018 | -0.0013 | 1/5 | 656 |

No group improves F1: a decrease of the loss of order 3e-5 does not move the ranking
of the rows around the threshold. The clinical interactions do not enter the top 20
and do not improve either, so the change of the automatic interactions does not
explain the result. Level 3 (boosting, ensemble) was not run.

## Summary

| Experiment | Result |
|---|---|
| Newton: powers, sqrt / log | no gain (at most +0.0001) |
| Newton: top 20 interactions | +0.0015, 5/5 folds |
| Logistic boosting: top 20 interactions | -0.0005, 1/5 folds |
| Hand-made clinical features | -0.0004 to -0.0013 |
| Ensemble Newton (interactions) + logistic boosting | +0.0018 over logistic boosting, 5/5 folds: the submitted model |
