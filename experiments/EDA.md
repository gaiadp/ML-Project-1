# Exploratory data analysis: `experiments/eda.py`

The report has two pages, so it only states the conclusions of the exploratory data
analysis (EDA). This file gives the details: what each of the nine analyses A1–A9
measures and why, where its output is, the main numbers, and which step of the
preprocessing pipeline or which hand-written column list each analysis led to.
Every number below is copied from the text tables in `build/eda/` (the file is cited
each time) or from the files of `src/` and `results/` named next to it.

Contents:

1. [Terms and abbreviations](#1-terms-and-abbreviations)
2. [Running the EDA](#2-running-the-eda)
3. [The nine analyses](#3-the-nine-analyses)
4. [Choice of the columns](#4-choice-of-the-columns)
5. [Limitations](#5-limitations)
6. [HAREHAB1: the label leakage column](#6-harehab1-the-label-leakage-column)

## 1. Terms and abbreviations

| Term | Meaning |
|---|---|
| BRFSS | Behavioral Risk Factor Surveillance System, the 2015 telephone health survey of the US Centers for Disease Control and Prevention (CDC) that the data comes from |
| MICHD | the label: the person reported a myocardial infarction (heart attack) or coronary heart disease; "MICHD prevalence" of a group = share of its rows with label 1 |
| codebook | the 2015 BRFSS codebook, which gives the meaning of every column and of its codes |
| special codes | answer codes that are not values, e.g. 7 / 77 = "don't know", 9 / 99 = "refused", 88 = "none" |
| NaN | missing value (not a number) |
| optional module | a block of questions asked only in some states or only to a subgroup, hence mostly NaN |
| skip logic | a follow-up question is not asked when the gate question rules it out (no stroke → no question on stroke rehabilitation) |
| MI | mutual information between a column and MICHD, in millibits (mbit); NaN counts as one more value. 0 = independent; the entropy of the label itself is 431 mbit (comment above `NOT_SELECTED` in `src/feature_types.py`) |
| \|r\| | absolute Pearson correlation with the 0/1 label, on the rows where the column is answered |
| log-odds | log(p / (1 − p)) of the MICHD prevalence p of a group; logistic regression is linear on this scale |
| R² line / parabola | weighted R² of a line / parabola fitted to the log-odds per value (or per decile); tells if one linear weight per column fits |
| U(b\|a) | "uncertainty coefficient" MI(a, b) / H(b): 1 when column b is a function of column a (H = entropy) |
| TVD | total variation distance between two distributions: 0 = identical, 1 = no overlap |
| F1 | the metric of the challenge, harmonic mean of precision and recall on the class MICHD |
| CV | cross-validation (5 folds, seed 1, in all our experiments) |
| FIPS code | the numeric code of a US state used in `_STATE` |
| V0, V1 | pipeline configurations of `src/pipeline.py`: `V0_CONFIG` (baseline) and `V1_CONFIG` (full pipeline + feature selection, the one of the final model) |

Pipeline steps referred to below (names B1–B13 from the docstring of
`src/pipeline.py`; B2 has three parts, called B2, B2b, B2c in the comments of
`src/feature_types.py`):

| Step | Config key | What it does |
|---|---|---|
| B1 | – (always on) | `codes_to_nan`: special codes → NaN ("don't know", "refused") or 0 ("none") |
| B2 | `domain_drop` | drop ids, interview process, phone screening, survey weights, child module (`DOMAIN_DROP`) |
| B2b | `drop_redundant` | drop the duplicates of other columns (`REDUNDANT`) |
| B2c | `select_features` | drop the columns not selected from the EDA (`NOT_SELECTED`, rules R1–R4) |
| B3 | `convert_units` | one numeric scale per column; raw mixed-units columns dropped (`CONVERTED`) |
| B4 | `skip_logic_fill` | NaN of questions not asked because of the skip logic → "no" / 0 |
| B5 | `max_missing_frac`, `keep_informative` | drop columns with too many NaN, except those whose missingness is informative |
| B6 | `missing_indicators` | one is-missing column per group of columns missing together |
| B7 | `impute` | median, or the mode for binary / nominal columns |
| B8 | `recode_binary` | 1/2 → 1/0 |
| B9 | `one_hot`, `one_hot_ordinal` | nominal columns (and the ordinal ones of `ONE_HOT_ORDINAL`) → one column per value with ≥ 500 training rows |
| B10 | `binning` | count columns of `BINNED` (only `PHYSHLTH`) → bin index, then one-hot |
| B11 | `poly_degree`, `interaction_top`, `interaction_pairs` | powers and products of the standardized columns (off by default, tuned per method) |
| B12 | – | standardization with the training mean / standard deviation |
| B13 | – | bias column |

## 2. Running the EDA

From the repository root:

```bash
python experiments/eda.py           # all the analyses, A1 to A9
python experiments/eda.py A6 A7     # only some of them (any subset of A1 ... A9)
```

An unknown name stops the script with the list of the valid ones. The data is read
with `src.data.load_data`: from the cache `build/data.npz`, which is built first from
`dataset/x_train.csv`, `dataset/y_train.csv`, `dataset/x_test.csv` if it does not exist.
Running time: not measured.

What each analysis sees:

- A1–A4 work on the raw data and on the output of `codes_to_nan` (B1). A1 and A2 look at
  all 321 columns, A4 at the 280 columns that are not in `DOMAIN_DROP` (its leakage
  scan looks at all of them).
- A5–A9 look at the values the model sees: after `codes_to_nan`, `convert_units` and
  `fill_skip_logic` (B1, B3, B4, steps that use no labels), on the 266 feature columns,
  i.e. without `DOMAIN_DROP` (41 columns) and without the raw columns of `CONVERTED`
  (14). The duplicates of `REDUNDANT` are still included, so A7 can check them.

### Outputs

| Analysis | Function | Text table | Figures |
|---|---|---|---|
| A1 structural overview | `structural_overview` | `build/eda/overview.txt` | – |
| A2 feature types | `feature_types_table` | `build/eda/feature_types.txt` | – |
| A3 special codes | `special_codes_audit` | `build/eda/special_codes.txt` | – |
| A4 missing values | `missingness_analysis` | `build/eda/missingness.txt` | `report/figures/missing_balance.png` |
| A5 target | `target_overview` | `build/eda/target.txt` | – |
| A6 feature–target relation | `target_relation` | `build/eda/target_relation.txt` | `build/eda/figures/mi_ranking.png`, `mi_vs_corr.png`, `log_odds_top24.png`; `report/figures/log_odds.png` |
| A7 redundancy | `redundancy` | `build/eda/redundancy.txt` | `build/eda/figures/corr_top.png` |
| A8 train vs test | `train_test_shift` | `build/eda/shift.txt` | – |
| A9 ranges and outliers | `ranges_and_outliers` | `build/eda/outliers.txt` | – |

- `build/` is in `.gitignore`: the text tables and the exploratory figures of
  `build/eda/figures/` are not in the repository. Run the script to get them.
- The two figures `report/figures/missing_balance.png` (A4) and
  `report/figures/log_odds.png` (A6) are tracked by git but not used by the final
  report, which has no room for them (its only figure, `report/figures/ablation.pdf`,
  is written by `report/make_figures.py`). **Running A4 or A6 rewrites them**: after a
  run, `git status` shows them as modified; commit them only if the change is wanted.
- The exploratory figures: `mi_ranking.png` (the 24 columns with the highest MI),
  `mi_vs_corr.png` (MI against |r|: columns far above the bulk depend on MICHD in a
  non-linear way or through their NaN), `log_odds_top24.png` (log-odds of MICHD per
  value or decile of the 24 columns with the highest MI, NaN group drawn apart),
  `corr_top.png` (Pearson correlations among the same 24 columns).
- The figure `log_odds.png` is the same log-odds grid for 16 columns
  (`REPORT_FEATURES`): the highest MI, skipping the columns of `REDUNDANT` and any
  column with |r| > 0.7 with one already chosen (one panel per quantity: one age
  variable, not four).
- A7 and A8 draw random rows with a fixed seed (`np.random.default_rng(0)`).
- The helper functions (mutual information, discretization, U, pairwise correlation,
  best-rule F1, log-odds fit, TVD) are tested in `tests/test_eda.py` on small fake data.

## 3. The nine analyses

Each analysis below: what it measures, why, the key results (with the file they come
from) and what it was used for in `src/pipeline.py` or `src/feature_types.py`.

### A1 – Structural overview

**What.** Shapes, labels and ids; constant and near-constant columns (one value or NaN
in more than 99% of the rows), on the raw data and after `codes_to_nan`; MICHD
prevalence per value (or per decile) of the columns dropped for domain reasons
(`DOMAIN_DROP`) and of six columns whose fate the overview alone does not decide
(`DOMAIN_UNDECIDED` in `eda.py`: `_STATE`, `NUMADULT`, `NUMMEN`, `NUMWOMEN`, `HHADULT`,
`MSCODE`).

**Why.** Check the data before anything else, and see how much signal the domain drop
gives up: a dropped column that separates MICHD well needs a reason.

**Key results** (`build/eda/overview.txt`):

- Training set 328,135 rows × 321 columns, test set 109,379 × 321; 8.83% positives;
  ids unique in both sets, no id in both.
- After `codes_to_nan`: no column entirely NaN or constant, 6 columns with one value +
  NaN, 2 near-constant (`_FRT16`, `_VEG23`, the data-quality flags), 28 "mostly NaN".
  26 flagged columns are not in `DOMAIN_DROP`: all are optional-module questions
  (asthma history `AS…`, visual impairment `VI…`, prostate cancer screening, `STREHAB1`,
  …), left to the feature selection (section 4).
- The phone-screening columns split the people into two samples: landline interviews
  (57.5%, MICHD 10.7%) and cell-phone interviews (42.5%, MICHD 6.2%), e.g. `CTELENUM`,
  `STATERES`. This is a property of the sampling frame, not of the person.
- Dropped columns can still carry univariate signal: the survey weight `_LLCPWT` goes
  from 5.1% to 11.5% MICHD across its deciles, the id `SEQNO` from 5.9% to 11.1%. The
  weights are built from age, sex and race, which stay as features (docstring of
  `domain_columns` in `eda.py`); ids and dates are not properties of the person.
- `HAREHAB1`: 99.8% NaN, MICHD 100.0% on its 758 answered rows (section 6).
- Undecided columns: `_STATE` from 5.3% (FIPS 49) to 14.1% (FIPS 5) over 53 states;
  `NUMADULT` 13.3% for one adult in the household down to 6.7% for 4 or 5;
  `MSCODE` 6.3% (NaN) to 11.9%.

**Used for.** `DOMAIN_DROP` (B2) and `LABEL_LEAKAGE`; rule R3 of the feature selection
(the prevalence by value of `NUMMEN` and `MSCODE`). `_STATE`, `NUMADULT`, `NUMWOMEN`
and `HHADULT` stayed as features.

### A2 – Feature types

**What.** The type chosen by hand from the codebook for each column (`FEATURE_TYPES`:
drop, binary, ordinal, nominal, count, continuous, mixed units), next to an automatic
guess from the values and the special codes, with the distinct values after
`codes_to_nan`.

**Why.** The type decides how the pipeline encodes a column; the automatic guess is a
check of the hand-made choice. A2 uses no labels.

**Key results** (`build/eda/feature_types.txt`):

- 41 drop, 103 binary, 54 ordinal, 39 nominal, 31 count, 34 continuous, 19 mixed units.
- The guess differs for 16 columns, all explained: count columns with few values
  guessed "categorical" (`NUMADULT`, `NUMMEN`, `NUMWOMEN`, `HHADULT`, `NUMPHON2`,
  `AVEDRNK2`, `MAXDRNKS`, `_MISFRTN`, `_MISVEGN`), the activity types `EXRACT11` /
  `EXRACT21` (75 / 76 codes, nominal) guessed "continuous", `HTIN4` and `DROCDY3_`
  (continuous with few distinct values) guessed "categorical".

**Used for.** `FEATURE_TYPES`, which drives B3 (which columns are mixed units), B7
(mode imputation of binary and nominal columns), B8 (binary recoding) and B9 (one-hot
of nominal columns).

### A3 – Special codes

**What.** Column by column, how many answers `codes_to_nan` turns into NaN and into 0,
and which code-like values (7, 8, 9, 77, 88, 98, 555, 999, …) it keeps although they
stand apart from the other answers (e.g. 8 after 1..4).

**Why.** The same number is a missing code in one column and a real answer in another
(`EMPLOY1` = 7 "retired", `_STATE` = 9 Connecticut): the rules are per column (`_RULES`
in `src/preprocessing.py`) and need a check.

**Key results** (`build/eda/special_codes.txt`):

- 237 columns are changed by `codes_to_nan`. Examples: `PHYSHLTH` 7,107 answers → NaN
  and 204,264 → 0 (88 = no bad day); `INCOME2` 56,123 → NaN (77 / 99).
- 35 code-like values are kept. Each is a real answer that needs care:
  - `CHECKUP1` = 8 (2,834 rows) and `EYEEXAM` = 8 (652 rows) mean "never": they come
    after the last ordered code;
  - `EXRACT21` = 88 (68,581 rows, 31.8% of the answers) means "no other activity";
  - `LONGWTCH` = 555 (2,148 rows) means "all my life";
  - `_STATE` = 8 (10,065 rows) is a FIPS code (Colorado), not a missing code.

**Used for.** B1 (the per-column rules), B3 (`CHECKUP1` and `EYEEXAM`: 8 → 5, after
"4 = longest ago"; `LONGWTCH`: 555 → the age), B4 (88 is the "no activity" value that
the skip logic fills in `EXRACT11` / `EXRACT21`), and the comments on special codes in
`FEATURE_TYPES`.

### A4 – Missing values

**What.** On the 280 columns not in `DOMAIN_DROP`: how many NaN per column (raw and
after `codes_to_nan`), a label leakage scan (any value, or NaN, with MICHD ≥ 50% on
≥ 30 rows, in all 321 columns), groups of columns missing together (correlation of
their NaN masks > 0.9), MICHD prevalence when a column is missing vs answered, NaN per
person.

**Why.** In BRFSS most columns are mostly NaN (optional modules, skip logic). The
question is which NaN carry information about MICHD, so that they are kept as
indicators rather than imputed away.

**Key results** (`build/eda/missingness.txt`):

- NaN fraction per column after `codes_to_nan`: 10 columns complete, 65 with ≤ 5%,
  75 with 5–50%, 36 with 50–90%, 68 with 90–99%, 26 with > 99%. Mean NaN fraction:
  45.6% raw, 47.1% after `codes_to_nan`.
- Leakage scan: one column only, `HAREHAB1` (answer 1: 300 rows, answer 2: 458 rows,
  both 100.0% MICHD). See section 6.
- 53 groups of columns missing together (among 258 columns with ≥ 500 missing and
  ≥ 500 answered rows). Examples:

  | Group | Columns | NaN | MICHD missing / answered |
  |---|---|---|---|
  | landline sample | `NUMADULT`, `NUMMEN`, `NUMWOMEN`, `NUMHHOL2`, `CPDEMO1`, `MSCODE` | 42.5–43.4% | 6.2% / 10.7% |
  | physical activity | 18 columns (`EXRACT11`, `PAFREQ1_`, `PA1MIN_`, …) | 33.1–36.3% | 11.5% / 7.5% |
  | diabetes module | 9 columns (`INSULIN`, `EYEEXAM`, `DOCTDIAB`, …) | 93.4–94.0% | 7.9% / 21.9% |
  | cognitive decline | `CDASSIST`, `CDDISCUS`, `CDSOCIAL`, `CDHOUSE` | 97.1% | 8.4% / 22.7% |

- Largest differences missing − answered: `STREHAB1` −26.9 points (99.8% NaN),
  `CVDSTRK3` +22.7 (31.5% MICHD on its 0.2% NaN), `CDHELP` −20.1, `DIABAGE2` −14.9.
- NaN per person (over the 280 columns): min 70, median 126, max 247; correlation with
  MICHD r = −0.031, but the prevalence is not monotone across the deciles (13.7% in
  the first, 6.6% in the sixth, 11.3% in the eighth).

**Used for.** B5 and B6: a column's missingness is "informative" when the MICHD
prevalence differs by ≥ 3 points between ≥ 500 missing and ≥ 500 answered rows
(`min_missing_diff=0.03`, `min_missing_rows=500` in `DEFAULT_CONFIG`); B6 adds one
indicator per group, with the same 0.9 threshold on the mask correlation
(`one_per_group` in `src/preprocessing.py`). B5 and B6 use the labels, so the pipeline
recomputes them on the training part of each fold. A4 also gave rules R2 and R3 of the
feature selection and `LABEL_LEAKAGE`. The figure `report/figures/missing_balance.png`
shows the histogram of the NaN fraction per column and the class balance.

### A5 – Target

**What.** Class balance; F1 of trivial rules; MICHD prevalence by age group and sex;
prevalence by state, raw and age-adjusted (each state's prevalence per age group,
averaged with the age distribution of the whole training set).

**Why.** Know the scale of the problem and the score that any model must beat.

**Key results** (`build/eda/target.txt`):

- 28,975 MICHD out of 328,135 rows (8.8%).
- F1 of trivial rules, measured on the rows they are fitted on: everyone MICHD 0.162,
  random 0.088, best rule on the age group 0.274, on age group × sex 0.294. The same
  rules in 5-fold CV are in `results/baselines.md` (0.1623, 0.2743, 0.2944, and 0.3088
  for a rule on `GENHLTH`).
- Prevalence grows from 0.5% (18–24 years) to 22.2% (80+); 11.2% for men, 7.1% for
  women; at 80+ 29.9% vs 18.3%.
- States: raw 5.3–14.1% (standard deviation 2.0 points), age-adjusted 6.3–13.9%
  (1.6 points): part of the difference between states is age, most of it remains.

**Used for.** The trivial baselines of `experiments/baselines.py`. The imbalance is
why every method tunes its decision threshold for F1 (with threshold 0.5 the final
model drops from 0.4338 to 0.2395, `results/ablation.md`). `_STATE` is kept as a
nominal column (one-hot, B9).

### A6 – Feature–target relation

**What.** For each of the 266 feature columns: MI with MICHD, |r|, shape of the
log-odds across values or deciles (R² of a line and of a parabola), a signal-to-noise
ratio of that shape, F1 of the best one-column rule, and a proposed encoding
(`encoding_hint` in `eda.py`).

**Why.** Rank the columns, find those with no signal, and see which ones a single
linear weight cannot describe (not monotone, curved).

**Key results** (`build/eda/target_relation.txt`):

- Highest MI: `GENHLTH` 43.56 mbit (|r| 0.249), `_AGEG5YR` 43.16, `_AGE80` 42.96,
  `MAXVO2_` 42.11, `FC60_` 42.11, `_AGE_G` 39.86, `EMPLOY1` 38.31, `BPMEDS` 36.14.
  General health, age and blood pressure lead.
- MI carried by the NaN pattern: `_HCVU651` has MI 26.75 but |r| 0.010 (35.9% NaN:
  the question is blank for people aged 65+, so its NaN encodes age); the same for
  `_FLSHOT6` (23.89 / 0.032) and `_PNEUMO2` (24.55 / 0.081).
- Proposals: 29 nominal columns to one-hot; 4 nominal with more than 10 frequent values
  (`EXRACT11`, `EXRACT21`, `_STATE`, `CRGVREL1`); 15 ordinal columns whose log-odds are
  not monotone; 31 curved numeric columns; 9 neither linear nor curved; 3 with no
  signal (`CDHELP`, `LSTBLDS3`, `NUMPHON2`).
- Not monotone ordinals: `_SMOKER3` (R² 0.32 for a line, 0.92 for a parabola),
  `LASTSMK2` (0.06 / 0.51), `_PACAT1` (0.47 / 0.99), `CHECKUP1` (0.79 / 1.00);
  `CHOLCHK` is monotone but curved (0.90 / 0.99).
- `STREHAB1` (after B4) and `CVDSTRK3` have almost the same MI: 19.28 and 19.71.

**Used for.**

- Rules R1, R2 and R4 of the feature selection (section 4).
- `ONE_HOT_ORDINAL` (B9 `one_hot_ordinal`): `_SMOKER3`, `LASTSMK2`, `_PACAT1`,
  `CHECKUP1` from the "not monotone" list, `CHOLCHK` for its curvature, `_ASTHMS1` so
  that its one-hot rebuilds the asthma columns of `REDUNDANT` (A7).
- B9 `one_hot_min_rows=500`: one column per value with ≥ 500 rows, the rare values of
  `EXRACT11` / `EXRACT21` are merged.
- The curved columns are the reason to try B11 (`poly_degree`); B11 is tuned per method
  in phases 02-03. It is off in model A (`logistic_boosting`), and on in the Newton
  component of the submitted ensemble (degree 2, log transforms, top 20 interactions:
  `results/reg_logistic_newton/extra/04_final_interactions.json`). The log transform
  is the option `extra_transforms` of B11 (see A9).

### A7 – Redundancy

**What.** On 100,000 random rows, every pair of the 266 feature columns, plus the 14 raw
columns of `CONVERTED`, compared before any encoding: |r| and U in both directions. A
greedy proposal of duplicates to drop: a column is a duplicate when a kept column with
almost the same NaN fraction (within 2 points) is the same quantity (|r| ≥ 0.999) or
determines it (U ≥ 0.95) and the pipeline encodes that kept column so that a linear
model can rebuild the duplicate exactly (one-hot or binary). Then a check of every
entry of `REDUNDANT` and `CONVERTED`.

**Why.** BRFSS has many calculated variables that regroup the raw answers (`_RFHYPE5`
from `BPHIGH4`, four race variables, …). Exact duplicates add nothing for a linear
model and make the fit less stable.

**Key results** (`build/eda/redundancy.txt`):

- The greedy proposal finds 30 duplicates; 21 more columns are "conditional
  duplicates" and are kept: groupings of a numeric column, i.e. step functions of it
  that a linear weight on that column cannot rebuild (`_AGEG5YR` from `_AGE80`,
  U = 0.987; `_RFHLTH` from `GENHLTH`; `_INCOMG` from `INCOME2`, …).
- 37 families of linked columns, 138 pairs with |r| or U ≥ 0.9.
- All 30 entries of `REDUNDANT` pass the check ("ok"), and all 14 of `CONVERTED`
  (U(converted | raw) = 1.000 for each).
- The hand-made list differs from the greedy proposal in three places, on purpose (see
  the comments of `REDUNDANT`): it keeps `HTM4` and drops `HTIN4` (the greedy does the
  opposite; `HTM4` also has the answers given in metric units: 3.5% NaN vs 3.9%), it
  keeps the raw question `TOLDHI2` and drops `_RFCHOL` (|r| = 1.000), and it attaches
  `SMOKE100` and `_RFSMOK3` to `_SMOKER3` instead of `LASTSMK2` (U(b|a) = 1.000).
- `_HCVU651`, `_FLSHOT6` and `_PNEUMO2` repeat `HLTHPLN1`, `FLUSHOT6` and `PNEUVAC3` for
  one age range only: their NaN fraction differs (e.g. 36.0% vs 0.4%), so they are kept.

**Used for.** `REDUNDANT` (B2b `drop_redundant`) and `CONVERTED` (B3: the raw
mixed-units column is dropped, its converted version kept); the choice of `CHOLCHK` and
`_ASTHMS1` in `ONE_HOT_ORDINAL`.

### A8 – Train vs test

**What.** For each feature column, the TVD between its distribution in the training set
and in the test set, next to the TVD between two random parts of the training set of
the same sizes (the noise level).

**Why.** Cross-validation on the training set estimates the test score only if the test
rows come from the same distribution.

**Key results** (`build/eda/shift.txt`): TVD train–test median 0.0011, maximum 0.0094
(`_STATE`); TVD between two random training parts median 0.0015, maximum 0.0099. No
column has a TVD above twice the noise and above 0.01.

**Used for.** No pipeline step. It supports using the CV score as the estimate of the
test score (for the inputs; the test labels are unknown).

### A9 – Ranges and outliers

**What.** For the 78 count, continuous and converted mixed-units columns: percentiles,
skewness, answers outside a plausible range (`PLAUSIBLE` in `eda.py`), MICHD above and
below the 99th percentile, and |r| of the raw values, of the values clipped at the
0.5th and 99.5th percentiles, and of log(1 + clipped value).

**Why.** Long tails and impossible values can dominate a linear model; a transform is
proposed when it raises |r| by at least 10% and by at least 0.005.

**Key results** (`build/eda/outliers.txt`):

- Few impossible answers: `_BMI5` 402 outside 12–80 (maximum 97.65), `WTKG3` 11,
  `_MINAC21` 16, `_MINAC11` 10, `HHADULT` 7 (maximum 60 adults), `CHILDREN` 6
  (maximum 52).
- Proposed: clipping + log for 22 columns, clipping alone for 7. The gains are large
  for the activity and alcohol quantities, e.g. `_DRNKWEK` |r| 0.024 → 0.087,
  `PA1MIN_` 0.003 → 0.070.

**Used for.** The log transform became the option `extra_transforms` of B11
(`log(1 + x - min)` of the continuous and count columns, `src/README.md`), tested in
phase 02b of `reg_logistic_newton`: 0.4323 -> 0.4324
(`results/reg_logistic_newton/02b_transforms.json`), kept only in the Newton component
of the submitted ensemble. The percentile clipping is not implemented, and the
out-of-range answers are left as they are (section 5).

### Summary

| Analysis | Main output used | Pipeline step / list |
|---|---|---|
| A1 | signal of the dropped columns, landline vs cell sample | `DOMAIN_DROP` (B2), `LABEL_LEAKAGE`, R3 |
| A2 | type of each column | `FEATURE_TYPES` → B3, B7, B8, B9 |
| A3 | special codes kept or changed | B1 (`_RULES`), B3, B4 |
| A4 | informative missingness, NaN groups, leakage scan | B5, B6, R2, R3, `LABEL_LEAKAGE` |
| A5 | imbalance, trivial rules | threshold tuning, `experiments/baselines.py` |
| A6 | MI ranking, shape of the log-odds | R1, R2, R4, `ONE_HOT_ORDINAL` (B9), B11 to try |
| A7 | duplicates | `REDUNDANT` (B2b), `CONVERTED` (B3), `ONE_HOT_ORDINAL` |
| A8 | no shift | none |
| A9 | outliers, proposed transforms | `extra_transforms` (log, B11); clipping not implemented |

## 4. Choice of the columns

The columns that reach the model are decided by four lists in `src/feature_types.py`.
**They are written by hand**, from the codebook and from the tables of `build/eda/`.
The code does not recompute them: `eda.py` reads `DOMAIN_DROP`, `REDUNDANT` and
`CONVERTED` only to report on them (A1, A7 check them), and nothing reads or writes
`NOT_SELECTED` except the pipeline. If the EDA changes, the lists must be updated by
hand. `tests/test_feature_types.py` checks that the lists are consistent with each
other (for example 236 candidates and 124 kept columns), not that they match the EDA
tables.

From 321 columns to the 124 of pipeline V1:

| Step | List | Removed | Left | Based on |
|---|---|---|---|---|
| all columns | – | – | 321 | – |
| domain drop (B2) | `DOMAIN_DROP` | 41 | 280 | codebook; A1 |
| raw mixed-units columns (B3) | `CONVERTED` | 14 | 266 | codebook; A7 check |
| duplicates (B2b) | `REDUNDANT` | 30 | 236 | A7 + codebook |
| feature selection (B2c, V1 only) | `NOT_SELECTED` | 112 | 124 | A1, A4, A6 (rules R1–R4) |

After the encoding (one-hot, missing indicators, bias) these 124 columns become the
334 inputs of the model (`results/ablation.md`, row "final").

### `DOMAIN_DROP`: not properties of the person

Interview ids and dates, interview process, phone screening, survey design and weights,
the random-child module (it describes a child, not the respondent), two data-quality
flags that are 99.98% one value, and `HAREHAB1`. Each entry has its reason in the dict.
The choice comes from the codebook; A1 shows what it costs in univariate signal.

### `CONVERTED`: one version of each answer

Fourteen answers exist twice: raw, with the unit in the first digit (e.g. `ALCDAY5`:
1xx per week, 2xx per month), and converted to one unit by the CDC (`DROCDY3_`). The
pipeline keeps the converted one. A7 checks that the converted column is a function of
the raw codes. The five mixed-units columns without a converted version are converted
by `convert_units` (B3).

### `REDUNDANT`: duplicates a linear model can rebuild

A column is listed only if (comments of `REDUNDANT`): A7 shows that a kept column
determines it (U ≥ 0.95) or is the same quantity (|r| ≥ 0.999); the codebook says it is
computed from that column or is the same question; the kept column is encoded so that a
linear model rebuilds the dropped one exactly, and both have the same NaN (within 2
points). Groupings of a numeric column (e.g. age classes of the age) are not listed:
they add a step function that the numeric column alone does not give.

### `NOT_SELECTED`: the feature selection of V1, rules R1–R4

Applied in this order to the 236 candidates; the reason of each column starts with its
rule.

| Rule | In words | Measure | Columns |
|---|---|---|---|
| R1 weak signal | the column tells almost nothing about MICHD on its own | MI < 1 mbit in A6 | 89 |
| R2 optional module | a module answered by fewer than 15% of the people: its columns share one NaN pattern, so most of their MI is "who was asked", the same for all of them; keep the column with the highest MI of each module | NaN > 85% in A6 (after B4), groups of A4 | 19 |
| R3 landline sample | questions asked only in landline interviews: their signal is the landline / cell split (A1, A4), which the kept landline columns already carry; their own answers carry almost nothing | A6 MI 4.55–4.94 mbit, |r| ≤ 0.026 vs 0.061–0.068 for the kept `NUMADULT`, `NUMWOMEN`, `CPDEMO1` | 3 |
| R4 copy of another column | `STREHAB1` (stroke rehabilitation, 515 answers): after the skip logic fills "no" for everyone without a stroke, it repeats `CVDSTRK3` | A6 MI 19.28 vs 19.71 | 1 |

**The 1 mbit threshold in simple words.** MI measures how much knowing the column
reduces the uncertainty on the label (431 mbit in total). For small effects (comment
above `NOT_SELECTED`), 1 mbit is about the information of a yes/no column that splits
the people into two halves whose MICHD prevalences differ by 2 points (e.g. 7.8% vs
9.8%). For comparison: `GENHLTH` 43.6 mbit, `SEX` 3.6 mbit. R1 removes mostly the
answers of optional modules (caregiving, visual impairment, asthma history, cancer
screening, social context, anxiety and depression) and details of the fruit,
vegetable and seat-belt answers.

R1 is not the "weak signal" hint of A6 (3 columns): that hint is a test on the shape
of the log-odds, R1 a threshold on MI.

Columns kept by R2, one per module: `CDASSIST` (cognitive decline), `DRADVISE`
(sodium), `SCNTPAID` (social context), `ARTTODAY` (arthritis management), `CVDASPRN`
(aspirin), `HADSIGM3` (colorectal cancer screening); `EYEEXAM`, `SHINGLE2`, `HPVADVC2`,
`PREGNANT` and `DIABAGE2` are the only column of their module above 85% NaN. V1 sets
`max_missing_frac=1.0`, so B5 does not drop them again (e.g. `CVDASPRN`: 95.7% NaN,
but |r| = 0.257 on its answers); B6 still adds their missing indicators.

Consistency with the tables (checked for this file against
`build/eda/target_relation.txt`): every R1 column has MI < 1 mbit, no column kept by V1
has MI < 1 mbit, and every R2 column has more than 85% NaN.

## 5. Limitations

- **The lists are chosen with the labels of the whole training set.** R1–R4 (MI,
  prevalence when missing), the duplicates of `REDUNDANT`, `ONE_HOT_ORDINAL` and the
  bins of `PHYSHLTH` (`BINNED`) were chosen looking at MICHD on all the training rows,
  then used as fixed lists inside every CV fold. The validation part of each fold has
  therefore already influenced the choice of columns, and the CV score of pipeline V1
  is slightly optimistic. This is stated in the comments of `src/feature_types.py`
  (above `NOT_SELECTED`) and in the docstring of `target_relation` in `eda.py`.
  `DOMAIN_DROP`, `FEATURE_TYPES` and `CONVERTED` come from the codebook and use no
  labels; the label-based steps B5, B6 and `interaction_top` are fitted inside each
  fold and do not have this problem.
- **How large the effect is.** The ablation (`results/ablation.md`, Newton logistic
  regression, lambda tuned again for each variant) removes the selection: without it
  F1 = 0.4321 ± 0.0025 against 0.4323 ± 0.0020 with it (−0.0002, better in 3 of 5
  folds); removing one rule at a time changes F1 by −0.0006 to +0.0004. The selection
  brings no measurable gain in CV, so the final score does not rest on it. The size of
  the optimism itself is not measured: that would need the selection to be redone
  inside every fold.
- **One column at a time.** A6 is univariate: a column with no signal alone could help
  together with others. The ablation row "selection without R1" (the 89 columns put
  back: 0.4317, −0.0006) does not show such a gain for a linear model.
- **Discretization.** MI and the log-odds shape of numeric columns are computed on 10
  quantile bins (A6), and A7 uses 100 bins on a sample of 100,000 rows: other choices
  would move the numbers near the thresholds.
- **Open points.** The percentile clipping proposed by A9 is not implemented (the log
  transform is, as `extra_transforms`); the out-of-range answers (e.g. 402 values of
  `_BMI5`) are kept. A6 proposes `PHYSHLTH`
  as "linear" (R² 0.91), while the pipeline bins it (B10); the ablation keeps the bins
  (without them −0.0010).

## 6. HAREHAB1: the label leakage column

**What it is.** `HAREHAB1` asks "after your heart attack, did you go to outpatient
rehabilitation?". It is asked only to people who reported a heart attack, which is part
of the definition of MICHD (docstring of `src/rules.py`). Whatever the answer, a row
where the question was asked is a positive: the column reveals the label instead of
predicting it.

**Where the EDA shows it.**

- A1 (`build/eda/overview.txt`, constant-column table): 99.8% NaN; the 758 answered
  rows have MICHD 100.0%, the other rows 8.6%.
- A4 (`build/eda/missingness.txt`, leakage scan): the only hit among all 321 columns,
  answer 1 (300 rows) and answer 2 (458 rows), both 100.0% MICHD.
- The EDA counts 758 rows because `codes_to_nan` turns the answers 7 ("don't know") and
  9 ("refused") into NaN. On the raw data the question was asked in 773 training rows
  (300 + 458 + 9 answers 7 + 6 answers 9), all positive: the number given in
  `src/rules.py` and `experiments/README.md`.

**What we decided.**

- The column is in `DOMAIN_DROP` and in `LABEL_LEAKAGE` (`src/feature_types.py`). Every
  pipeline configuration drops it, also when `domain_drop` is switched off for the
  ablation (`_candidate_columns` in `src/pipeline.py`). The model we submit (the
  ensemble of `experiments/ensemble.py`, reproduced by `run.py`) never sees it.
- `keep_leakage=True` keeps it, only to measure what the leakage is worth
  (`experiments/leakage.py`, model B).
- `src/rules.py` (`known_positives`, `apply_known_positives`) sets to 1, after the
  threshold, the rows where the question was asked (model C). It is a certain
  correction of labels for this survey, not part of the predictive model; `run.py` does
  not apply it. On the submitted ensemble it gives 0.4414 ± 0.0021
  (`results/ensemble/ensemble.json`).
- What it is worth (`results/ablation.md`, model table, same folds): final model
  0.4338 ± 0.0020; with `HAREHAB1` as a feature 0.4413 ± 0.0032 (+0.0075, better in
  5/5 folds); final model + rule 0.4400 ± 0.0026 (+0.0062, 5/5). In the report, the
  submitted ensemble (without `HAREHAB1`) is the result; B and C are mentioned as a
  measure of the leakage.

`STREHAB1` (rehabilitation after a stroke) is a different case: a stroke is not part of
MICHD, and the column has 35.7% MICHD among its answers, not 100% (A4). It is removed by
rule R4 as a copy of `CVDSTRK3`, not as leakage.
