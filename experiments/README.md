# Experiments

One folder per method of `implementations.py`, with the same four phases in each:

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `01_baseline.py` | pipeline v0 vs full pipeline vs v1 (full + feature selection from the EDA, `V1_CONFIG`), no expansion | – | `results/<method>/01_baseline.json` |
| `02_degree.py` | polynomial degree (`poly_degree`) | best of 01 | `results/<method>/02_degree.json` |
| `03_interactions.py` | interaction columns (`interaction_top`, `interaction_pairs`) | best of 02 | `results/<method>/03_interactions.json` |
| `04_final.py` | CV of the best setting, threshold on the out-of-fold scores, fit on all the training set | best of 03 (`FROM_PHASE`) | `results/<method>/04_final.json`, `submissions/<method>.csv` |

Run them in order, with the Run button or `python experiments/<method>/<phase>.py`.

The extra methods of `src/extra_implementations` (`reg_logistic_adaptive_gd`,
`reg_logistic_newton`, `reg_logistic_sophia`, `reg_logistic_unixgrad`, `random_forest`,
`gradient_boosting`, `logistic_boosting`) have only `01_baseline.py` and `04_final.py` (`FROM_PHASE =
"01_baseline"`): the four optimizers minimize the same loss as `reg_logistic_regression`
(phases 02 / 03 are the same as there), and the trees build interactions by themselves.
The trees run on pipeline v1 only, since they are slow (minutes per fit).

## One-off runs outside the phases

A grid that is not one of the phases (e.g. a finer lambda grid, more trees) goes to
`results/<method>/extra/<name>.json`: give `run_grid` a phase name starting with
`extra/`, so the files read by the next phases and by `baselines.py`
(`results/<method>/0*.json`) do not change:

```bash
python -c "
import runpy
from experiments.common import param_grid, run_grid
s = runpy.run_path('experiments/logistic_boosting/01_baseline.py')
run_grid(s['METHOD'], 'extra/01_more_trees', {'v1': s['V1_CONFIG']},
         param_grid(lambda_=[1e-4], max_iters=[20], n_trees=[400], max_depth=[3, 4],
                    learning_rate=[0.05]), 5, 1)
"
```

## Baseline table

`baselines.py` puts phase 01 of every method in one table (best F1 per pipeline, best
setting), next to trivial rules computed on the same folds (everyone MICHD, best rule
on the age group, on age group x sex, on GENHLTH). There is no separate tuning script:
the hyperparameters are the grids of the phases, the threshold is chosen in every fold.

```bash
python experiments/baselines.py            # from results/*/01_baseline.json (seconds)
python experiments/baselines.py --rerun    # run phase 01 of every method again first (hours)
python experiments/baselines.py --rerun ridge_regression least_squares   # only some methods
```

It writes `results/baselines.md` and `results/baselines.tex` (a LaTeX tabular for the
report). `--rerun` runs the `01_baseline.py` scripts as they are (all their pipelines).

## Changing an experiment

Every script starts with a settings block, the only part to edit:

```python
# ---- settings of the experiment ----
METHOD = "ridge_regression"
FROM_PHASE = "01_baseline"
DEGREES = (1, 2, 3)
FIT_GRID = param_grid(lambda_=np.logspace(-7, -1, 7))
K_FOLD = 5
SEED = 1
# -------------------------------------
```

- `FIT_GRID`: every combination of the given values is tried in CV, e.g.
  `param_grid(gamma=[0.1, 0.3], max_iters=[500], pos_weight=[3.2])`.
- Each phase starts from the **best pipeline** of `FROM_PHASE` (the full config is in
  the JSON, under `best.pipeline`). To skip a phase, point `FROM_PHASE` to an earlier
  one; to force a value in the final model, use `PIPELINE_OVERRIDES` / `FIT_OVERRIDES`
  in `04_final.py`.
- Keep `K_FOLD = 5` and `SEED = 1` in every phase and method: same folds = comparable
  F1 between methods.

The logic lives in `experiments/common.py` (grid, JSON, final fit) and in `src/`:
the scripts contain only settings.

## Notes

- F1 is measured on the validation part of each fold with the threshold chosen on the
  training part (`src/cross_validation.py`); `val-best` in the table is the optimistic
  F1 with the threshold chosen on the validation part itself.
- The polynomial and interaction columns are pipeline step B11 (`src/pipeline.py`):
  they are fitted inside each fold, and the interaction columns chosen by
  `interaction_top` use the labels of the training part only.
- On the full pipeline, ~317 columns become ~400 with degree 2 and ~480 with degree 3;
  `interaction_top = k` adds k (k - 1) / 2 columns. With 260k training rows per fold,
  ~600 columns already take ~1.3 GB per matrix.
