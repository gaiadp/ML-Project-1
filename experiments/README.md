# Experiments

One folder per method of `implementations.py`, with the same four phases in each:

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `01_baseline.py` | pipeline v0 vs full pipeline, no expansion | – | `results/<method>/01_baseline.json` |
| `02_degree.py` | polynomial degree (`poly_degree`) | best of 01 | `results/<method>/02_degree.json` |
| `03_interactions.py` | interaction columns (`interaction_top`, `interaction_pairs`) | best of 02 | `results/<method>/03_interactions.json` |
| `04_final.py` | CV of the best setting, threshold on the out-of-fold scores, fit on all the training set | best of 03 (`FROM_PHASE`) | `results/<method>/04_final.json`, `submissions/<method>.csv` |

Run them in order, with the Run button or `python experiments/<method>/<phase>.py`.

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
