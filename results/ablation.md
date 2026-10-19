Data preparation: reg_logistic_newton, lambda in [3e-05, 0.0001, 0.001] (best for every variant), one step switched off at a time

| Variant | F1 | Δ F1 | folds better | lambda | columns |
|---|---|---|---|---|---|
| final | 0.4323 ± 0.0020 | reference | – | 0.0001 | 334 |
| selection without R1 | 0.4317 ± 0.0021 | -0.0006 | 2/5 | 0.0001 | 450 |
| selection without R2 | 0.4327 ± 0.0019 | +0.0004 | 3/5 | 3e-05 | 362 |
| selection without R3 | 0.4320 ± 0.0013 | -0.0003 | 2/5 | 0.001 | 337 |
| selection without R4 | 0.4320 ± 0.0020 | -0.0003 | 2/5 | 0.0001 | 336 |
| no feature selection | 0.4321 ± 0.0025 | -0.0002 | 3/5 | 0.001 | 483 |
| NaN rule (B5) instead of the selection | 0.4316 ± 0.0035 | -0.0007 | 3/5 | 0.0001 | 415 |
| no domain drop (B2) | 0.4323 ± 0.0021 | -0.0000 | 2/5 | 0.001 | 378 |
| no duplicate drop (B2) | 0.4322 ± 0.0015 | -0.0001 | 2/5 | 0.001 | 372 |
| no unit conversion (B3) | 0.4317 ± 0.0023 | -0.0006 | 2/5 | 0.0001 | 348 |
| no skip-logic fill (B4) | 0.4321 ± 0.0017 | -0.0002 | 3/5 | 0.001 | 335 |
| no missing indicators (B6) | 0.4307 ± 0.0016 | -0.0015 | 1/5 | 0.0001 | 298 |
| no one-hot encoding (B9) | 0.4255 ± 0.0016 | -0.0068 | 0/5 | 0.001 | 161 |
| no one-hot of the ordinals (B9) | 0.4306 ± 0.0016 | -0.0017 | 0/5 | 0.0001 | 316 |
| no binning of PHYSHLTH (B10) | 0.4313 ± 0.0025 | -0.0010 | 1/5 | 0.001 | 330 |
| median imputation (B7) | 0.4323 ± 0.0020 | +0.0000 | 3/5 | 0.001 | 334 |

Model: logistic_boosting with the setting of 04_final, hyperparameters fixed

| Variant | F1 | Δ F1 | folds better |
|---|---|---|---|
| final model | 0.4338 ± 0.0020 | reference | – |
| no trees on the residuals (logistic regression, Newton) | 0.4323 ± 0.0020 | -0.0015 | 0/5 |
| no trees, plain GD instead of Newton | 0.4308 ± 0.0013 | -0.0029 | 0/5 |
| threshold 0.5 instead of the tuned threshold | 0.2395 ± 0.0061 | -0.1943 | 0/5 |
| with HAREHAB1 as a feature (label leakage) | 0.4413 ± 0.0032 | +0.0075 | 5/5 |
| final + known-positive rule (HAREHAB1 asked -> 1) | 0.4400 ± 0.0026 | +0.0062 | 5/5 |
