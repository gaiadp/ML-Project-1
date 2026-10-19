| Trivial rule | F1 (5-fold CV) |
|---|---|
| everyone MICHD | 0.1623 ± 0.0000 |
| age group | 0.2743 ± 0.0019 |
| age group x sex | 0.2944 ± 0.0019 |
| general health | 0.3088 ± 0.0022 |

| Method | v0 | full | v1 | best setting |
|---|---|---|---|---|
| mean_squared_error_gd | 0.4184 ± 0.0024 | 0.4205 ± 0.0013 | 0.4211 ± 0.0018 | v1: gamma=0.1, max_iters=300 |
| mean_squared_error_sgd | 0.3996 ± 0.0111 | 0.4039 ± 0.0025 | 0.4084 ± 0.0028 | v1: gamma=0.0001, max_iters=50000 |
| least_squares | 0.4179 ± 0.0023 | 0.4209 ± 0.0016 | 0.4212 ± 0.0025 | v1: no parameters |
| ridge_regression | 0.4181 ± 0.0028 | 0.4211 ± 0.0013 | 0.4216 ± 0.0026 | v1: lambda_=0.001 |
| logistic_regression | 0.4270 ± 0.0029 | 0.4312 ± 0.0024 | 0.4311 ± 0.0017 | full: gamma=0.3, max_iters=500 |
| reg_logistic_regression | 0.4270 ± 0.0025 | 0.4313 ± 0.0025 | 0.4308 ± 0.0013 | full: lambda_=0.0001, gamma=0.3, max_iters=500 |
| reg_logistic_adaptive_gd | – | – | 0.4322 ± 0.0019 | v1: lambda_=0.0001, gamma=1, max_iters=300 |
| reg_logistic_newton | – | – | 0.4323 ± 0.0020 | v1: lambda_=0.0001, max_iters=20 |
| reg_logistic_sophia | – | – | 0.3928 ± 0.0050 | v1: lambda_=0.0001, gamma=0.02, max_iters=3000, batch_size=2048 |
| reg_logistic_unixgrad | – | – | 0.4318 ± 0.0016 | v1: lambda_=0.0001, max_iters=200, radius=5 |
| random_forest | – | – | 0.4158 ± 0.0018 | v1: n_trees=100, max_depth=14, min_samples_leaf=50 |
| gradient_boosting | – | – | 0.4298 ± 0.0024 | v1: n_trees=200, learning_rate=0.1, max_depth=3 |
| logistic_boosting | – | – | 0.4338 ± 0.0020 | v1: lambda_=0.0001, max_iters=20, n_trees=400, max_depth=3, learning_rate=0.05 |
