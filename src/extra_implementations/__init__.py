"""Methods beyond the 6 of implementations.py, NumPy only.

Optimizers of the regularized logistic objective of reg_logistic_regression (same
loss and penalty convention, logistic_objective.py), y in {0, 1}, return (w, loss):
    reg_logistic_adaptive_gd   gradient descent, Barzilai-Borwein step + backtracking
    reg_logistic_newton        damped Newton / IRLS
    reg_logistic_sophia        Sophia, clipped diagonal Newton on minibatches
    reg_logistic_unixgrad      UniXGrad, adaptive accelerated extra-gradient
Tree ensembles (trees.py), y in {0, 1}, return (model, loss) and have a
predict_proba function:
    random_forest              random forest of histogram trees
    gradient_boosting          gradient boosting of histogram trees, log-loss
    logistic_boosting          hybrid: Newton logistic regression + gradient
                               boosting on its residuals (same (y, tx, lambda_,
                               initial_w, max_iters) arguments as the optimizers)

src/models.py gives them the same fit / predict_scores interface as the 6 methods.
"""

from src.extra_implementations.adaptive_gd import reg_logistic_adaptive_gd
from src.extra_implementations.gradient_boosting import (
    gradient_boosting,
    gradient_boosting_predict_proba,
)
from src.extra_implementations.logistic_boosting import (
    logistic_boosting,
    logistic_boosting_predict_proba,
)
from src.extra_implementations.newton import reg_logistic_newton
from src.extra_implementations.random_forest import (
    random_forest,
    random_forest_predict_proba,
)
from src.extra_implementations.sophia import reg_logistic_sophia
from src.extra_implementations.unixgrad import reg_logistic_unixgrad
