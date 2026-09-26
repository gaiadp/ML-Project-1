"""The 6 required ML methods (Step 2 of the project description).

Graders import these functions directly from this file, so:
- do NOT rename this file or change the function signatures;
- every function returns (w, loss): the LAST weight vector and its loss;
- losses of regularized methods must NOT include the penalty term;
- MSE uses the 1/2 factor; vectors are 1D arrays of shape (D,);
- only NumPy is allowed (numpy.linalg.lstsq is forbidden).
"""

import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    raise NotImplementedError


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1)."""
    raise NotImplementedError


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    raise NotImplementedError


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    raise NotImplementedError


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1})."""
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using GD (y in {0, 1}, penalty lambda_ * ||w||^2)."""
    raise NotImplementedError
