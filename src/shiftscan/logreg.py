"""L2-regularised logistic regression on dense or sparse inputs.

The objective is the one scikit-learn uses with ``C = 1 / l2``::

    sum_i log(1 + exp(-s_i * (x_i . w + b))) + (l2 / 2) * ||w||^2,   s_i = 2 * y_i - 1

with an unpenalised intercept ``b``. It is minimised with L-BFGS using the exact
gradient, which needs only ``X @ w`` and ``X.T @ r`` and so works on scipy
sparse matrices without densifying them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, sparse
from scipy.special import expit

Matrix = np.ndarray | sparse.spmatrix


@dataclass
class LogisticModel:
    coef: np.ndarray
    intercept: float
    n_iter: int
    converged: bool

    def decision_function(self, features: Matrix) -> np.ndarray:
        return np.asarray(features @ self.coef).ravel() + self.intercept

    def predict_proba(self, features: Matrix) -> np.ndarray:
        """Probability of class 1 for every row."""
        return expit(self.decision_function(features))


def _loss_and_grad(
    params: np.ndarray, features: Matrix, signs: np.ndarray, l2: float
) -> tuple[float, np.ndarray]:
    weights, bias = params[:-1], params[-1]
    margins = signs * (np.asarray(features @ weights).ravel() + bias)
    loss = float(np.logaddexp(0.0, -margins).sum() + 0.5 * l2 * weights @ weights)
    residual = -signs * expit(-margins)
    grad = np.empty_like(params)
    grad[:-1] = np.asarray(features.T @ residual).ravel() + l2 * weights
    grad[-1] = residual.sum()
    return loss, grad


def fit_logistic(
    features: Matrix,
    labels: np.ndarray,
    l2: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> LogisticModel:
    """Fit the model on rows ``features`` with 0/1 ``labels``."""
    labels = np.asarray(labels)
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels have different numbers of rows")
    if l2 <= 0:
        raise ValueError("l2 must be positive")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    signs = 2.0 * labels.astype(np.float64) - 1.0
    start = np.zeros(features.shape[1] + 1)
    result = optimize.minimize(
        _loss_and_grad,
        start,
        args=(features, signs, l2),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": tol, "ftol": 1e-12, "maxcor": 10},
    )
    return LogisticModel(
        coef=result.x[:-1].copy(),
        intercept=float(result.x[-1]),
        n_iter=int(result.nit),
        converged=bool(result.success),
    )
