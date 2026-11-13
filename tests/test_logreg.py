from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize, sparse
from sklearn.linear_model import LogisticRegression

from shiftscan.logreg import _loss_and_grad, fit_logistic


def _problem(rng, n=120, d=6):
    features = rng.normal(size=(n, d))
    truth = rng.normal(size=d)
    labels = (features @ truth + 0.5 * rng.normal(size=n) > 0.2).astype(int)
    return features, labels


@pytest.mark.parametrize("l2", [0.1, 1.0, 10.0])
def test_matches_scikit_learn_on_a_small_dense_problem(rng, l2):
    features, labels = _problem(rng)
    ours = fit_logistic(features, labels, l2=l2, max_iter=2000, tol=1e-10)
    reference = LogisticRegression(C=1.0 / l2, tol=1e-10, max_iter=5000).fit(features, labels)
    assert ours.coef == pytest.approx(reference.coef_.ravel(), abs=2e-4)
    assert ours.intercept == pytest.approx(float(reference.intercept_[0]), abs=2e-4)
    assert ours.predict_proba(features) == pytest.approx(
        reference.predict_proba(features)[:, 1], abs=1e-4
    )


def test_sparse_and_dense_inputs_agree(rng):
    features, labels = _problem(rng, n=80, d=5)
    features[np.abs(features) < 0.8] = 0.0
    dense = fit_logistic(features, labels, max_iter=1000, tol=1e-10)
    as_sparse = fit_logistic(sparse.csr_matrix(features), labels, max_iter=1000, tol=1e-10)
    assert as_sparse.coef == pytest.approx(dense.coef, abs=1e-6)
    assert as_sparse.intercept == pytest.approx(dense.intercept, abs=1e-6)


def test_gradient_matches_finite_differences(rng):
    features, labels = _problem(rng, n=40, d=4)
    signs = 2.0 * labels - 1.0
    point = rng.normal(size=5)
    error = optimize.check_grad(
        lambda p: _loss_and_grad(p, features, signs, 0.7)[0],
        lambda p: _loss_and_grad(p, features, signs, 0.7)[1],
        point,
    )
    assert error < 1e-4


def test_separable_data_stays_finite_under_the_penalty():
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.array([0, 0, 1, 1])
    model = fit_logistic(features, labels, l2=1.0)
    assert np.isfinite(model.coef).all()
    assert model.coef[0] > 0
    assert (model.predict_proba(features) > 0.5).astype(int).tolist() == [0, 0, 1, 1]


def test_bad_inputs_are_rejected():
    features = np.zeros((3, 2))
    with pytest.raises(ValueError):
        fit_logistic(features, np.array([0, 1]))
    with pytest.raises(ValueError):
        fit_logistic(features, np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        fit_logistic(features, np.array([0, 1, 1]), l2=0.0)
