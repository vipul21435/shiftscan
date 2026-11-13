from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from shiftscan.c2st import auc_score, classifier_test, cross_fitted_scores, stratified_folds


def test_auc_matches_scikit_learn_with_ties(rng):
    labels = rng.integers(0, 2, size=200)
    scores = rng.integers(0, 6, size=200).astype(float)
    assert auc_score(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


def test_auc_by_hand():
    # pairs (neg, pos): (1,2) win, (1,3) win, (3,2) loss, (3,3) tie -> (2 + 0.5) / 4
    assert auc_score(np.array([0, 0, 1, 1]), np.array([1.0, 3.0, 2.0, 3.0])) == pytest.approx(0.625)


def test_auc_needs_both_sides():
    with pytest.raises(ValueError):
        auc_score(np.array([1, 1]), np.array([0.1, 0.2]))


def test_stratified_folds_are_balanced(rng):
    labels = np.array([0] * 23 + [1] * 17)
    folds = stratified_folds(labels, 5, rng)
    for fold in range(5):
        assert 4 <= int(((folds == fold) & (labels == 0)).sum()) <= 5
        assert 3 <= int(((folds == fold) & (labels == 1)).sum()) <= 4


def test_every_record_is_scored_out_of_fold(rng):
    features = rng.normal(size=(30, 3))
    labels = np.array([0] * 15 + [1] * 15)
    scores = cross_fitted_scores(features, labels, 3, rng)
    assert scores.shape == (30,)
    assert np.isfinite(scores).all()


def test_planted_shift_is_detected(rng):
    features = rng.normal(size=(200, 5))
    labels = np.array([0] * 100 + [1] * 100)
    features[labels == 1, 0] += 1.5
    result = classifier_test(features, labels, permutations=49, seed=1)
    assert result.auc > 0.8
    assert result.accuracy > 0.7
    assert result.binomial_p < 1e-6
    assert result.permutation_p == pytest.approx(1 / 50)
    assert result.null_auc_q95 < result.auc


def test_same_source_gives_chance_level(rng):
    features = rng.normal(size=(300, 5))
    labels = np.array([0] * 150 + [1] * 150)
    result = classifier_test(features, labels, permutations=0, seed=2)
    assert 0.38 < result.auc < 0.62
    assert result.permutation_p is None
    assert result.to_dict()["permutations"] == 0


def test_permutation_p_values_are_close_to_uniform_under_the_null():
    rng = np.random.default_rng(7)
    labels = np.array([0] * 20 + [1] * 20)
    p_values = []
    for rep in range(120):
        features = rng.normal(size=(40, 3))
        result = classifier_test(features, labels, folds=2, permutations=19, seed=rep)
        p_values.append(result.permutation_p)
    p_values = np.array(p_values)
    # uniform on {1/20, ..., 20/20}: mean 0.525, P(p <= 0.1) = 0.1
    assert abs(p_values.mean() - 0.525) < 0.08
    assert 0.02 <= (p_values <= 0.1).mean() <= 0.2
    assert 0.35 <= (p_values <= 0.5).mean() <= 0.65


def test_same_seed_same_result(rng):
    features = rng.normal(size=(60, 4))
    labels = np.array([0] * 30 + [1] * 30)
    first = classifier_test(features, labels, permutations=9, seed=5)
    second = classifier_test(features, labels, permutations=9, seed=5)
    assert first.auc == second.auc
    assert first.permutation_p == second.permutation_p
    assert np.array_equal(first.scores, second.scores)


def test_too_few_records_or_folds_is_an_error(rng):
    features = rng.normal(size=(6, 2))
    labels = np.array([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError):
        classifier_test(features, labels, folds=5)
    with pytest.raises(ValueError):
        classifier_test(features, labels, folds=1)
