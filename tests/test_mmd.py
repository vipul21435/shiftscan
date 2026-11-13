from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from shiftscan.mmd import _mmd2_for_indicators, gram_matrix, mmd2_unbiased, mmd_test


def brute_force_mmd2(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    def k(u, v):
        return np.exp(-np.sum((u - v) ** 2) / (2 * bandwidth**2))

    m, n = len(x), len(y)
    xx = sum(k(x[i], x[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    yy = sum(k(y[i], y[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    xy = sum(k(x[i], y[j]) for i in range(m) for j in range(n)) / (m * n)
    return xx + yy - 2 * xy


def test_unbiased_mmd_matches_the_brute_force_formula(rng):
    x = rng.normal(size=(9, 3))
    y = rng.normal(loc=0.4, size=(7, 3))
    rows = np.vstack([x, y])
    labels = np.array([0] * 9 + [1] * 7)
    gram, bandwidth = gram_matrix(rows, kernel="rbf", bandwidth=1.3)
    assert bandwidth == pytest.approx(1.3)
    assert mmd2_unbiased(gram, labels) == pytest.approx(brute_force_mmd2(x, y, 1.3), abs=1e-12)


def test_median_bandwidth_is_the_median_pairwise_distance(rng):
    rows = rng.normal(size=(12, 2))
    _, bandwidth = gram_matrix(rows)
    distances = [np.linalg.norm(rows[i] - rows[j]) for i in range(12) for j in range(i + 1, 12)]
    assert bandwidth == pytest.approx(np.median(distances))


def test_sparse_rows_and_linear_kernel(rng):
    dense = rng.poisson(0.5, size=(10, 6)).astype(float)
    gram, bandwidth = gram_matrix(sparse.csr_matrix(dense), kernel="linear")
    assert bandwidth is None
    assert gram == pytest.approx(dense @ dense.T)
    with pytest.raises(ValueError):
        gram_matrix(dense, kernel="cubic")


def test_identical_rows_fall_back_to_unit_bandwidth():
    gram, bandwidth = gram_matrix(np.ones((4, 2)))
    assert bandwidth == pytest.approx(1.0)
    assert gram == pytest.approx(np.ones((4, 4)))


def test_vectorised_permutation_statistic_equals_the_direct_one(rng):
    rows = rng.normal(size=(14, 3))
    gram, _ = gram_matrix(rows)
    for _ in range(5):
        indicator = rng.permutation(np.array([1.0] * 6 + [0.0] * 8))
        direct = mmd2_unbiased(gram, (indicator == 0).astype(int))
        assert _mmd2_for_indicators(gram, indicator[None, :])[0] == pytest.approx(direct, abs=1e-12)


def test_planted_shift_has_a_small_p_value(rng):
    rows = np.vstack([rng.normal(size=(40, 4)), rng.normal(loc=1.0, size=(40, 4))])
    labels = np.array([0] * 40 + [1] * 40)
    gram, bandwidth = gram_matrix(rows)
    result = mmd_test(gram, labels, permutations=199, seed=1, bandwidth=bandwidth)
    assert result.p_value == pytest.approx(1 / 200)
    assert result.mmd2 > result.null_q95
    assert result.to_dict()["n_a"] == 40


def test_p_values_are_close_to_uniform_under_the_null():
    rng = np.random.default_rng(11)
    labels = np.array([0] * 15 + [1] * 15)
    p_values = []
    for rep in range(300):
        gram, _ = gram_matrix(rng.normal(size=(30, 3)))
        p_values.append(mmd_test(gram, labels, permutations=19, seed=rep).p_value)
    p_values = np.array(p_values)
    assert abs(p_values.mean() - 0.525) < 0.06
    assert 0.04 <= (p_values <= 0.1).mean() <= 0.17
    assert 0.4 <= (p_values <= 0.5).mean() <= 0.6


def test_seed_determinism_and_no_permutations(rng):
    gram, _ = gram_matrix(rng.normal(size=(20, 2)))
    labels = np.array([0] * 10 + [1] * 10)
    first = mmd_test(gram, labels, permutations=50, seed=3)
    second = mmd_test(gram, labels, permutations=50, seed=3)
    assert first.p_value == second.p_value
    assert mmd_test(gram, labels, permutations=0).p_value is None


def test_needs_two_records_per_side():
    with pytest.raises(ValueError):
        mmd2_unbiased(np.eye(3), np.array([0, 0, 1]))
