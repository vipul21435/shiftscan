from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
from scipy import special, stats
from scipy.spatial.distance import jensenshannon

from shiftscan.simple import (
    cliffs_delta,
    compare_labels,
    compare_lengths,
    jensen_shannon,
    kolmogorov_sf,
    ks_two_sample,
    total_variation,
    unigram_counts,
)


@pytest.mark.parametrize("value", [0.0, 0.1, 0.2, 0.25, 0.5, 0.8, 1.0, 1.36, 2.0, 3.5])
def test_kolmogorov_survival_matches_scipy(value):
    assert kolmogorov_sf(value) == pytest.approx(float(special.kolmogorov(value)), abs=1e-9)


def test_ks_statistic_matches_scipy_with_ties(rng):
    a = rng.poisson(20, size=400)
    b = rng.poisson(22, size=350)
    ours = ks_two_sample(a, b)
    reference = stats.ks_2samp(a, b, method="asymp")
    assert ours.statistic == pytest.approx(reference.statistic, abs=1e-12)
    assert ours.p_value == pytest.approx(reference.pvalue, rel=0.25, abs=0.01)


def test_ks_p_value_close_to_scipy_on_large_continuous_samples(rng):
    a = rng.normal(size=3000)
    b = rng.normal(loc=0.05, size=3000)
    ours = ks_two_sample(a, b)
    reference = stats.ks_2samp(a, b, method="asymp")
    assert ours.statistic == pytest.approx(reference.statistic, abs=1e-12)
    assert ours.p_value == pytest.approx(reference.pvalue, abs=0.02)


def test_ks_by_hand():
    # F_a jumps to 1 at 2; F_b is 0 until 3: the gap at 2 is 1
    assert ks_two_sample([1, 2], [3, 4]).statistic == pytest.approx(1.0)
    assert ks_two_sample([1, 2, 3, 4], [1, 2, 3, 4]).statistic == pytest.approx(0.0)
    with pytest.raises(ValueError):
        ks_two_sample([], [1])


def test_cliffs_delta_by_hand():
    # pairs (a, b): b larger in 3 of 4, smaller in 1 of 4
    assert cliffs_delta([1, 3], [2, 4]) == pytest.approx(0.5)
    assert cliffs_delta([1, 2], [1, 2]) == pytest.approx(0.0)


def test_compare_lengths_reports_both_units():
    out = compare_lengths(["a b", "a b c"], ["a b c d e", "a b c d e f"])
    assert out["tokens"]["mean_a"] == pytest.approx(2.5)
    assert out["tokens"]["median_b"] == pytest.approx(5.5)
    assert out["chars"]["mean_a"] == pytest.approx(4.0)
    assert out["tokens"]["ks_statistic"] == pytest.approx(1.0)
    assert out["tokens"]["cliffs_delta"] == pytest.approx(1.0)


def test_g_test_matches_scipy(rng):
    labels_a = list(rng.choice(["x", "y", "z"], size=300, p=[0.5, 0.3, 0.2]))
    labels_b = list(rng.choice(["x", "y", "z"], size=260, p=[0.4, 0.3, 0.3]))
    out = compare_labels(labels_a, labels_b)
    table = np.array([out["counts_a"], out["counts_b"]])
    g, p, dof, _ = stats.chi2_contingency(table, correction=False, lambda_="log-likelihood")
    assert out["g_statistic"] == pytest.approx(g)
    assert out["g_p"] == pytest.approx(p)
    assert out["dof"] == dof == 2
    shares = table / table.sum(axis=1, keepdims=True)
    assert out["total_variation"] == pytest.approx(0.5 * np.abs(shares[0] - shares[1]).sum())


def test_labels_with_a_category_missing_on_one_side_and_single_category():
    out = compare_labels(["x", "x", "y"], ["x", "x", "x"])
    assert out["categories"] == ["x", "y"]
    assert out["counts_b"] == [3, 0]
    assert out["total_variation"] == pytest.approx(1 / 3)
    assert math.isfinite(out["g_statistic"])
    same = compare_labels(["x", "x"], ["x"])
    assert same["dof"] == 0 and same["g_p"] == pytest.approx(1.0)
    assert same["g_statistic"] == pytest.approx(0.0)


def test_total_variation_bounds():
    assert total_variation({"a": 2}, {"b": 5}) == pytest.approx(1.0)
    assert total_variation({"a": 1, "b": 1}, {"a": 4, "b": 4}) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        total_variation({}, {"a": 1})


def test_jensen_shannon_matches_scipy(rng):
    vocabulary = [f"w{i}" for i in range(40)]
    counts_a = Counter(dict(zip(vocabulary, rng.poisson(5, size=40) + 1, strict=True)))
    counts_b = Counter(dict(zip(vocabulary, rng.poisson(5, size=40), strict=True)))
    p = np.array([counts_a[w] for w in vocabulary], dtype=float)
    q = np.array([counts_b[w] for w in vocabulary], dtype=float)
    reference = jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2
    assert jensen_shannon(counts_a, counts_b) == pytest.approx(reference, abs=1e-12)


def test_jensen_shannon_extremes():
    assert jensen_shannon({"a": 3}, {"b": 2}) == pytest.approx(1.0)
    assert jensen_shannon({"a": 3, "b": 1}, {"a": 6, "b": 2}) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        jensen_shannon({"a": 1}, {})


def test_unigram_counts_with_bigrams():
    assert unigram_counts(["The cat. The dog!"]) == Counter({"the": 2, "cat": 1, "dog": 1})
    assert unigram_counts(["a b a"], ngram=2) == Counter({"a": 2, "b": 1, "a b": 1, "b a": 1})
