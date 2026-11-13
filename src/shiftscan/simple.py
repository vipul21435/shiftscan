"""Simple comparisons: lengths, label shares and unigram distributions."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .c2st import auc_score
from .features import tokenize, whitespace_tokens


def kolmogorov_sf(value: float) -> float:
    """Survival function of the Kolmogorov distribution, ``2 * sum (-1)^(k-1) exp(-2 k^2 x^2)``."""
    if value < 0.2:
        return 1.0
    total = 0.0
    for k in range(1, 201):
        term = math.exp(-2.0 * k * k * value * value)
        total += term if k % 2 else -term
        if term < 1e-18:
            break
    return min(max(2.0 * total, 0.0), 1.0)


@dataclass
class KSResult:
    statistic: float
    p_value: float
    n_a: int
    n_b: int


def ks_two_sample(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> KSResult:
    """Two-sample Kolmogorov-Smirnov test.

    The statistic is the largest gap between the two empirical distribution
    functions. The p-value is the asymptotic one, ``Q(sqrt(mn / (m + n)) * D)``; with
    many ties (lengths are whole numbers) it is conservative.
    """
    x = np.sort(np.asarray(a, dtype=float))
    y = np.sort(np.asarray(b, dtype=float))
    if x.size == 0 or y.size == 0:
        raise ValueError("both samples need at least one value")
    points = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, points, side="right") / x.size
    cdf_y = np.searchsorted(y, points, side="right") / y.size
    statistic = float(np.max(np.abs(cdf_x - cdf_y)))
    effective = x.size * y.size / (x.size + y.size)
    return KSResult(statistic, kolmogorov_sf(math.sqrt(effective) * statistic), x.size, y.size)


def cliffs_delta(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """``P(b > a) - P(b < a)`` over all pairs; 0 means neither side tends to be larger."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    labels = np.concatenate([np.zeros(x.size, dtype=int), np.ones(y.size, dtype=int)])
    return 2.0 * auc_score(labels, np.concatenate([x, y])) - 1.0


def length_arrays(texts: Iterable[str]) -> dict[str, np.ndarray]:
    chars, tokens = [], []
    for text in texts:
        chars.append(len(text))
        tokens.append(len(whitespace_tokens(text)))
    return {"chars": np.asarray(chars, dtype=float), "tokens": np.asarray(tokens, dtype=float)}


def compare_lengths(texts_a: Sequence[str], texts_b: Sequence[str]) -> dict[str, dict]:
    """Kolmogorov-Smirnov test and effect sizes for length in characters and in tokens."""
    lengths_a, lengths_b = length_arrays(texts_a), length_arrays(texts_b)
    out = {}
    for unit in ("chars", "tokens"):
        x, y = lengths_a[unit], lengths_b[unit]
        test = ks_two_sample(x, y)
        out[unit] = {
            "n_a": test.n_a,
            "n_b": test.n_b,
            "mean_a": float(x.mean()),
            "mean_b": float(y.mean()),
            "median_a": float(np.median(x)),
            "median_b": float(np.median(y)),
            "ks_statistic": test.statistic,
            "ks_p": test.p_value,
            "cliffs_delta": cliffs_delta(x, y),
        }
    return out


def total_variation(counts_a: Mapping[str, float], counts_b: Mapping[str, float]) -> float:
    """Half the L1 distance between the two share vectors."""
    total_a, total_b = sum(counts_a.values()), sum(counts_b.values())
    if total_a <= 0 or total_b <= 0:
        raise ValueError("both sides need at least one count")
    keys = set(counts_a) | set(counts_b)
    return 0.5 * sum(
        abs(counts_a.get(key, 0) / total_a - counts_b.get(key, 0) / total_b) for key in keys
    )


def compare_labels(labels_a: Sequence[str], labels_b: Sequence[str]) -> dict:
    """G-test of homogeneity on the 2 by k table of label counts, and total variation.

    ``G = 2 * sum O * ln(O / E)`` with ``E`` the product of the margins over the grand
    total; it is referred to a chi-square distribution with ``k - 1`` degrees of freedom.
    """
    counts_a, counts_b = Counter(labels_a), Counter(labels_b)
    categories = sorted(set(counts_a) | set(counts_b))
    table = np.array(
        [[counts_a.get(c, 0) for c in categories], [counts_b.get(c, 0) for c in categories]],
        dtype=float,
    )
    if table[0].sum() == 0 or table[1].sum() == 0:
        raise ValueError("both sides need at least one label")
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    observed = table > 0
    ratios = np.log(table[observed] / expected[observed])
    g_statistic = max(float(2.0 * (table[observed] * ratios).sum()), 0.0)
    dof = len(categories) - 1
    p_value = float(stats.chi2.sf(g_statistic, dof)) if dof > 0 else 1.0
    return {
        "categories": categories,
        "counts_a": [int(v) for v in table[0]],
        "counts_b": [int(v) for v in table[1]],
        "shares_a": [float(v) for v in table[0] / table[0].sum()],
        "shares_b": [float(v) for v in table[1] / table[1].sum()],
        "g_statistic": g_statistic,
        "dof": dof,
        "g_p": p_value,
        "total_variation": float(total_variation(counts_a, counts_b)),
    }


def unigram_counts(texts: Iterable[str], ngram: int = 1) -> Counter[str]:
    """Counts of word n-grams up to ``ngram`` words (default: single words)."""
    counts: Counter[str] = Counter()
    for text in texts:
        tokens = tokenize(text)
        counts.update(tokens)
        for size in range(2, ngram + 1):
            counts.update(" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1))
    return counts


def jensen_shannon(counts_a: Mapping[str, float], counts_b: Mapping[str, float]) -> float:
    """Jensen-Shannon divergence in bits between two count vectors (0 same, 1 disjoint).

    On finite samples it is biased upwards: two samples of one source never have
    identical counts. Read it against the split-half baseline.
    """
    total_a, total_b = sum(counts_a.values()), sum(counts_b.values())
    if total_a <= 0 or total_b <= 0:
        raise ValueError("both sides need at least one count")
    divergence = 0.0
    for key in set(counts_a) | set(counts_b):
        p = counts_a.get(key, 0) / total_a
        q = counts_b.get(key, 0) / total_b
        mid = 0.5 * (p + q)
        if p > 0:
            divergence += 0.5 * p * math.log2(p / mid)
        if q > 0:
            divergence += 0.5 * q * math.log2(q / mid)
    return min(max(divergence, 0.0), 1.0)
