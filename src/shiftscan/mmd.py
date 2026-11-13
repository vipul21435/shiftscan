"""Kernel two-sample test: maximum mean discrepancy with a permutation null.

The statistic is the unbiased estimate of the squared MMD of Gretton et al.
(2012, equation 3 of the JMLR paper)::

    MMD2_u = 1/(m(m-1)) sum_{i != j} k(x_i, x_j)
           + 1/(n(n-1)) sum_{i != j} k(y_i, y_j)
           - 2/(mn)     sum_{i, j}   k(x_i, y_j)

It is zero in expectation when both samples share a distribution and can be
slightly negative. The kernel is Gaussian on the rows given (TF-IDF vectors of
unit length in this package), with the bandwidth set to the median pairwise
distance of the pooled sample, or linear (the plain dot product).

The p-value comes from recomputing the statistic on random reassignments of the
pooled records to the two sides; the Gram matrix is built once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class MMDTest:
    n_a: int
    n_b: int
    kernel: str
    bandwidth: float | None
    mmd2: float
    p_value: float | None
    permutations: int
    null_mean: float | None
    null_q95: float | None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def gram_matrix(
    rows: np.ndarray | sparse.spmatrix, kernel: str = "rbf", bandwidth: float | None = None
) -> tuple[np.ndarray, float | None]:
    """Kernel matrix of all rows against all rows, and the bandwidth used (``rbf`` only)."""
    product = rows @ rows.T
    dots = product.toarray() if sparse.issparse(product) else np.asarray(product, dtype=float)
    if kernel == "linear":
        return dots, None
    if kernel != "rbf":
        raise ValueError(f"unknown kernel {kernel!r}")
    norms = np.diag(dots)
    squared = np.maximum(norms[:, None] + norms[None, :] - 2.0 * dots, 0.0)
    if bandwidth is None:
        upper = squared[np.triu_indices_from(squared, k=1)]
        median = float(np.median(np.sqrt(upper))) if upper.size else 0.0
        bandwidth = median if median > 0 else 1.0
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    return np.exp(-squared / (2.0 * bandwidth**2)), float(bandwidth)


def mmd2_unbiased(gram: np.ndarray, labels: np.ndarray) -> float:
    """Unbiased squared MMD between rows labelled 0 and rows labelled 1."""
    labels = np.asarray(labels)
    side_a = labels == 0
    side_b = ~side_a
    m, n = int(side_a.sum()), int(side_b.sum())
    if m < 2 or n < 2:
        raise ValueError("MMD needs at least two records on each side")
    block_a = gram[np.ix_(side_a, side_a)]
    block_b = gram[np.ix_(side_b, side_b)]
    cross = gram[np.ix_(side_a, side_b)]
    within_a = (block_a.sum() - np.trace(block_a)) / (m * (m - 1))
    within_b = (block_b.sum() - np.trace(block_b)) / (n * (n - 1))
    return float(within_a + within_b - 2.0 * cross.sum() / (m * n))


def _mmd2_for_indicators(gram: np.ndarray, indicators: np.ndarray) -> np.ndarray:
    """Statistic for every row of 0/1 ``indicators`` (1 marks side A), without slicing."""
    diagonal = np.diag(gram)
    row_sums = gram.sum(axis=1)
    total = row_sums.sum()
    projected = indicators @ gram
    full_aa = np.einsum("ij,ij->i", projected, indicators)
    sum_ab = indicators @ row_sums - full_aa
    full_bb = total - full_aa - 2.0 * sum_ab
    diag_a = indicators @ diagonal
    diag_b = diagonal.sum() - diag_a
    m = indicators.sum(axis=1)
    n = indicators.shape[1] - m
    return (
        (full_aa - diag_a) / (m * (m - 1))
        + (full_bb - diag_b) / (n * (n - 1))
        - 2.0 * sum_ab / (m * n)
    )


def mmd_test(
    gram: np.ndarray,
    labels: np.ndarray,
    permutations: int = 500,
    seed: int = 0,
    kernel: str = "rbf",
    bandwidth: float | None = None,
    chunk: int = 50,
) -> MMDTest:
    """Permutation test on a precomputed Gram matrix."""
    labels = np.asarray(labels).astype(np.int64)
    observed = mmd2_unbiased(gram, labels)
    n_b = int(labels.sum())
    n_a = int(labels.size - n_b)
    p_value = null_mean = null_q95 = None
    if permutations > 0:
        rng = np.random.default_rng(seed)
        base = (labels == 0).astype(np.float64)
        null = np.empty(permutations)
        for start in range(0, permutations, chunk):
            stop = min(start + chunk, permutations)
            batch = np.stack([rng.permutation(base) for _ in range(stop - start)])
            null[start:stop] = _mmd2_for_indicators(gram, batch)
        p_value = float((1 + int((null >= observed).sum())) / (permutations + 1))
        null_mean = float(null.mean())
        null_q95 = float(np.quantile(null, 0.95))
    return MMDTest(
        n_a=n_a,
        n_b=n_b,
        kernel=kernel,
        bandwidth=bandwidth,
        mmd2=observed,
        p_value=p_value,
        permutations=permutations,
        null_mean=null_mean,
        null_q95=null_q95,
    )
