"""Glue between files and the tests: subsampling, feature building and verdicts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .c2st import ClassifierTest, classifier_test
from .features import (
    FeatureConfig,
    HashedFeatures,
    drop_rare_columns,
    hashed_counts,
    sublinear_rows,
)
from .io import Dataset


@dataclass
class Pooled:
    """Both samples stacked: side A rows first, then side B rows."""

    a: Dataset
    b: Dataset
    hashed: HashedFeatures
    labels: np.ndarray

    @property
    def counts(self) -> sparse.csr_matrix:
        return self.hashed.counts


def equal_size_subsample(
    a: Dataset, b: Dataset, max_records: int | None, rng: np.random.Generator
) -> tuple[Dataset, Dataset]:
    """Draw the same number of records from each side without replacement.

    The size is the smaller of the two datasets, capped at ``max_records``. Records
    keep their file order inside each sample.
    """
    size = min(len(a), len(b))
    if max_records is not None:
        size = min(size, max_records)
    if size < 1:
        raise ValueError("both datasets need at least one record")
    pick_a = np.sort(rng.choice(len(a), size=size, replace=False))
    pick_b = np.sort(rng.choice(len(b), size=size, replace=False))
    return a.take(pick_a), b.take(pick_b)


def split_halves(
    data: Dataset, rng: np.random.Generator, size: int | None = None
) -> tuple[Dataset, Dataset]:
    """Two disjoint random parts of one dataset, ``size`` records each (default: half)."""
    half = len(data) // 2
    size = half if size is None else min(size, half)
    if size < 1:
        raise ValueError(f"{data.name}: too few records to split")
    order = rng.permutation(len(data))
    first = np.sort(order[:size])
    second = np.sort(order[size : 2 * size])
    return data.take(first, f"{data.name} (half 1)"), data.take(second, f"{data.name} (half 2)")


def pool(a: Dataset, b: Dataset, config: FeatureConfig | None = None) -> Pooled:
    hashed = hashed_counts(a.texts + b.texts, config)
    labels = np.concatenate([np.zeros(len(a), dtype=np.int64), np.ones(len(b), dtype=np.int64)])
    return Pooled(a=a, b=b, hashed=hashed, labels=labels)


def run_classifier_test(
    pooled: Pooled,
    folds: int = 5,
    permutations: int = 100,
    seed: int = 0,
    l2: float = 1.0,
) -> ClassifierTest:
    """Classifier test on sublinear, unit-length rows; columns seen once are dropped."""
    matrix, _ = drop_rare_columns(sublinear_rows(pooled.counts), min_df=2)
    return classifier_test(
        matrix, pooled.labels, folds=folds, permutations=permutations, seed=seed, l2=l2
    )


def verdict(p_value: float, auc: float, alpha: float = 0.05, min_auc: float = 0.6) -> str:
    """One plain line from a p-value and the held-out AUC."""
    if p_value >= alpha:
        return (
            f"no shift detected: p = {p_value:.3g} is not below alpha = {alpha:g} "
            f"(held-out AUC {auc:.3f})"
        )
    if auc < min_auc:
        return (
            f"shift detected but small: p = {p_value:.3g} < alpha = {alpha:g}, "
            f"held-out AUC {auc:.3f} is below --min-auc {min_auc:g}"
        )
    return (
        f"shift detected: p = {p_value:.3g} < alpha = {alpha:g}, "
        f"held-out AUC {auc:.3f} is at or above --min-auc {min_auc:g}"
    )
