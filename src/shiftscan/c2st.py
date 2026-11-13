"""Classifier two-sample test with cross-fitted held-out predictions.

If two samples come from one distribution, no classifier can tell held-out
records of one from the other better than chance (Lopez-Paz and Oquab, 2017).
Here every record is scored by a model that never saw it: the pooled sample is
cut into stratified folds, a logistic regression is fitted on all folds but one
and scores the fold left out. The pooled held-out accuracy and AUC are the
effect sizes.

Two p-values are reported. The binomial one is the single-split test of the
paper: it looks at the first fold only, whose records were scored by one model
trained on the other folds, and refers the number of correct predictions to
Binomial(n_fold, 1/2). Pooling all folds into one binomial count would ignore
that the fold models share training data. The permutation one repeats the whole
cross-fitting on shuffled side labels and uses every record.

The fold models are fitted to a looser tolerance than :func:`fit_logistic` uses
by default (``FOLD_TOL`` and ``FOLD_FTOL``): the ranking of held-out records
settles long before the weights do, and the null needs hundreds of fits. The
same settings are used for the observed and the permuted labels, so the
permutation p-value is valid whatever the tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .logreg import Matrix, fit_logistic

FOLD_TOL = 1e-3
FOLD_FTOL = 1e-7


@dataclass
class ClassifierTest:
    n_a: int
    n_b: int
    folds: int
    accuracy: float
    auc: float
    binomial_p: float
    binomial_n: int
    binomial_correct: int
    permutation_p: float | None
    permutations: int
    null_auc_mean: float | None
    null_auc_q95: float | None
    scores: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "n_a": self.n_a,
            "n_b": self.n_b,
            "folds": self.folds,
            "accuracy": self.accuracy,
            "auc": self.auc,
            "binomial_p": self.binomial_p,
            "binomial_n": self.binomial_n,
            "binomial_correct": self.binomial_correct,
            "permutation_p": self.permutation_p,
            "permutations": self.permutations,
            "null_auc_mean": self.null_auc_mean,
            "null_auc_q95": self.null_auc_q95,
        }


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve from mid-ranks (ties count one half)."""
    labels = np.asarray(labels)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs records from both sides")
    ranks = stats.rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def stratified_folds(labels: np.ndarray, folds: int, rng: np.random.Generator) -> np.ndarray:
    """Fold number of every record, each side spread evenly over the folds."""
    labels = np.asarray(labels)
    assignment = np.empty(labels.size, dtype=np.int64)
    for value in (0, 1):
        members = np.flatnonzero(labels == value)
        order = rng.permutation(members)
        assignment[order] = np.arange(order.size) % folds
    return assignment


def _cross_fit(
    features: Matrix,
    labels: np.ndarray,
    folds: int,
    rng: np.random.Generator,
    l2: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Held-out decision scores and the fold number of every record."""
    labels = np.asarray(labels)
    assignment = stratified_folds(labels, folds, rng)
    scores = np.zeros(labels.size)
    for fold in range(folds):
        held_out = assignment == fold
        train = ~held_out
        model = fit_logistic(
            features[train], labels[train], l2=l2, max_iter=max_iter, tol=FOLD_TOL, ftol=FOLD_FTOL
        )
        scores[held_out] = model.decision_function(features[held_out])
    return scores, assignment


def cross_fitted_scores(
    features: Matrix,
    labels: np.ndarray,
    folds: int,
    rng: np.random.Generator,
    l2: float = 1.0,
    max_iter: int = 100,
) -> np.ndarray:
    """Decision score of every record from the model that did not train on it."""
    return _cross_fit(features, labels, folds, rng, l2, max_iter)[0]


def classifier_test(
    features: Matrix,
    labels: np.ndarray,
    folds: int = 5,
    permutations: int = 100,
    seed: int = 0,
    l2: float = 1.0,
    max_iter: int = 100,
) -> ClassifierTest:
    """Run the test on pooled ``features`` with ``labels`` 0 (side A) and 1 (side B)."""
    labels = np.asarray(labels).astype(np.int64)
    n_b = int(labels.sum())
    n_a = int(labels.size - n_b)
    if min(n_a, n_b) < folds:
        raise ValueError(f"each side needs at least {folds} records for {folds} folds")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    rng = np.random.default_rng(seed)
    scores, assignment = _cross_fit(features, labels, folds, rng, l2, max_iter)
    hits = (scores > 0).astype(np.int64) == labels
    accuracy = int(hits.sum()) / labels.size
    auc = auc_score(labels, scores)
    first_fold = assignment == 0
    binomial_n = int(first_fold.sum())
    binomial_correct = int(hits[first_fold].sum())
    binomial_p = float(stats.binom.sf(binomial_correct - 1, binomial_n, 0.5))
    permutation_p = null_mean = null_q95 = None
    if permutations > 0:
        null = np.empty(permutations)
        for i in range(permutations):
            shuffled = rng.permutation(labels)
            null_scores = cross_fitted_scores(
                features, shuffled, folds, rng, l2=l2, max_iter=max_iter
            )
            null[i] = auc_score(shuffled, null_scores)
        permutation_p = float((1 + int((null >= auc).sum())) / (permutations + 1))
        null_mean = float(null.mean())
        null_q95 = float(np.quantile(null, 0.95))
    return ClassifierTest(
        n_a=n_a,
        n_b=n_b,
        folds=folds,
        accuracy=float(accuracy),
        auc=auc,
        binomial_p=binomial_p,
        binomial_n=binomial_n,
        binomial_correct=binomial_correct,
        permutation_p=permutation_p,
        permutations=permutations,
        null_auc_mean=null_mean,
        null_auc_q95=null_q95,
        scores=scores,
        labels=labels,
    )
