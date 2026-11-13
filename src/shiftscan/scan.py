"""Glue between datasets and statistics: subsampling, the split-half baseline, attribution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.special import expit

from .c2st import ClassifierTest, classifier_test
from .features import (
    FeatureConfig,
    HashedFeatures,
    drop_rare_columns,
    hashed_counts,
    sublinear_rows,
    tfidf_rows,
)
from .io import Dataset
from .logodds import weighted_log_odds
from .logreg import fit_logistic
from .mmd import MMDTest, gram_matrix, mmd2_unbiased, mmd_test
from .simple import (
    compare_labels,
    compare_lengths,
    jensen_shannon,
    ks_two_sample,
    length_arrays,
    total_variation,
    unigram_counts,
)


@dataclass
class Options:
    """Every knob of a run; the command line flags map onto these one to one."""

    seed: int = 0
    max_records: int = 1000
    features: FeatureConfig = field(default_factory=FeatureConfig)
    folds: int = 5
    permutations: int = 100
    l2: float = 1.0
    mmd_records: int = 1000
    mmd_permutations: int = 500
    kernel: str = "rbf"
    alpha: float = 0.05
    min_auc: float = 0.6
    baseline_repeats: int = 3
    top: int = 15
    prior_total: float = 1000.0
    min_count: int = 5
    log_odds_ngrams: int = 1


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


def _mmd_rows(
    counts: sparse.csr_matrix, labels: np.ndarray, per_side: int, rng: np.random.Generator
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """TF-IDF rows of at most ``per_side`` random records from each side."""
    picks = []
    for value in (0, 1):
        members = np.flatnonzero(labels == value)
        size = min(per_side, members.size)
        picks.append(np.sort(rng.choice(members, size=size, replace=False)))
    index = np.concatenate(picks)
    return tfidf_rows(counts[index]), labels[index]


def run_mmd_test(
    pooled: Pooled,
    per_side: int = 1000,
    permutations: int = 500,
    seed: int = 0,
    kernel: str = "rbf",
) -> MMDTest:
    """MMD test on TF-IDF rows of the hashed counts, subsampled to ``per_side`` records a side."""
    rng = np.random.default_rng((seed, 1))
    rows, labels = _mmd_rows(pooled.counts, pooled.labels, per_side, rng)
    gram, bandwidth = gram_matrix(rows, kernel=kernel)
    return mmd_test(
        gram, labels, permutations=permutations, seed=seed, kernel=kernel, bandwidth=bandwidth
    )


def verdict(p_value: float, auc: float, alpha: float = 0.05, min_auc: float = 0.6) -> str:
    """One plain line from the classifier p-value and the held-out AUC."""
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


def tests_section(
    a: Dataset, b: Dataset, options: Options, permutations: int | None = None
) -> tuple[dict, Pooled, ClassifierTest]:
    """Classifier and MMD tests on an equal-size subsample, with the verdict line."""
    rng = np.random.default_rng(options.seed)
    sample_a, sample_b = equal_size_subsample(a, b, options.max_records, rng)
    pooled = pool(sample_a, sample_b, options.features)
    runs = options.permutations if permutations is None else permutations
    classifier = run_classifier_test(
        pooled, folds=options.folds, permutations=runs, seed=options.seed, l2=options.l2
    )
    mmd = run_mmd_test(
        pooled,
        per_side=options.mmd_records,
        permutations=options.mmd_permutations,
        seed=options.seed,
        kernel=options.kernel,
    )
    p_value = classifier.permutation_p
    p_kind = "permutation"
    if p_value is None:
        p_value, p_kind = classifier.binomial_p, "binomial"
    section = {
        "classifier": classifier.to_dict(),
        "mmd": mmd.to_dict(),
        "verdict_p_value": p_kind,
        "verdict": verdict(p_value, classifier.auc, options.alpha, options.min_auc),
    }
    return section, pooled, classifier


def features_section(
    a: Dataset, b: Dataset, pooled: Pooled, classifier: ClassifierTest, options: Options
) -> dict:
    """What drives the difference: log-odds, classifier weights and typical records."""
    counts_a = unigram_counts(a.texts, options.log_odds_ngrams)
    counts_b = unigram_counts(b.texts, options.log_odds_ngrams)
    scored = weighted_log_odds(counts_a, counts_b, options.prior_total, options.min_count)
    more_b = [row.to_dict() for row in scored[: options.top] if row.z > 0]
    more_a = [row.to_dict() for row in reversed(scored[-options.top :]) if row.z < 0]

    matrix, kept = drop_rare_columns(sublinear_rows(pooled.counts), min_df=2)
    model = fit_logistic(matrix, pooled.labels, l2=options.l2, max_iter=100)
    order = np.argsort(model.coef)

    def describe(column: int) -> dict:
        bucket = int(kept[column])
        return {
            "feature": pooled.hashed.name(bucket),
            "weight": float(model.coef[column]),
            "names_in_bucket": int(pooled.hashed.collisions.get(bucket, 0)),
        }

    toward_a = [describe(c) for c in order[: options.top] if model.coef[c] < 0]
    toward_b = [describe(c) for c in order[::-1][: options.top] if model.coef[c] > 0]

    ids = np.array(pooled.a.ids + pooled.b.ids, dtype=object)
    probability_b = expit(classifier.scores)
    rows_a = np.flatnonzero(pooled.labels == 0)
    rows_b = np.flatnonzero(pooled.labels == 1)
    typical_a = rows_a[np.argsort(probability_b[rows_a], kind="stable")[: options.top]]
    typical_b = rows_b[np.argsort(-probability_b[rows_b], kind="stable")[: options.top]]

    def record(row: int) -> dict:
        return {"id": str(ids[row]), "p_b": float(probability_b[row])}

    return {
        "log_odds": {
            "prior_total": options.prior_total,
            "min_count": options.min_count,
            "ngrams": options.log_odds_ngrams,
            "more_in_a": more_a,
            "more_in_b": more_b,
        },
        "classifier_weights": {"toward_a": toward_a, "toward_b": toward_b},
        "records": {
            "typical_of_a": [record(r) for r in typical_a],
            "typical_of_b": [record(r) for r in typical_b],
        },
    }


def _summarise(values: dict[str, list[float]]) -> dict:
    cross = float(np.mean(values["cross"]))
    within_a = float(np.mean(values["within_a"]))
    within_b = float(np.mean(values["within_b"]))
    baseline = 0.5 * (within_a + within_b)
    return {
        "cross": cross,
        "within_a": within_a,
        "within_b": within_b,
        "baseline": baseline,
        "excess": cross - baseline,
        "ratio": cross / baseline if baseline > 0 else None,
    }


def baseline_section(a: Dataset, b: Dataset, options: Options, heavy: bool = True) -> dict:
    """Every statistic between A and B and between two random halves of each, at one size.

    Each repeat draws two disjoint random parts of A and of B, all of the same size
    (half of the smaller dataset, capped at ``max_records``). ``cross`` compares the
    first part of A with the first part of B; ``within_a`` and ``within_b`` compare the
    two parts of one dataset. Sample-size effects (the upward bias of the
    Jensen-Shannon divergence, classifier noise) are the same in all three, so
    ``excess = cross - baseline`` and ``ratio = cross / baseline`` isolate the shift.
    With ``heavy=False`` the classifier and MMD rows are skipped.
    """
    size = min(len(a) // 2, len(b) // 2, options.max_records)
    if size < 2:
        raise ValueError("the split-half baseline needs at least 4 records in each dataset")
    run_heavy = heavy and size >= options.folds
    has_labels = a.labels is not None and b.labels is not None
    collected: dict[str, dict[str, list[float]]] = {}

    def add(name: str, comparison: str, value: float) -> None:
        collected.setdefault(name, {"cross": [], "within_a": [], "within_b": []})
        collected[name][comparison].append(float(value))

    for repeat in range(options.baseline_repeats):
        rng = np.random.default_rng((options.seed, 2, repeat))
        a1, a2 = split_halves(a, rng, size)
        b1, b2 = split_halves(b, rng, size)
        parts = {"a1": a1, "a2": a2, "b1": b1, "b2": b2}
        words = {key: unigram_counts(part.texts) for key, part in parts.items()}
        lengths = {key: length_arrays(part.texts) for key, part in parts.items()}
        offsets = {key: i * size for i, key in enumerate(parts)}
        hashed = None
        if run_heavy:
            hashed = hashed_counts(
                a1.texts + a2.texts + b1.texts + b2.texts, options.features
            ).counts
        pairs = {"cross": ("a1", "b1"), "within_a": ("a1", "a2"), "within_b": ("b1", "b2")}
        for comparison, (left, right) in pairs.items():
            add("unigram_jsd", comparison, jensen_shannon(words[left], words[right]))
            for unit in ("chars", "tokens"):
                test = ks_two_sample(lengths[left][unit], lengths[right][unit])
                add(f"length_ks_{unit}", comparison, test.statistic)
            if has_labels:
                tv = total_variation(
                    _label_counts(parts[left].labels), _label_counts(parts[right].labels)
                )
                add("label_total_variation", comparison, tv)
            if hashed is None:
                continue
            index = np.concatenate(
                [
                    np.arange(offsets[left], offsets[left] + size),
                    np.arange(offsets[right], offsets[right] + size),
                ]
            )
            labels = np.concatenate([np.zeros(size, dtype=np.int64), np.ones(size, dtype=np.int64)])
            rows = hashed[index]
            matrix, _ = drop_rare_columns(sublinear_rows(rows), min_df=2)
            fitted = classifier_test(
                matrix,
                labels,
                folds=options.folds,
                permutations=0,
                seed=options.seed + repeat,
                l2=options.l2,
            )
            add("classifier_auc", comparison, fitted.auc)
            add("classifier_accuracy", comparison, fitted.accuracy)
            if size >= 2:
                mmd_rng = np.random.default_rng((options.seed, 3, repeat))
                kernel_rows, kernel_labels = _mmd_rows(rows, labels, options.mmd_records, mmd_rng)
                gram, _ = gram_matrix(kernel_rows, kernel=options.kernel)
                add("mmd2", comparison, mmd2_unbiased(gram, kernel_labels))
    return {
        "records_per_part": size,
        "repeats": options.baseline_repeats,
        "mmd_records_per_side": min(size, options.mmd_records) if run_heavy else None,
        "statistics": {name: _summarise(values) for name, values in collected.items()},
    }


def _label_counts(labels: list[str] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels or []:
        counts[label] = counts.get(label, 0) + 1
    return counts


def describe(data: Dataset) -> dict:
    return {"name": data.name, "records": len(data)}


def lengths_section(a: Dataset, b: Dataset) -> dict:
    return compare_lengths(a.texts, b.texts)


def labels_section(a: Dataset, b: Dataset) -> dict:
    if a.labels is None or b.labels is None:
        raise ValueError("no label field was given")
    return compare_labels(a.labels, b.labels)


def full_report(a: Dataset, b: Dataset, options: Options) -> dict:
    """Everything: tests, simple comparisons, the split-half baseline and attribution."""
    tests, pooled, classifier = tests_section(a, b, options)
    baseline = baseline_section(a, b, options)
    report = {
        "a": describe(a),
        "b": describe(b),
        "settings": {
            "seed": options.seed,
            "max_records": options.max_records,
            "folds": options.folds,
            "buckets": options.features.n_buckets,
            "word_ngrams": list(options.features.word_ngrams),
            "char_ngrams": list(options.features.char_ngrams),
            "l2": options.l2,
            "alpha": options.alpha,
            "min_auc": options.min_auc,
        },
        "tests": tests,
        "lengths": lengths_section(a, b),
        "labels": labels_section(a, b) if a.labels is not None and b.labels is not None else None,
        "unigram_jsd": jensen_shannon(unigram_counts(a.texts), unigram_counts(b.texts)),
        "baseline": baseline,
        "features": features_section(a, b, pooled, classifier, options),
    }
    auc_row = baseline["statistics"].get("classifier_auc")
    if auc_row is not None:
        report["tests"]["verdict"] += (
            f"; at {baseline['records_per_part']} records per part the AUC is "
            f"{auc_row['cross']:.3f} between the datasets and {auc_row['baseline']:.3f} "
            "between halves of one dataset"
        )
    return report
