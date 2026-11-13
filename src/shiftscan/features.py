"""Hashed word and character n-gram features for short and long texts.

Every feature name (``w:the cat`` for a word bigram, ``c: ca`` for a character
trigram inside a padded word) is mapped to one of ``n_buckets`` columns by
CRC-32, so the feature space is fixed in advance and does not depend on which
records were seen. That keeps the classifier test free of vocabulary leakage
between folds and makes the permutation null cheap: the matrix is built once.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


@dataclass(frozen=True)
class FeatureConfig:
    """Which n-grams are extracted and how many hash buckets they share.

    A range ``(0, 0)`` switches that family off.
    """

    word_ngrams: tuple[int, int] = (1, 2)
    char_ngrams: tuple[int, int] = (3, 5)
    n_buckets: int = 2**18

    def __post_init__(self) -> None:
        if self.n_buckets < 2:
            raise ValueError("n_buckets must be at least 2")
        for lo, hi in (self.word_ngrams, self.char_ngrams):
            if lo < 0 or hi < lo:
                raise ValueError(f"invalid n-gram range ({lo}, {hi})")
        if self.word_ngrams[1] == 0 and self.char_ngrams[1] == 0:
            raise ValueError("at least one n-gram family must be enabled")


@dataclass
class HashedFeatures:
    """Counts per record and bucket, with a readable name for every used bucket."""

    counts: sparse.csr_matrix
    names: dict[int, str]
    collisions: dict[int, int]

    def name(self, bucket: int) -> str:
        return self.names.get(bucket, f"bucket:{bucket}")


def tokenize(text: str) -> list[str]:
    """Lower-cased runs of letters and digits; an inner apostrophe stays inside a word."""
    return _WORD_RE.findall(text.lower())


def whitespace_tokens(text: str) -> list[str]:
    return text.split()


def word_ngram_names(tokens: Sequence[str], lo: int, hi: int) -> list[str]:
    out: list[str] = []
    for n in range(max(lo, 1), hi + 1):
        if n == 1:
            out.extend("w:" + t for t in tokens)
        else:
            out.extend("w:" + " ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return out


def char_ngram_names(tokens: Sequence[str], lo: int, hi: int) -> list[str]:
    """Character n-grams inside each word, the word padded with one space on each side."""
    out: list[str] = []
    for token in tokens:
        padded = " " + token + " "
        size = len(padded)
        for n in range(max(lo, 1), hi + 1):
            if n > size:
                break
            out.extend("c:" + padded[i : i + n] for i in range(size - n + 1))
    return out


def feature_names(text: str, config: FeatureConfig) -> list[str]:
    tokens = tokenize(text)
    out: list[str] = []
    if config.word_ngrams[1] > 0:
        out.extend(word_ngram_names(tokens, *config.word_ngrams))
    if config.char_ngrams[1] > 0:
        out.extend(char_ngram_names(tokens, *config.char_ngrams))
    return out


def bucket_of(name: str, n_buckets: int) -> int:
    """Column of a feature name. CRC-32 is stable across runs, platforms and Python versions."""
    return zlib.crc32(name.encode("utf-8")) % n_buckets


def hashed_counts(texts: Iterable[str], config: FeatureConfig | None = None) -> HashedFeatures:
    """Count hashed features of every text.

    Returns the count matrix (records by buckets) and, for every bucket in use, the
    most frequent feature name that landed in it (ties go to the alphabetically
    first) together with how many distinct names share the bucket.
    """
    config = config or FeatureConfig()
    cache: dict[str, int] = {}
    totals: Counter[str] = Counter()
    indptr = [0]
    cols: list[int] = []
    vals: list[int] = []
    for text in texts:
        seen = Counter(feature_names(text, config))
        for name, count in seen.items():
            bucket = cache.get(name)
            if bucket is None:
                bucket = bucket_of(name, config.n_buckets)
                cache[name] = bucket
            cols.append(bucket)
            vals.append(count)
        totals.update(seen)
        indptr.append(len(cols))
    counts = sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float64), np.asarray(cols, dtype=np.int64), indptr),
        shape=(len(indptr) - 1, config.n_buckets),
    )
    counts.sum_duplicates()
    best: dict[int, tuple[int, str]] = {}
    collisions: dict[int, int] = {}
    for name, bucket in cache.items():
        collisions[bucket] = collisions.get(bucket, 0) + 1
        key = (-totals[name], name)
        if bucket not in best or key < best[bucket]:
            best[bucket] = key
    names = {bucket: key[1] for bucket, key in best.items()}
    return HashedFeatures(counts=counts, names=names, collisions=collisions)


def _l2_normalise_rows(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0.0] = 1.0
    return sparse.diags(1.0 / norms) @ matrix


def sublinear_rows(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """``1 + log(count)`` for every non-zero count, then rows scaled to unit length.

    Uses nothing but the record itself, so it can be applied before any split.
    """
    out = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
    out.data = 1.0 + np.log(out.data)
    return sparse.csr_matrix(_l2_normalise_rows(out))


def tfidf_rows(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """TF-IDF with smoothed inverse document frequency, rows scaled to unit length.

    ``idf = log((1 + n) / (1 + df)) + 1`` over the rows given. Computed on the
    pooled sample it does not depend on which side a record came from, so it is
    unchanged under a permutation of the side labels.
    """
    matrix = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
    n_rows = matrix.shape[0]
    doc_freq = np.bincount(matrix.indices, minlength=matrix.shape[1]).astype(np.float64)
    idf = np.log((1.0 + n_rows) / (1.0 + doc_freq)) + 1.0
    matrix.data = matrix.data * idf[matrix.indices]
    return sparse.csr_matrix(_l2_normalise_rows(matrix))


def drop_rare_columns(
    matrix: sparse.csr_matrix, min_df: int = 2
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Keep only columns present in at least ``min_df`` rows.

    Returns the narrower matrix and the original column index of every kept column.
    The choice looks at the pooled rows only, never at the side labels.
    """
    doc_freq = np.bincount(matrix.indices, minlength=matrix.shape[1])
    keep = np.flatnonzero(doc_freq >= min_df)
    return sparse.csr_matrix(matrix[:, keep]), keep
