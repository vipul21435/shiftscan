from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

from shiftscan.features import (
    FeatureConfig,
    bucket_of,
    char_ngram_names,
    drop_rare_columns,
    feature_names,
    hashed_counts,
    sublinear_rows,
    tfidf_rows,
    tokenize,
    word_ngram_names,
)


def test_tokenize_lowercases_and_keeps_inner_apostrophes():
    assert tokenize("Don't STOP, the_end 42!") == ["don't", "stop", "the", "end", "42"]


def test_word_ngrams_by_hand():
    assert word_ngram_names(["a", "b", "c"], 1, 2) == ["w:a", "w:b", "w:c", "w:a b", "w:b c"]


def test_char_ngrams_are_padded_inside_words():
    assert char_ngram_names(["ab"], 3, 4) == ["c: ab", "c:ab ", "c: ab "]
    assert char_ngram_names(["a"], 4, 5) == []


def test_feature_names_respects_disabled_family():
    config = FeatureConfig(word_ngrams=(1, 1), char_ngrams=(0, 0))
    assert feature_names("Red fox", config) == ["w:red", "w:fox"]


def test_bucket_is_crc32_and_stable():
    # zlib.crc32(b"w:cat") is 1935231421 on every platform
    assert bucket_of("w:cat", 2**18) == 1935231421 % 2**18 == 84413
    assert bucket_of("w:cat", 7) == 1935231421 % 7


def test_hashed_counts_totals_and_names():
    config = FeatureConfig(word_ngrams=(1, 1), char_ngrams=(0, 0), n_buckets=2**16)
    hashed = hashed_counts(["cat cat dog", "dog", ""], config)
    assert hashed.counts.shape == (3, 2**16)
    assert hashed.counts.sum(axis=1).A1.tolist() == [3.0, 1.0, 0.0]
    cat = bucket_of("w:cat", 2**16)
    assert hashed.counts[0, cat] == pytest.approx(2.0)
    assert hashed.name(cat) == "w:cat"
    assert hashed.name(1) in ("bucket:1", hashed.names.get(1))


def test_colliding_names_share_a_bucket_and_report_it():
    config = FeatureConfig(word_ngrams=(1, 1), char_ngrams=(0, 0), n_buckets=2)
    hashed = hashed_counts(["one two three four five five"], config)
    assert hashed.counts.sum() == pytest.approx(6.0)
    assert sum(hashed.collisions.values()) == 5
    for bucket, name in hashed.names.items():
        assert bucket_of(name, 2) == bucket


def test_sublinear_rows_have_unit_length_and_keep_empty_rows():
    counts = sparse.csr_matrix(np.array([[4.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 2.0]]))
    out = sublinear_rows(counts).toarray()
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)
    assert np.linalg.norm(out[1]) == pytest.approx(0.0)
    expected = np.array([1 + np.log(4.0), 1.0, 0.0])
    assert out[0] == pytest.approx(expected / np.linalg.norm(expected))
    assert counts[0, 0] == pytest.approx(4.0)


def test_tfidf_matches_scikit_learn(rng):
    counts = sparse.csr_matrix(rng.poisson(0.6, size=(30, 12)).astype(float))
    ours = tfidf_rows(counts).toarray()
    reference = TfidfTransformer(norm="l2", smooth_idf=True, sublinear_tf=False)
    assert ours == pytest.approx(reference.fit_transform(counts).toarray(), abs=1e-12)


def test_drop_rare_columns_keeps_shared_ones():
    matrix = sparse.csr_matrix(np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 3.0]]))
    kept, index = drop_rare_columns(matrix, min_df=2)
    assert index.tolist() == [0]
    assert kept.shape == (3, 1)


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        FeatureConfig(n_buckets=1)
    with pytest.raises(ValueError):
        FeatureConfig(word_ngrams=(0, 0), char_ngrams=(0, 0))
    with pytest.raises(ValueError):
        FeatureConfig(word_ngrams=(3, 1))
