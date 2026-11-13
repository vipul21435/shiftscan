from __future__ import annotations

import json

import numpy as np
import pytest

from shiftscan.features import FeatureConfig
from shiftscan.io import Dataset
from shiftscan.scan import (
    Options,
    baseline_section,
    equal_size_subsample,
    features_section,
    full_report,
    labels_section,
    split_halves,
    verdict,
)
from shiftscan.scan import tests_section as run_tests

from .conftest import ANIMALS, TOOLS, make_texts


def dataset(name: str, texts: list[str], labels: list[str] | None = None) -> Dataset:
    return Dataset(name, [f"{name}-{i}" for i in range(len(texts))], texts, labels)


@pytest.fixture
def options() -> Options:
    return Options(
        max_records=80,
        permutations=19,
        mmd_records=60,
        mmd_permutations=99,
        baseline_repeats=2,
        top=5,
        min_count=2,
        features=FeatureConfig(n_buckets=2**14),
    )


@pytest.fixture
def shifted():
    rng = np.random.default_rng(5)
    a = dataset("a", make_texts(rng, 120, ANIMALS, 0.4), list(rng.choice(["x", "y"], size=120)))
    b = dataset("b", make_texts(rng, 100, TOOLS, 0.4), list(rng.choice(["x", "y"], size=100)))
    return a, b


@pytest.fixture
def same_source():
    rng = np.random.default_rng(6)
    return (
        dataset("a", make_texts(rng, 120, ANIMALS, 0.4)),
        dataset("b", make_texts(rng, 120, ANIMALS, 0.4)),
    )


def test_equal_size_subsample_sizes_and_seed(shifted):
    a, b = shifted
    first = equal_size_subsample(a, b, 50, np.random.default_rng(1))
    second = equal_size_subsample(a, b, 50, np.random.default_rng(1))
    assert len(first[0]) == len(first[1]) == 50
    assert first[0].ids == second[0].ids and first[1].ids == second[1].ids
    assert len(set(first[0].ids)) == 50
    uncapped = equal_size_subsample(a, b, None, np.random.default_rng(1))
    assert len(uncapped[0]) == len(uncapped[1]) == 100
    with pytest.raises(ValueError):
        equal_size_subsample(dataset("e", []), b, None, np.random.default_rng(1))


def test_split_halves_are_disjoint(shifted):
    a, _ = shifted
    left, right = split_halves(a, np.random.default_rng(2))
    assert len(left) == len(right) == 60
    assert not set(left.ids) & set(right.ids)
    small_left, small_right = split_halves(a, np.random.default_rng(2), size=10)
    assert len(small_left) == len(small_right) == 10
    with pytest.raises(ValueError):
        split_halves(dataset("one", ["x"]), np.random.default_rng(2))


def test_planted_shift_is_found_by_both_tests(shifted, options):
    section, pooled, classifier = run_tests(*shifted, options)
    assert section["classifier"]["auc"] > 0.95
    assert section["classifier"]["permutation_p"] == pytest.approx(1 / 20)
    assert section["mmd"]["p_value"] == pytest.approx(1 / 100)
    assert section["mmd"]["n_a"] == 60
    assert section["verdict"].startswith("shift detected but small") is False
    assert pooled.counts.shape[0] == 160
    assert classifier.scores.shape == (160,)


def test_same_source_is_not_flagged(same_source, options):
    section, _, _ = run_tests(*same_source, options)
    assert section["classifier"]["auc"] < 0.7
    assert section["mmd"]["p_value"] > 0.01
    assert section["verdict"].startswith(("no shift detected", "shift detected but small"))


def test_binomial_p_value_is_used_without_permutations(shifted, options):
    section, _, _ = run_tests(*shifted, options, permutations=0)
    assert section["verdict_p_value"] == "binomial"
    assert section["classifier"]["permutation_p"] is None


def test_verdict_has_three_outcomes():
    assert verdict(0.2, 0.52).startswith("no shift detected")
    assert verdict(0.01, 0.55).startswith("shift detected but small")
    assert verdict(0.01, 0.9).startswith("shift detected:")
    assert verdict(0.01, 0.55, min_auc=0.5).startswith("shift detected:")
    assert verdict(0.04, 0.9, alpha=0.01).startswith("no shift detected")


def test_baseline_separates_a_real_shift_from_split_noise(shifted, same_source, options):
    moved = baseline_section(*shifted, options)["statistics"]
    assert moved["unigram_jsd"]["ratio"] > 3
    assert moved["classifier_auc"]["excess"] > 0.3
    assert moved["mmd2"]["excess"] > 0
    assert moved["label_total_variation"]["baseline"] >= 0
    still = baseline_section(*same_source, options)
    stats = still["statistics"]
    assert still["records_per_part"] == 60
    assert 0.5 < stats["unigram_jsd"]["ratio"] < 1.6
    assert abs(stats["classifier_auc"]["excess"]) < 0.2
    assert "label_total_variation" not in stats
    row = stats["unigram_jsd"]
    assert row["baseline"] == pytest.approx(0.5 * (row["within_a"] + row["within_b"]))
    assert row["excess"] == pytest.approx(row["cross"] - row["baseline"])


def test_light_baseline_skips_the_classifier_and_mmd(shifted, options):
    light = baseline_section(*shifted, options, heavy=False)
    assert "classifier_auc" not in light["statistics"]
    assert "mmd2" not in light["statistics"]
    assert light["mmd_records_per_side"] is None
    with pytest.raises(ValueError):
        baseline_section(dataset("a", ["x", "y"]), dataset("b", ["x", "y"]), options)


def test_attribution_names_the_planted_words_and_the_right_records(shifted, options):
    a, b = shifted
    _, pooled, classifier = run_tests(a, b, options, permutations=0)
    section = features_section(a, b, pooled, classifier, options)
    assert {row["feature"] for row in section["log_odds"]["more_in_b"]} <= set(TOOLS)
    assert {row["feature"] for row in section["log_odds"]["more_in_a"]} <= set(ANIMALS)
    assert all(row["z"] > 0 for row in section["log_odds"]["more_in_b"])
    assert all(row["weight"] > 0 for row in section["classifier_weights"]["toward_b"])
    assert all(row["weight"] < 0 for row in section["classifier_weights"]["toward_a"])
    assert all(r["id"].startswith("a-") for r in section["records"]["typical_of_a"])
    assert all(r["id"].startswith("b-") for r in section["records"]["typical_of_b"])
    assert section["records"]["typical_of_b"][0]["p_b"] > 0.5
    assert len(section["records"]["typical_of_a"]) == 5


def test_full_report_is_deterministic_under_a_seed(shifted, options):
    first = json.dumps(full_report(*shifted, options), sort_keys=True)
    second = json.dumps(full_report(*shifted, options), sort_keys=True)
    assert first == second
    report = json.loads(first)
    assert report["labels"]["categories"] == ["x", "y"]
    assert "between halves of one dataset" in report["tests"]["verdict"]
    other = Options(**{**options.__dict__, "seed": 1})
    assert json.dumps(full_report(*shifted, other), sort_keys=True) != first


def test_labels_section_needs_labels(same_source):
    with pytest.raises(ValueError):
        labels_section(*same_source)
