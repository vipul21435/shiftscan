from __future__ import annotations

import math

import pytest

from shiftscan.logodds import weighted_log_odds


def test_hand_worked_example():
    # 40 words on each side, prior total 4 -> alpha = 4 * 40/80 = 2 for both words.
    # x: delta = log(32/12) - log(12/32) = 2 log(8/3); var = 1/32 + 1/12
    rows = weighted_log_odds({"x": 10, "y": 30}, {"x": 30, "y": 10}, prior_total=4.0)
    by_name = {row.feature: row for row in rows}
    delta = 2 * math.log(8 / 3)
    z = delta / math.sqrt(1 / 32 + 1 / 12)
    assert by_name["x"].log_odds == pytest.approx(delta)
    assert by_name["x"].z == pytest.approx(z)
    assert by_name["x"].z == pytest.approx(5.7951, abs=1e-4)
    assert by_name["y"].z == pytest.approx(-z)
    assert [row.feature for row in rows] == ["x", "y"]
    assert by_name["x"].to_dict()["count_b"] == 30


def test_unequal_sizes_by_hand():
    # n_a = 100, n_b = 50, prior total 15: alpha_w = 15 * (y_a + y_b) / 150
    counts_a = {"u": 20, "v": 80}
    counts_b = {"u": 25, "v": 25}
    row = {r.feature: r for r in weighted_log_odds(counts_a, counts_b, prior_total=15.0)}["u"]
    alpha = 15 * 45 / 150
    delta = math.log((25 + alpha) / (50 + 15 - 25 - alpha)) - math.log(
        (20 + alpha) / (100 + 15 - 20 - alpha)
    )
    assert alpha == pytest.approx(4.5)
    assert row.log_odds == pytest.approx(delta)
    assert row.z == pytest.approx(delta / math.sqrt(1 / 29.5 + 1 / 24.5))


def test_same_shares_give_zero_and_rare_words_are_shrunk():
    rows = weighted_log_odds({"a": 50, "b": 50}, {"a": 500, "b": 500})
    assert all(abs(row.z) < 1e-9 for row in rows)
    counts_a = {"common": 1000, "rare": 0, "mid": 100}
    counts_b = {"common": 1000, "rare": 2, "mid": 200}
    by_name = {r.feature: r for r in weighted_log_odds(counts_a, counts_b, prior_total=100.0)}
    assert abs(by_name["rare"].z) < abs(by_name["mid"].z)
    assert by_name["rare"].log_odds > by_name["mid"].log_odds


def test_min_count_and_errors():
    rows = weighted_log_odds({"a": 9, "b": 1}, {"a": 9, "c": 1}, min_count=2)
    assert [row.feature for row in rows] == ["a"]
    with pytest.raises(ValueError):
        weighted_log_odds({}, {"a": 1})
    with pytest.raises(ValueError):
        weighted_log_odds({"a": 1}, {"a": 1}, prior_total=0)
