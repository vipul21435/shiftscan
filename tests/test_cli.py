from __future__ import annotations

import json

import numpy as np
import pytest

from shiftscan.cli import main

from .conftest import ANIMALS, TOOLS, make_texts, write_jsonl


@pytest.fixture
def pair(tmp_path):
    rng = np.random.default_rng(3)
    a = write_jsonl(tmp_path / "a.jsonl", make_texts(rng, 120, ANIMALS, 0.5))
    b = write_jsonl(tmp_path / "b.jsonl", make_texts(rng, 120, TOOLS, 0.5))
    return a, b


def test_test_command_detects_a_planted_shift(pair, capsys):
    a, b = pair
    code = main(["test", str(a), str(b), "--text", "text", "--permutations", "39"])
    out = capsys.readouterr().out
    assert code == 0
    assert "held-out AUC: 1.000" in out
    assert "Verdict: shift detected:" in out


def test_test_command_json_and_self_split(pair, capsys):
    a, _ = pair
    args = ["test", str(a), "--self-split", "--text", "text", "--permutations", "19"]
    code = main([*args, "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["classifier"]["n_a"] == 60
    assert payload["classifier"]["auc"] < 0.75
    assert payload["a"]["name"].endswith("(half 1)")


def test_missing_second_file_is_a_usage_error(pair, capsys):
    a, _ = pair
    assert main(["test", str(a), "--text", "text"]) == 2
    assert "second file" in capsys.readouterr().err


def test_unknown_field_names_the_record(pair, capsys):
    a, b = pair
    assert main(["test", str(a), str(b), "--text", "body"]) == 2
    assert "record 0 has no field 'body'" in capsys.readouterr().err
