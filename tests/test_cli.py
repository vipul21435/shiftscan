from __future__ import annotations

import json

import numpy as np
import pytest

from shiftscan.cli import main
from shiftscan.report import cell, num

from .conftest import ANIMALS, TOOLS, make_texts, write_jsonl

FAST = ["--max-records", "60", "--buckets", "16384"]
FAST_TESTS = [*FAST, "--permutations", "39", "--mmd-permutations", "99"]


@pytest.fixture
def pair(tmp_path):
    rng = np.random.default_rng(3)
    labels_a = list(rng.choice(["x", "y"], size=120, p=[0.8, 0.2]))
    labels_b = list(rng.choice(["x", "y"], size=120, p=[0.3, 0.7]))
    a = write_jsonl(tmp_path / "a.jsonl", make_texts(rng, 120, ANIMALS, 0.5), labels_a)
    b = write_jsonl(tmp_path / "b.jsonl", make_texts(rng, 120, TOOLS, 0.5), labels_b)
    return str(a), str(b)


def test_test_command_detects_a_planted_shift(pair, capsys):
    code = main(["test", *pair, "--text", "text", *FAST_TESTS])
    out = capsys.readouterr().out
    assert code == 0
    assert "AUC 1.000" in out
    assert "MMD, rbf kernel on TF-IDF" in out
    assert "Verdict: shift detected:" in out


def test_test_command_json_and_self_split(pair, capsys):
    args = ["test", pair[0], "--self-split", "--text", "text", *FAST_TESTS]
    code = main([*args, "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["tests"]["classifier"]["n_a"] == 60
    assert payload["tests"]["classifier"]["auc"] < 0.75
    assert payload["a"]["name"].endswith("(half 1)")
    assert payload["a"]["records"] == 60


def test_features_command_lists_planted_words_and_record_ids(pair, capsys):
    code = main(["features", *pair, "--text", "text", "--id", "id", *FAST, "--top", "3"])
    out = capsys.readouterr().out
    assert code == 0
    assert "## Features: weighted log-odds" in out
    assert "More typical of B:" in out
    assert "Most typical of A: `r" in out
    assert out.count("\n| `") >= 12


def test_lengths_and_labels_commands(pair, capsys):
    assert main(["lengths", *pair, "--text", "text"]) == 0
    out = capsys.readouterr().out
    assert "| whitespace tokens | 12 | 12 |" in out
    assert "length in characters, KS D" in out
    assert main(["labels", *pair, "--label", "label", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["labels"]["categories"] == ["x", "y"]
    assert payload["labels"]["g_p"] < 1e-6
    assert payload["labels"]["total_variation"] > 0.3
    assert payload["baseline"]["statistics"]["label_total_variation"]["ratio"] > 2


def test_report_command_markdown_and_json_to_a_file(pair, tmp_path, capsys):
    args = ["report", *pair, "--text", "text", "--label", "label", "--id", "id", *FAST_TESTS]
    assert main([*args, "--baseline-repeats", "1", "--top", "3"]) == 0
    out = capsys.readouterr().out
    for heading in (
        "# shiftscan report: a.jsonl vs b.jsonl",
        "## Two-sample tests",
        "## Against the split-half baseline",
        "## Lengths",
        "## Label shares",
        "## Unigram Jensen-Shannon divergence",
        "## Features: classifier weights",
        "## Records",
    ):
        assert heading in out
    target = tmp_path / "report.json"
    code = main([*args, "--baseline-repeats", "1", "--format", "json", "--out", str(target)])
    assert code == 0
    assert capsys.readouterr().out == ""
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["baseline"]["statistics"]["unigram_jsd"]["ratio"] > 3
    assert payload["settings"]["buckets"] == 16384


def test_same_file_twice_gets_distinct_names(pair, capsys):
    assert main(["lengths", pair[0], pair[0], "--text", "text"]) == 0
    assert "A:a.jsonl vs B:a.jsonl" in capsys.readouterr().out


def test_usage_errors_exit_with_2(pair, capsys):
    assert main(["test", pair[0], "--text", "text"]) == 2
    assert "second file" in capsys.readouterr().err
    assert main(["test", *pair, "--text", "body"]) == 2
    assert "record 0 has no field 'body'" in capsys.readouterr().err
    assert main(["test", *pair, "--self-split", "--text", "text"]) == 2
    assert "takes one file" in capsys.readouterr().err
    assert main(["test", *pair, "--text", "text", "--alpha", "2"]) == 2
    assert "--alpha" in capsys.readouterr().err
    assert main(["test", *pair, "--text", "text", "--folds", "0"]) == 2
    assert "--folds must be at least 1" in capsys.readouterr().err


def test_table_cells_escape_pipes_and_backticks():
    assert cell("a|b`c\nd") == "`a\\|b'c d`"
    assert num(None) == "n/a"
    assert num(0.123456) == "0.1235"
