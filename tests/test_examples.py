from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np

from shiftscan.features import FeatureConfig
from shiftscan.io import Dataset, read_dataset
from shiftscan.scan import Options, full_report

from .conftest import ANIMALS, TOOLS, make_texts

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flatten_writes_one_row_per_question(tmp_path):
    document = {
        "data": [
            {
                "title": "Rivers",
                "paragraphs": [
                    {
                        "context": "The river is long.",
                        "qas": [
                            {"id": "q1", "question": "How long is it?"},
                            {"id": "q2", "question": "Where does it end?"},
                        ],
                    }
                ],
            },
            {"paragraphs": [{"context": "x", "qas": [{"id": 3, "question": "Why?"}]}]},
        ]
    }
    source = tmp_path / "in.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    target = tmp_path / "out.jsonl"
    assert load("fetch_real_data").flatten(source, target) == 3
    data = read_dataset(target, ["question"], id_field="id")
    assert data.ids == ["q1", "q2", "3"]
    assert data.texts[2] == "Why?"


def test_example_rows_are_reproducible():
    module = load("make_example_data")
    first = module.make_rows(random.Random(0), module.BILLING, "a", 5)
    second = module.make_rows(random.Random(0), module.BILLING, "a", 5)
    assert first == second
    assert [row["id"] for row in first] == ["a000", "a001", "a002", "a003", "a004"]
    assert {row["label"] for row in first} <= {"urgent", "normal"}


def test_summary_row_quotes_the_report_numbers(tmp_path, capsys):
    rng = np.random.default_rng(8)
    a = Dataset("a", [str(i) for i in range(60)], make_texts(rng, 60, ANIMALS, 0.4))
    b = Dataset("b", [str(i) for i in range(60)], make_texts(rng, 60, TOOLS, 0.4))
    options = Options(
        max_records=40,
        permutations=9,
        mmd_permutations=19,
        baseline_repeats=1,
        features=FeatureConfig(n_buckets=2**12),
    )
    report = full_report(a, b, options)
    path = tmp_path / "pair.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    module = load("summarise_reports")
    assert module.main(["summarise_reports.py", str(path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("| report | AUC |")
    assert lines[2].startswith(f"| pair | {report['tests']['classifier']['auc']:.3f} |")
    assert module.main(["summarise_reports.py"]) == 2
