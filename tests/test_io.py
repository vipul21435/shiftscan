from __future__ import annotations

import json

import pytest

from shiftscan.io import Dataset, InputError, read_dataset


def test_reads_json_lines_with_ids_labels_and_blank_lines(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        '{"id": "x1", "q": "one", "c": "two", "y": "pos"}\n\n'
        '{"id": 7, "q": null, "c": {"k": 1}, "y": 0}\n',
        encoding="utf-8",
    )
    data = read_dataset(path, ["q", "c"], label_field="y", id_field="id")
    assert len(data) == 2
    assert data.ids == ["x1", "7"]
    assert data.texts == ["one\ntwo", '\n{"k": 1}']
    assert data.labels == ["pos", "0"]
    assert data.name == "d.jsonl"


def test_row_number_is_the_default_id(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"t": "a"}\n{"t": "b"}\n', encoding="utf-8")
    data = read_dataset(path, ["t"])
    assert data.ids == ["0", "1"]
    assert data.labels is None


def test_reads_csv_and_tsv(tmp_path):
    csv_path = tmp_path / "d.csv"
    csv_path.write_text('text,label\n"hello, world",a\nbye,b\n', encoding="utf-8")
    data = read_dataset(csv_path, ["text"], label_field="label")
    assert data.texts == ["hello, world", "bye"]
    assert data.labels == ["a", "b"]
    tsv_path = tmp_path / "d.tsv"
    tsv_path.write_text("text\tn\nhello\t1\n", encoding="utf-8")
    assert read_dataset(tsv_path, ["text"]).texts == ["hello"]


def test_reads_parquet_with_the_optional_dependency(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "d.parquet"
    pq.write_table(pa.table({"text": ["a b", "c"], "label": [1, 0]}), path)
    data = read_dataset(path, ["text"], label_field="label")
    assert data.texts == ["a b", "c"]
    assert data.labels == ["1", "0"]


def test_format_override(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text('{"t": "a"}\n', encoding="utf-8")
    with pytest.raises(InputError, match="cannot tell the format"):
        read_dataset(path, ["t"])
    assert read_dataset(path, ["t"], input_format="jsonl").texts == ["a"]
    with pytest.raises(InputError, match="unknown input format"):
        read_dataset(path, ["t"], input_format="xml")


def test_errors_name_the_file_and_the_row(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"t": "a"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(InputError, match=r"bad\.jsonl:2: not valid JSON"):
        read_dataset(bad, ["t"])
    array = tmp_path / "array.jsonl"
    array.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(InputError, match="expected a JSON object"):
        read_dataset(array, ["t"])
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(InputError, match="no records"):
        read_dataset(empty, ["t"])
    with pytest.raises(InputError, match="no such file"):
        read_dataset(tmp_path / "missing.jsonl", ["t"])
    ok = tmp_path / "ok.jsonl"
    ok.write_text(json.dumps({"t": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(InputError, match="record 0 has no field 'label'"):
        read_dataset(ok, ["t"], label_field="label")
    with pytest.raises(InputError, match="at least one text field"):
        read_dataset(ok, [])


def test_take_keeps_ids_texts_and_labels_together():
    data = Dataset("d", ["a", "b", "c"], ["x", "y", "z"], ["0", "1", "0"])
    part = data.take([2, 0], name="part")
    assert (part.name, part.ids) == ("part", ["c", "a"])
    assert (part.texts, part.labels) == (["z", "x"], ["0", "0"])


def test_undecodable_and_damaged_files_are_input_errors(tmp_path):
    latin = tmp_path / "latin.jsonl"
    latin.write_bytes(b'{"t": "caf\xe9"}\n')
    with pytest.raises(InputError, match="not valid UTF-8"):
        read_dataset(latin, ["t"])
    pytest.importorskip("pyarrow")
    broken = tmp_path / "broken.parquet"
    broken.write_bytes(b"this is not a parquet file")
    with pytest.raises(InputError, match="cannot read as Parquet"):
        read_dataset(broken, ["t"])


def test_csv_rows_with_missing_cells_read_as_empty_text(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("text,label\nfull,a\nonly\n", encoding="utf-8")
    data = read_dataset(path, ["text"], label_field="label")
    assert data.texts == ["full", "only"]
    assert data.labels == ["a", ""]
