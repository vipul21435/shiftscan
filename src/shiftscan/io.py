"""Read JSON lines, CSV or Parquet files into a list of records."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class InputError(ValueError):
    """A file could not be read as asked; the message names the file and the row."""


@dataclass
class Dataset:
    name: str
    ids: list[str]
    texts: list[str]
    labels: list[str] | None = None

    def __len__(self) -> int:
        return len(self.texts)

    def take(self, indices: Sequence[int] | np.ndarray, name: str | None = None) -> Dataset:
        picked = [int(i) for i in indices]
        return Dataset(
            name=name or self.name,
            ids=[self.ids[i] for i in picked],
            texts=[self.texts[i] for i in picked],
            labels=None if self.labels is None else [self.labels[i] for i in picked],
        )


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson", ".jsonlines"):
        return "jsonl"
    if suffix in (".csv", ".tsv"):
        return suffix[1:]
    if suffix in (".parquet", ".pq"):
        return "parquet"
    raise InputError(f"{path}: cannot tell the format from the suffix; pass --input-format")


def _rows_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}:{number}: not valid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise InputError(f"{path}:{number}: expected a JSON object on every line")
            rows.append(row)
    return rows


def _rows_csv(path: Path, delimiter: str) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _rows_parquet(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise InputError(
            f"{path}: reading Parquet needs pyarrow (pip install 'shiftscan[parquet]')"
        ) from exc
    return pq.read_table(path).to_pylist()


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)


def read_dataset(
    path: str | Path,
    text_fields: Sequence[str],
    label_field: str | None = None,
    id_field: str | None = None,
    input_format: str | None = None,
    name: str | None = None,
) -> Dataset:
    """Load one file.

    Several text fields are joined with a newline. A null text counts as empty.
    Without ``id_field`` the id of a record is its 0-based row number in the file.
    """
    path = Path(path)
    if not text_fields:
        raise InputError("name at least one text field")
    if not path.is_file():
        raise InputError(f"{path}: no such file")
    kind = input_format or _detect_format(path)
    if kind == "jsonl":
        rows = _rows_jsonl(path)
    elif kind == "csv":
        rows = _rows_csv(path, ",")
    elif kind == "tsv":
        rows = _rows_csv(path, "\t")
    elif kind == "parquet":
        rows = _rows_parquet(path)
    else:
        raise InputError(f"unknown input format {kind!r}")
    if not rows:
        raise InputError(f"{path}: no records")
    needed = list(text_fields) + [f for f in (label_field, id_field) if f]
    ids: list[str] = []
    texts: list[str] = []
    labels: list[str] | None = [] if label_field else None
    for number, row in enumerate(rows):
        for column in needed:
            if column not in row:
                raise InputError(f"{path}: record {number} has no field {column!r}")
        texts.append("\n".join(_as_text(row[column]) for column in text_fields))
        ids.append(_as_text(row[id_field]) if id_field else str(number))
        if labels is not None and label_field:
            labels.append(_as_text(row[label_field]))
    return Dataset(name=name or path.name, ids=ids, texts=texts, labels=labels)
