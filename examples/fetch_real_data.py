"""Download the public files behind the README numbers and flatten the SQuAD-style ones.

    python examples/fetch_real_data.py data/

writes, under the directory given:

- ``mnli_validation_matched.parquet`` and ``mnli_validation_mismatched.parquet``
  (MultiNLI validation sets, as published on the Hugging Face hub by the dataset authors'
  group), read directly by shiftscan with the ``parquet`` extra;
- ``squad_dev.jsonl``, ``new_wiki.jsonl``, ``nyt.jsonl``, ``reddit.jsonl`` and ``amazon.jsonl``:
  one record per question (``id``, ``question``, ``title``) from the SQuAD v1.1 development
  set and the four SQuADShifts test sets.

Nothing is downloaded twice. The files stay on your disk; this repository holds no rows.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co/datasets/nyu-mll/multi_nli/resolve/main/data/"
SHIFTS = "https://raw.githubusercontent.com/modestyachts/squadshifts-website/master/datasets/"
PARQUET = {
    "mnli_validation_matched.parquet": HUB + "validation_matched-00000-of-00001.parquet",
    "mnli_validation_mismatched.parquet": HUB + "validation_mismatched-00000-of-00001.parquet",
}
SQUAD_STYLE = {
    "squad_dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    "new_wiki": SHIFTS + "new_wiki_v1.0.json",
    "nyt": SHIFTS + "nyt_v1.0.json",
    "reddit": SHIFTS + "reddit_v1.0.json",
    "amazon": SHIFTS + "amazon_reviews_v1.0.json",
}


def download(url: str, target: Path) -> None:
    if target.exists():
        return
    request = urllib.request.Request(url, headers={"User-Agent": "shiftscan-example"})
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())
    print(f"downloaded {target.name} ({target.stat().st_size} bytes)")


def flatten(source: Path, target: Path) -> int:
    """One JSON line per question of a SQuAD-format file."""
    document = json.loads(source.read_text(encoding="utf-8"))
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for article in document["data"]:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    row = {
                        "id": str(qa["id"]),
                        "question": qa["question"],
                        "title": article.get("title", ""),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
    return count


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    out = Path(argv[1])
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in PARQUET.items():
        download(url, out / name)
    for name, url in SQUAD_STYLE.items():
        download(url, raw / f"{name}.json")
        count = flatten(raw / f"{name}.json", out / f"{name}.jsonl")
        print(f"{name}.jsonl: {count} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
