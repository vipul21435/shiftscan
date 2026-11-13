"""Use shiftscan from Python on the synthetic example files.

    python examples/make_example_data.py
    python examples/api_example.py
"""

from __future__ import annotations

from pathlib import Path

from shiftscan.io import read_dataset
from shiftscan.scan import Options, full_report

DATA = Path(__file__).parent / "data"


def main() -> None:
    a = read_dataset(DATA / "billing.jsonl", ["text"], label_field="label", id_field="id")
    b = read_dataset(DATA / "login.jsonl", ["text"], label_field="label", id_field="id")
    report = full_report(a, b, Options(seed=0, permutations=50, mmd_permutations=200, top=5))

    tests = report["tests"]
    print(tests["verdict"])
    print(f"held-out AUC      {tests['classifier']['auc']:.3f}")
    print(f"MMD p-value       {tests['mmd']['p_value']:.4f}")
    row = report["baseline"]["statistics"]["unigram_jsd"]
    print(f"unigram JSD       {row['cross']:.4f} between, {row['baseline']:.4f} within halves")
    print(f"label p-value     {report['labels']['g_p']:.3f}")
    words = [item["feature"] for item in report["features"]["log_odds"]["more_in_b"]]
    print("more typical of B:", ", ".join(words))
    ids = [item["id"] for item in report["features"]["records"]["typical_of_b"]]
    print("records typical of B:", ", ".join(ids))


if __name__ == "__main__":
    main()
