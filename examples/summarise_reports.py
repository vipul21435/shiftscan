"""Print one Markdown table row per JSON report written by ``shiftscan report --format json``.

    python examples/summarise_reports.py reports/*.json

The columns are the ones quoted in the README: held-out AUC and its permutation p-value,
the MMD p-value, and the unigram Jensen-Shannon divergence between the datasets and between
halves of one dataset at the same size.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def row(path: Path) -> str:
    report = json.loads(path.read_text(encoding="utf-8"))
    classifier = report["tests"]["classifier"]
    mmd = report["tests"]["mmd"]
    stats = report["baseline"]["statistics"]
    jsd, auc = stats["unigram_jsd"], stats["classifier_auc"]
    cells = [
        path.stem,
        f"{classifier['auc']:.3f}",
        f"{classifier['permutation_p']:.4f}",
        f"{auc['cross']:.3f} / {auc['baseline']:.3f}",
        f"{mmd['mmd2']:.2e}",
        f"{mmd['p_value']:.4f}",
        f"{jsd['cross']:.4f} / {jsd['baseline']:.4f}",
        f"{jsd['ratio']:.2f}",
    ]
    return "| " + " | ".join(cells) + " |"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    header = [
        "report",
        "AUC",
        "permutation p",
        "AUC between / halves",
        "MMD²",
        "MMD p",
        "JSD between / halves",
        "JSD ratio",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(" --- " for _ in header) + "|")
    for name in argv[1:]:
        print(row(Path(name)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
