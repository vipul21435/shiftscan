"""Write two small synthetic files with a planted shift, for trying the command line.

Side A is support-style messages about billing, side B about login problems; both
share the same filler words. Run from the repository root:

    python examples/make_example_data.py
    shiftscan test examples/data/billing.jsonl examples/data/login.jsonl --text text
"""

from __future__ import annotations

import json
import random
from pathlib import Path

FILLER = (
    "hello i have a question about my account and it has been two days since "
    "please can you help me with this thanks the app is on my phone".split()
)
BILLING = "invoice refund charge card payment receipt billing subscription price renewal".split()
LOGIN = "password login reset email code locked username signin verify device".split()


def make_rows(rng: random.Random, topic: list[str], prefix: str, n: int) -> list[dict]:
    rows = []
    for i in range(n):
        words = [
            rng.choice(topic) if rng.random() < 0.25 else rng.choice(FILLER) for _ in range(14)
        ]
        label = "urgent" if rng.random() < (0.3 if prefix == "b" else 0.2) else "normal"
        rows.append({"id": f"{prefix}{i:03d}", "text": " ".join(words), "label": label})
    return rows


def main() -> None:
    rng = random.Random(0)
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    for name, topic, prefix in (("billing", BILLING, "a"), ("login", LOGIN, "b")):
        rows = make_rows(rng, topic, prefix, 200)
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"wrote {out / (name + '.jsonl')} ({len(rows)} records)")


if __name__ == "__main__":
    main()
