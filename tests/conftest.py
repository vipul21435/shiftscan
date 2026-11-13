from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ANIMALS = "cat dog horse sheep goat cow hen duck goose pig mouse rabbit fox owl deer".split()
TOOLS = "hammer saw drill wrench chisel plane file clamp vice lathe anvil rasp awl level".split()
COMMON = "the a of and to in is was it for on with as at by from this that".split()


def make_texts(
    rng: np.random.Generator, n: int, topic: list[str], share: float, length: int = 12
) -> list[str]:
    """Texts of ``length`` words; each word comes from ``topic`` with probability ``share``."""
    texts = []
    for _ in range(n):
        words = []
        for _ in range(length):
            source = topic if rng.random() < share else COMMON
            words.append(source[rng.integers(len(source))])
        texts.append(" ".join(words))
    return texts


def write_jsonl(path: Path, texts: list[str], labels: list[str] | None = None) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for i, text in enumerate(texts):
            row = {"id": f"r{i}", "text": text}
            if labels is not None:
                row["label"] = labels[i]
            handle.write(json.dumps(row) + "\n")
    return path


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
