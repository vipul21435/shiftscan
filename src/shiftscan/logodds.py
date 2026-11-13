"""Weighted log-odds ratio with an informative Dirichlet prior (Monroe et al., 2008).

For a word ``w`` with counts ``y_a`` and ``y_b`` in corpora of ``n_a`` and ``n_b``
words, and prior pseudo-counts ``alpha_w`` summing to ``alpha_0``::

    delta_w = log (y_b + alpha_w) / (n_b + alpha_0 - y_b - alpha_w)
            - log (y_a + alpha_w) / (n_a + alpha_0 - y_a - alpha_w)
    var_w   = 1 / (y_b + alpha_w) + 1 / (y_a + alpha_w)
    z_w     = delta_w / sqrt(var_w)

The prior is "informative" because ``alpha_w`` is proportional to how common the
word is overall: ``alpha_w = alpha_0 * (y_a + y_b) / (n_a + n_b)``. It pulls rare
words towards no difference, and dividing by the standard deviation stops them
from topping the list on a handful of occurrences. Positive ``z`` means the word
is more typical of side B.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass
class LogOdds:
    feature: str
    count_a: int
    count_b: int
    log_odds: float
    z: float

    def to_dict(self) -> dict:
        return asdict(self)


def weighted_log_odds(
    counts_a: Mapping[str, float],
    counts_b: Mapping[str, float],
    prior_total: float = 1000.0,
    min_count: int = 1,
) -> list[LogOdds]:
    """Score every feature; the list is sorted from most typical of B to most typical of A.

    ``prior_total`` is ``alpha_0``. Features with fewer than ``min_count`` occurrences
    over both sides are scored like the rest but left out of the returned list.
    """
    if prior_total <= 0:
        raise ValueError("prior_total must be positive")
    n_a, n_b = sum(counts_a.values()), sum(counts_b.values())
    if n_a <= 0 or n_b <= 0:
        raise ValueError("both sides need at least one count")
    rows = []
    for feature in set(counts_a) | set(counts_b):
        y_a, y_b = counts_a.get(feature, 0), counts_b.get(feature, 0)
        if y_a + y_b < min_count:
            continue
        alpha = prior_total * (y_a + y_b) / (n_a + n_b)
        odds_b = (y_b + alpha) / (n_b + prior_total - y_b - alpha)
        odds_a = (y_a + alpha) / (n_a + prior_total - y_a - alpha)
        delta = math.log(odds_b) - math.log(odds_a)
        variance = 1.0 / (y_b + alpha) + 1.0 / (y_a + alpha)
        rows.append(LogOdds(feature, int(y_a), int(y_b), delta, delta / math.sqrt(variance)))
    rows.sort(key=lambda row: (-row.z, row.feature))
    return rows
