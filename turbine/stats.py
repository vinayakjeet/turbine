"""Summary statistics.

Pure functions only, so the report's honesty gates are testable without a
single measurement. Variance is sample stdev throughout: reps are a sample
of what the hardware does, not the whole population.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of nothing")
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("variance needs at least two samples")
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between closest ranks, numpy's default."""
    if not values:
        raise ValueError("percentile of nothing")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile {q} outside [0, 100]")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    stdev: float
    p50: float
    p95: float
    p99: float


def summarize(values: list[float]) -> Summary:
    return Summary(
        n=len(values),
        mean=mean(values),
        stdev=stdev(values),
        p50=percentile(values, 50),
        p95=percentile(values, 95),
        p99=percentile(values, 99),
    )


@dataclass(frozen=True)
class RepStats:
    duration_s: float
    requests: int
    errors: int
    output_tokens: int
    tok_per_s: float
    ttft_s: Summary
    tpot_s: Summary


def cross_rep(values: list[float]) -> tuple[float, float]:
    """Mean plus sample stdev across reps, the shape every table cell takes."""
    return round(mean(values), 2), round(stdev(values), 4)
