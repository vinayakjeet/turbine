"""GPU tier table with prices.

Transcribed from Modal's published list prices on 2026-08-24. Re-check them
before any real run: a stale price silently skews every tok/s/$ figure and
the break-even column built on top of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

PRICE_SOURCE_DATE = date(2026, 8, 24)


@dataclass(frozen=True)
class GpuTier:
    name: str
    vram_gb: int
    usd_per_hour: Decimal


TIERS: dict[str, GpuTier] = {
    "T4": GpuTier(name="T4", vram_gb=16, usd_per_hour=Decimal("0.59")),
    "A10G": GpuTier(name="A10G", vram_gb=24, usd_per_hour=Decimal("1.10")),
    "A100-40G": GpuTier(name="A100-40G", vram_gb=40, usd_per_hour=Decimal("2.10")),
}


def tier(name: str) -> GpuTier:
    try:
        return TIERS[name]
    except KeyError:
        known = ", ".join(sorted(TIERS))
        raise ValueError(f"unknown GPU tier {name!r}, have: {known}") from None


def cost(gpu_hours: float | Decimal, name: str) -> Decimal:
    price = tier(name).usd_per_hour * Decimal(str(gpu_hours))
    return price.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
