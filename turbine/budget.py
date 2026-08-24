"""The credit guard.

Runs are estimated before launch, refused when the estimate would land
inside the safety margin, and cancelled mid-flight when their allotment runs
out. A harness that can spend a whole month's credit on one stuck run is
not a benchmark.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from decimal import Decimal

from turbine.clocks import Clock


class BudgetRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class Budget:
    cap_usd: Decimal
    margin_usd: Decimal

    def usable(self) -> Decimal:
        return self.cap_usd - self.margin_usd


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    remaining_after_estimate: Decimal


def plan(budget: Budget, spent: Decimal, estimate_usd: Decimal) -> Decision:
    if estimate_usd <= 0:
        raise ValueError("run estimate must be positive; free work needs no guard")
    remaining = budget.usable() - spent - estimate_usd
    if remaining < 0:
        return Decision(
            allowed=False,
            reason=(
                f"refused: estimate ${estimate_usd} against "
                f"${budget.usable() - spent} usable "
                f"(cap ${budget.cap_usd}, margin ${budget.margin_usd}, "
                f"spent ${spent})"
            ),
            remaining_after_estimate=remaining,
        )
    return Decision(
        allowed=True,
        reason=f"allowed: ${remaining} left inside usable budget after estimate",
        remaining_after_estimate=remaining,
    )


def allotment_seconds(budget: Budget, spent: Decimal, usd_per_hour: Decimal) -> float:
    """Seconds this run may bill before the watchdog cancels it."""
    usable = budget.usable() - spent
    if usable <= 0 or usd_per_hour <= 0:
        return 0.0
    return float(usable / usd_per_hour) * 3600.0


class RunWatchdog:
    """Cooperative kill switch for one measured run.

    Runs on wall time even when the measurement itself runs on a virtual
    clock, because billing accrues in wall hours either way. A virtual clock
    would also be wrong here by construction: its sleeps advance one shared
    timeline, so a long watchdog sleep would fast-forward every measured
    latency in the process.

    Cooperative because killing a request mid-stream corrupts the timing it
    belongs to; instead the load generator stops launching new requests when
    the event sets, finishes what is in flight, and the run is archived as
    terminated rather than published as complete. A limit of exactly zero
    means already exhausted: useful for exercising the termination path
    without waiting on any clock.
    """

    def __init__(self, limit_s: float, clock: Clock) -> None:
        self.cancel_event = asyncio.Event()
        self.timed_out = False
        self._clock = clock
        if limit_s == 0:
            self.timed_out = True
            self.cancel_event.set()
            self._task = None
        elif limit_s > 0 and limit_s != float("inf"):
            self._task = asyncio.create_task(self._fire(limit_s))
        else:
            self._task = None

    async def _fire(self, limit_s: float) -> None:
        await self._clock.sleep(limit_s)
        self.timed_out = True
        self.cancel_event.set()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
