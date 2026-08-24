"""Clocks.

The mock pipeline runs on a virtual clock so recorded latencies are exactly
the simulated ones, independent of how fast the host executes the harness.
Real measurements use wall time. Everything downstream of a Clock is
agnostic to which one it got.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class RealClock:
    def now(self) -> float:
        return time.perf_counter()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class VirtualClock:
    """Discrete-event clock for deterministic simulations.

    Sleepers register an absolute wake time and the clock jumps forward only
    as far as the earliest pending wakeup, so concurrent sleeps overlap the
    way they would in reality: a request queued behind others waits its turn
    instead of absorbing their whole service time. Recorded latencies are
    still exactly the simulated ones, byte-identical across machines.

    It remains one shared timeline. Anything that must not be moved by other
    sleepers, like the billing watchdog, belongs on RealClock.
    """

    def __init__(self) -> None:
        self._now = 0.0
        self._pending: list[float] = []

    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        wake = self._now + max(seconds, 0.0)
        if wake == self._now:
            await asyncio.sleep(0)
            return
        heapq.heappush(self._pending, wake)
        try:
            while self._now < wake:
                self._now = self._pending[0]
                await asyncio.sleep(0)
        finally:
            self._pending.remove(wake)
