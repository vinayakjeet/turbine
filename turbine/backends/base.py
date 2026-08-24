"""Backend interface: a token stream and nothing else.

Load generation, scoring and reporting are written against this protocol,
so the deterministic mock and a live vLLM server exercise identical code
paths and the mock cannot drift from what measurement actually does.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from turbine.clocks import Clock, RealClock


class CompletionError(RuntimeError):
    """One failed request inside a measured window. Counted, never retried:
    a retry would quietly replace the quantity being measured."""


@runtime_checkable
class Backend(Protocol):
    name: str

    def stream(
        self, prompt: str, *, max_tokens: int, request_seed: int
    ) -> AsyncIterator[str]: ...

    async def aclose(self) -> None: ...


def default_clock() -> Clock:
    return RealClock()
