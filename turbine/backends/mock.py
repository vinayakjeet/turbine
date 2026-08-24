"""Deterministic inference simulator.

Exists so the whole pipeline, prompt construction through report rendering,
runs on a laptop in about a minute and lands on byte-identical numbers every
time. Its latency profiles are invented: they preserve the orderings real
hardware is expected to show (A10G ahead of T4, quantized prefill paying a
dequantization tax when VRAM is not the binding constraint) but the
magnitudes mean nothing about GPUs.

Token timing advances in batches of `token_batch` tokens rather than one
sleep per token; the yielded token count and the clock-derived totals stay
exact, only sub-batch jitter is smoothed away.

Every result produced here carries backend="mock" end to end, and the
report layer refuses mock results anywhere a published number is expected.
The point is verifying plumbing, not previewing hardware.
"""

from __future__ import annotations

import hashlib
import random
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass

from turbine.backends.base import CompletionError
from turbine.clocks import Clock, VirtualClock

HIT_STRENGTH = 0.6
CACHE_CAPACITY = 64
QUEUE_PER_EXTRA = 0.12


@dataclass(frozen=True)
class Profile:
    base_ttft_s: float
    per_token_s: float


_PROFILES: dict[str, Profile] = {
    "T4": Profile(base_ttft_s=0.180, per_token_s=0.0160),
    "A10G": Profile(base_ttft_s=0.110, per_token_s=0.0090),
    "A100-40G": Profile(base_ttft_s=0.070, per_token_s=0.0055),
}

# AWQ decode gains from lighter weight reads; prefill pays dequantization
# because nothing here is VRAM-bound.
_QUANT_TTFT_MULTIPLIER = 1.25
_QUANT_TOKEN_MULTIPLIER = 0.92


def profile_for(tier: str, precision: str) -> Profile:
    base = _PROFILES[tier]
    if precision == "fp16":
        return base
    return Profile(
        base_ttft_s=base.base_ttft_s * _QUANT_TTFT_MULTIPLIER,
        per_token_s=base.per_token_s * _QUANT_TOKEN_MULTIPLIER,
    )


class MockBackend:
    name = "mock"

    def __init__(
        self,
        *,
        tier: str = "A10G",
        precision: str = "awq",
        concurrency: int = 1,
        seed: int = 0,
        failure_rate: float = 0.0,
        prefix_cache: bool = True,
        prefix_reuse: float = 0.0,
        token_batch: int = 16,
        clock: Clock | None = None,
    ) -> None:
        self._clock = clock or VirtualClock()
        self._profile = profile_for(tier, precision)
        self._seed = seed
        self._failure_rate = failure_rate
        self._prefix_cache = prefix_cache
        self._prefix_reuse = prefix_reuse
        self._token_batch = max(1, token_batch)
        self._queue = 1.0 + QUEUE_PER_EXTRA * (concurrency - 1)
        self._cache: OrderedDict[str, None] = OrderedDict()

    async def stream(
        self, prompt: str, *, max_tokens: int, request_seed: int
    ) -> AsyncIterator[str]:
        rng = random.Random(f"{self._seed}:{request_seed}")
        ttft = self._profile.base_ttft_s * self._queue * rng.uniform(0.9, 1.1)

        if self._prefix_cache:
            # Two hit sources, in the spirit of a block-level KV cache: an
            # exact repeat of an earlier prompt (retries, templated traffic)
            # and the shared-lead fraction of a reuse-heavy workload. The
            # discount scales with the shared fraction, which is what makes
            # the reuse sweep produce a crossover instead of a plateau.
            key = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).hexdigest()
            exact_hit = key in self._cache
            self._cache[key] = None
            self._cache.move_to_end(key)
            while len(self._cache) > CACHE_CAPACITY:
                self._cache.popitem(last=False)
            shared = max(1.0 if exact_hit else 0.0, self._prefix_reuse)
            if shared > 0:
                ttft *= 1 - HIT_STRENGTH * shared

        await self._clock.sleep(ttft)

        fail_at = -1
        if self._failure_rate > 0 and rng.random() < self._failure_rate:
            fail_at = rng.randint(max(1, max_tokens // 4), max_tokens)

        token_delay = self._profile.per_token_s * self._queue
        produced = 0
        while produced < max_tokens:
            batch = min(self._token_batch, max_tokens - produced)
            await self._clock.sleep(token_delay * batch * rng.uniform(0.85, 1.15))
            for _ in range(batch):
                produced += 1
                yield f"tok{produced} "
                if produced == fail_at:
                    raise CompletionError("simulated provider failure mid-stream")

    async def aclose(self) -> None:
        pass
