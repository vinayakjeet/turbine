"""Request generation and timing.

TTFT is first token minus send. TPOT is last token minus first token,
divided by tokens after the first, matching the TPOT definition in the vLLM
protocol. Prompt tokens approximate to whitespace words; that is exact
enough for arrival shaping and irrelevant to the clock-derived metrics.

A finite request rate produces Poisson arrivals against a concurrency
ceiling, which is open-loop load with queueing. An infinite rate skips the
arrival gaps and the same ceiling becomes closed-loop batching. One code
path, both regimes.
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field

from turbine.backends.base import Backend, CompletionError
from turbine.clocks import Clock

WORDLIST = (
    "serving", "throughput", "latency", "prefix", "cache", "batch", "token",
    "quantize", "kernel", "decode", "prefill", "queue", "budget", "credit",
    "gpu", "clock", "stream", "request", "concurrent", "memory", "weight",
    "matrix", "scheduler", "worker", "engine", "vram",
)


@dataclass(frozen=True)
class Sample:
    index: int
    ttft_s: float | None
    total_s: float | None
    output_tokens: int
    tpot_s: float | None
    error: str = ""


@dataclass
class LoadResult:
    samples: list[Sample] = field(default_factory=list)
    started_at: float = 0.0
    ended_at: float = 0.0
    launched: int = 0
    skipped: int = 0

    @property
    def duration_s(self) -> float:
        return self.ended_at - self.started_at

    @property
    def completed(self) -> list[Sample]:
        return [s for s in self.samples if not s.error]


def build_prompts(
    seed: int, count: int, target_tokens: int, prefix_reuse: float = 0.0
) -> list[str]:
    """Deterministic prompt pool.

    With prefix_reuse > 0 a shared lead of exactly that fraction repeats
    across every request, which is what drives the server-side prefix cache;
    the remainder stays unique.
    """
    rng = random.Random(f"prompts:{seed}")
    shared_n = int(target_tokens * prefix_reuse)
    shared = [rng.choice(WORDLIST) for _ in range(shared_n)]
    prompts = []
    for _ in range(count):
        body = [rng.choice(WORDLIST) for _ in range(target_tokens - shared_n)]
        prompts.append(" ".join(shared + body))
    return prompts


async def run_load(
    backend: Backend,
    *,
    prompts: list[str],
    max_tokens: int,
    rate: float,
    concurrency: int,
    timeout_s: float,
    clock: Clock,
    cancel_event=None,
    seed: int = 0,
) -> LoadResult:
    result = LoadResult()
    result.started_at = clock.now()
    sem = asyncio.Semaphore(concurrency)

    rng = random.Random(f"arrivals:{seed}")
    delays = []
    t = 0.0
    for _ in prompts:
        if math.isinf(rate):
            delays.append(0.0)
        else:
            t += rng.expovariate(rate)
            delays.append(t)

    async def fire(index: int) -> None:
        await clock.sleep(delays[index])
        async with sem:
            if cancel_event is not None and cancel_event.is_set():
                result.skipped += 1
                return
            result.launched += 1
            sample = await timed_request(
                backend,
                prompt=prompts[index],
                index=index,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                clock=clock,
                request_seed=seed + index,
            )
            result.samples.append(sample)

    tasks = [asyncio.create_task(fire(i)) for i in range(len(prompts))]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
    result.ended_at = clock.now()
    return result


async def timed_request(
    backend: Backend,
    *,
    prompt: str,
    index: int,
    max_tokens: int,
    timeout_s: float,
    clock: Clock,
    request_seed: int,
) -> Sample:
    t_send = clock.now()
    first: float | None = None
    last = t_send
    tokens = 0
    try:
        generator = backend.stream(prompt, max_tokens=max_tokens, request_seed=request_seed)
        try:
            async for _chunk in generator:
                now = clock.now()
                if first is None:
                    first = now
                last = now
                tokens += 1
                if now - t_send > timeout_s:
                    raise CompletionError(f"exceeded {timeout_s:g}s deadline")
        finally:
            await generator.aclose()
    except CompletionError as exc:
        return Sample(index, first, None, tokens, None, error=str(exc))
    if first is None or tokens == 0:
        return Sample(index, None, None, 0, None, error="empty completion")
    tpot = (last - first) / (tokens - 1) if tokens > 1 else 0.0
    return Sample(index, first - t_send, last - t_send, tokens, tpot)
