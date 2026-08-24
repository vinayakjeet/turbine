import pytest

from turbine.backends.mock import MockBackend, profile_for
from turbine.clocks import VirtualClock
from turbine.loadgen import build_prompts, run_load


async def collect(backend, prompt="hello world", max_tokens=16, seed=1):
    chunks = []
    async for chunk in backend.stream(prompt, max_tokens=max_tokens, request_seed=seed):
        chunks.append(chunk)
    return "".join(chunks)


def make(seed=7, **kw) -> MockBackend:
    return MockBackend(clock=VirtualClock(), seed=seed, **kw)


async def test_stream_yields_requested_token_count():
    out = await collect(make(), max_tokens=32)
    assert len(out.split()) == 32


async def test_same_seed_reproduces_byte_identical_output():
    a = await collect(make(), max_tokens=64, seed=3)
    b = await collect(make(), max_tokens=64, seed=3)
    assert a == b


async def test_prefix_cache_hit_cuts_ttft():
    clock = VirtualClock()
    backend = MockBackend(clock=clock, seed=5)
    shared = " ".join(["shared"] * 64)
    unique = " ".join(["solo"] * 64)

    async def ttft(prompt: str, seed: int) -> float:
        gen = backend.stream(prompt, max_tokens=1, request_seed=seed)
        start = clock.now()
        agen = gen.__aiter__()
        await agen.__anext__()
        await gen.aclose()
        return clock.now() - start

    cold = await ttft(shared, 1)
    await ttft(unique, 2)
    warm = await ttft(shared, 3)
    assert warm < cold


async def test_reuse_fraction_drives_a_monotone_crossover():
    async def median_ttft(reuse: float) -> float:
        clock = VirtualClock()
        backend = MockBackend(clock=clock, seed=9, prefix_reuse=reuse)
        prompts = build_prompts(42, 24, 128, prefix_reuse=reuse)
        times = []
        for i, p in enumerate(prompts):
            gen = backend.stream(p, max_tokens=1, request_seed=i)
            start = clock.now()
            await gen.__aiter__().__anext__()
            times.append(clock.now() - start)
            await gen.aclose()
        from turbine.stats import percentile

        return percentile(times, 50)

    zero = await median_ttft(0.0)
    half = await median_ttft(0.5)
    full = await median_ttft(1.0)
    assert full < half < zero


async def test_failure_rate_raises_midstream_and_is_counted_by_loader():
    backend = make(failure_rate=1.0)
    result = await run_load(
        backend,
        prompts=["p"] * 4,
        max_tokens=32,
        rate=float("inf"),
        concurrency=2,
        timeout_s=60,
        clock=VirtualClock(),
    )
    assert len(result.completed) == 0
    assert all("simulated provider failure" in s.error for s in result.samples)


def test_profile_ordering_matches_expected_hardware_shapes():
    t4, a10g = profile_for("T4", "fp16"), profile_for("A10G", "fp16")
    assert a10g.per_token_s < t4.per_token_s
    fp16, awq = profile_for("A10G", "fp16"), profile_for("A10G", "awq")
    assert awq.base_ttft_s > fp16.base_ttft_s, "quantized prefill pays dequantization tax"
    assert awq.per_token_s < fp16.per_token_s, "lighter weight reads on decode"


def test_unknown_tier_fails_loudly():
    with pytest.raises(KeyError):
        profile_for("H200", "fp16")


async def test_concurrency_never_exceeds_the_ceiling():
    in_flight = 0
    peak = 0

    class Probe(MockBackend):
        async def stream(self, prompt, *, max_tokens, request_seed):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                async for chunk in super().stream(
                    prompt, max_tokens=max_tokens, request_seed=request_seed
                ):
                    yield chunk
            finally:
                in_flight -= 1

    backend = Probe(clock=VirtualClock())
    await run_load(
        backend,
        prompts=["x"] * 24,
        max_tokens=8,
        rate=float("inf"),
        concurrency=5,
        timeout_s=60,
        clock=VirtualClock(),
    )
    assert peak <= 5
