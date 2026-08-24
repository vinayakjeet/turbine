
from turbine.backends.mock import MockBackend
from turbine.clocks import VirtualClock
from turbine.loadgen import build_prompts, run_load


def test_prompts_are_deterministic_per_seed():
    a = build_prompts(11, 5, 32)
    b = build_prompts(11, 5, 32)
    c = build_prompts(12, 5, 32)
    assert a == b
    assert a != c


def test_prefix_reuse_shares_exact_fraction():
    reuse = build_prompts(3, 10, 100, prefix_reuse=0.75)
    no_reuse = build_prompts(3, 10, 100, prefix_reuse=0.0)
    shared = " ".join(reuse[0].split()[:75])
    for p in reuse:
        assert " ".join(p.split()[:75]) == shared
    first_words = {p.split()[0] for p in no_reuse}
    assert len(first_words) > 1


async def test_open_loop_respects_rate_and_concurrency_and_finishes_all():
    clock = VirtualClock()
    backend = MockBackend(clock=clock)
    prompts = ["x"] * 30
    result = await run_load(
        backend,
        prompts=prompts,
        max_tokens=8,
        rate=50.0,
        concurrency=4,
        timeout_s=60,
        clock=clock,
    )
    assert len(result.samples) == 30
    assert len(result.completed) == 30
    assert result.duration_s > 0


async def test_preset_cancel_event_launches_nothing():
    import asyncio

    clock = VirtualClock()
    backend = MockBackend(clock=clock)
    event = asyncio.Event()
    event.set()
    result = await run_load(
        backend,
        prompts=["x"] * 50,
        max_tokens=4,
        rate=float("inf"),
        concurrency=2,
        timeout_s=60,
        clock=clock,
        cancel_event=event,
    )
    assert result.launched == 0
    assert result.skipped == 50


async def test_cancel_event_stops_launching_mid_run():
    import asyncio

    clock = VirtualClock()
    event = asyncio.Event()

    class Setter(MockBackend):
        async def stream(self, prompt, *, max_tokens, request_seed):
            event.set()
            async for chunk in super().stream(
                prompt, max_tokens=max_tokens, request_seed=request_seed
            ):
                yield chunk

    result = await run_load(
        Setter(clock=clock),
        prompts=["x"] * 100,
        max_tokens=4,
        rate=float("inf"),
        concurrency=2,
        timeout_s=60,
        clock=clock,
        cancel_event=event,
    )
    assert result.launched < 100


async def test_deadline_exceeded_is_an_error_sample_not_a_crash():
    clock = VirtualClock()
    slow = MockBackend(clock=clock, tier="A100-40G")
    result = await run_load(
        slow,
        prompts=["x"],
        max_tokens=500,
        rate=float("inf"),
        concurrency=1,
        timeout_s=0.001,
        clock=clock,
    )
    assert len(result.completed) == 0
    assert "deadline" in result.samples[0].error
