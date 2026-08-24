from decimal import Decimal
from pathlib import Path

import pytest

import turbine.spend as spend
from turbine.backends.mock import MockBackend
from turbine.budget import Budget, BudgetRefused
from turbine.clocks import VirtualClock
from turbine.config import RunEnvironment
from turbine.runner import month_spent, run_cell, stable_seed


def budget() -> Budget:
    return Budget(cap_usd=Decimal("30"), margin_usd=Decimal("5"))


def live_env() -> RunEnvironment:
    return RunEnvironment.live(
        vllm_version="0.11.0",
        cuda_version="12.9",
        driver_version="580.00",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_revision="abc1234",
        gpu_host="modal",
    )


def factory(clock):
    def make(cell, rep):
        rep_clock = VirtualClock()
        return (
            MockBackend(
                tier=cell.tier,
                precision=cell.precision,
                concurrency=cell.concurrency,
                seed=stable_seed(f"{cell.id()}:{rep}"),
                clock=rep_clock,
            ),
            rep_clock,
        )

    return make


async def test_run_cell_writes_reps_and_spend_entry(tmp_path: Path):
    from turbine.config import Cell

    cell = Cell(tier="T4", precision="fp16", concurrency=1, rate=float("inf"))
    out = tmp_path / "results"
    result = await run_cell(
        cell,
        factory(None),
        environment=live_env(),
        results_dir=out,
        budget=budget(),
        spend_path=tmp_path / "spend.jsonl",
    )
    assert len(result.reps) == 3
    assert result.kernel == "none"
    assert not result.terminated
    payload_file = out / f"{cell.id()}.json"
    assert payload_file.exists()
    entries = spend.load(tmp_path / "spend.jsonl")
    assert len(entries) == 1
    assert Decimal(entries[0].cost_usd) == result.cost_usd
    assert month_spent(tmp_path / "spend.jsonl") > 0


async def test_mock_runs_bill_nothing_and_cannot_exhaust_the_budget(tmp_path: Path):
    from turbine.config import Cell

    cell = Cell(tier="A100-40G", precision="fp16", concurrency=8, rate=float("inf"))
    tight = Budget(cap_usd=Decimal("30"), margin_usd=Decimal("5"))
    for _ in range(3):
        result = await run_cell(
            cell,
            factory(None),
            environment=RunEnvironment.synthetic(),
            results_dir=None,
            budget=tight,
            spend_path=tmp_path / "spend.jsonl",
        )
        assert result.cost_usd == Decimal("0.0000")
    entries = spend.load(tmp_path / "spend.jsonl")
    assert len(entries) == 3
    assert all(e.note == "synthetic run, no billing" for e in entries)
    assert month_spent(tmp_path / "spend.jsonl") == 0


async def test_budget_refusal_writes_nothing(tmp_path: Path):
    from turbine.config import Cell

    cell = Cell(tier="A100-40G", precision="fp16", concurrency=8, rate=float("inf"))
    tight = Budget(cap_usd=Decimal("30"), margin_usd=Decimal("5"))
    with pytest.raises(BudgetRefused):
        await run_cell(
            cell,
            factory(None),
            environment=live_env(),
            results_dir=None,
            budget=tight,
            spend_path=tmp_path / "spend.jsonl",
            planned_minutes=60 * 24,
        )
    assert not (tmp_path / "spend.jsonl").exists()


async def test_quantized_cell_carries_a_kernel_without_being_told(tmp_path: Path):
    from turbine.config import Cell

    cell = Cell(tier="A10G", precision="awq", concurrency=8, rate=float("inf"))
    result = await run_cell(
        cell,
        factory(None),
        environment=RunEnvironment.synthetic(),
        results_dir=None,
        budget=budget(),
        spend_path=tmp_path / "spend.jsonl",
    )
    assert result.kernel == "marlin"


async def test_watchdog_firing_marks_result_unpublishable_but_still_bills(
    tmp_path: Path, monkeypatch
):
    from turbine import runner as runner_mod
    from turbine.config import Cell

    monkeypatch.setattr(runner_mod, "allotment_seconds", lambda *a, **k: 0.0)

    cell = Cell(tier="T4", precision="fp16", concurrency=1, rate=float("inf"))
    result = await run_cell(
        cell,
        factory(None),
        environment=RunEnvironment.synthetic(),
        results_dir=None,
        budget=budget(),
        spend_path=tmp_path / "spend.jsonl",
    )
    assert result.terminated
    assert not result.publishable()
    entries = spend.load(tmp_path / "spend.jsonl")
    assert "terminated by watchdog" in entries[0].note


async def test_mock_results_are_byte_identical_across_calls(tmp_path: Path):
    from turbine.config import Cell

    async def once() -> dict:
        cell = Cell(tier="A10G", precision="fp16", concurrency=4, rate=float("inf"))
        result = await run_cell(
            cell,
            factory(None),
            environment=RunEnvironment.synthetic(),
            results_dir=None,
            budget=budget(),
            spend_path=tmp_path / "s.jsonl",
        )
        return {
            r.rep: r.stats.tok_per_s for r in result.reps
        } | {"ttft0": result.reps[0].stats.ttft_s.mean}

    first = await once()
    second = await once()
    assert first == second
