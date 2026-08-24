"""One grid position, measured.

The sequence per cell is fixed: budget preflight, watchdog armed from what
the budget can still pay for, five warmup requests, then MIN_RUNS measured
repetitions with fresh seeds. The spend entry is written even when the
watchdog fires, because terminated GPU time is billed GPU time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path

import turbine.hardware as hw
from turbine.budget import Budget, BudgetRefused, RunWatchdog, allotment_seconds, plan
from turbine.clocks import Clock, RealClock
from turbine.config import (
    DEFAULT_PLANNED_MINUTES,
    INPUT_TOKENS,
    MIN_RUNS,
    NUM_PROMPTS,
    OUTPUT_TOKENS,
    REQUEST_TIMEOUT_S,
    WARMUP_REQUESTS,
    Cell,
    RunEnvironment,
    now_iso,
)
from turbine.kernels import resolve_kernel
from turbine.loadgen import LoadResult, build_prompts, run_load
from turbine.spend import (
    SpendEntry,
    month_to_date,
    total_cost,
)
from turbine.spend import (
    append as append_spend,
)
from turbine.spend import (
    load as load_spend,
)
from turbine.stats import RepStats, Summary, summarize


@dataclass(frozen=True)
class RepRecord:
    rep: int
    stats: RepStats


@dataclass(frozen=True)
class CellResult:
    cell: Cell
    kernel: str
    environment: RunEnvironment
    reps: list[RepRecord]
    terminated: bool
    cost_usd: Decimal

    def publishable(self) -> bool:
        return not self.terminated and all(r.stats.errors == 0 for r in self.reps)


def cell_command(cell: Cell, quant_arg: str | None) -> str:
    parts = [
        "turbine cell",
        f"--tier {cell.tier}",
        f"--precision {cell.precision}",
        f"--concurrency {cell.concurrency}",
        f"--rate {cell.rate:g}",
    ]
    if quant_arg:
        parts.append(f"--quantization {quant_arg}")
    return " ".join(parts)


def month_spent(spend_path: Path) -> Decimal:
    return total_cost(month_to_date(load_spend(spend_path)))


def stable_seed(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


async def run_cell(
    cell: Cell,
    factory,
    *,
    environment: RunEnvironment,
    results_dir: Path | None,
    budget: Budget,
    spend_path: Path,
    reps: int = MIN_RUNS,
    planned_minutes: float = DEFAULT_PLANNED_MINUTES,
    quant_arg: str | None = None,
    spent_override: Decimal | None = None,
) -> CellResult:
    price = hw.tier(cell.tier).usd_per_hour
    billed = environment.backend != "mock"
    spent = spent_override if spent_override is not None else month_spent(spend_path)

    # Synthetic runs place no credit at risk, so the preflight guards only
    # billed backends; their ledger entries still land, at zero cost, so the
    # log stays complete.
    if billed:
        estimate = hw.cost(Decimal(str(planned_minutes)) / 60, cell.tier)
        decision = plan(budget, spent, estimate)
        if not decision.allowed:
            raise BudgetRefused(decision.reason)

    # The watchdog runs on wall time regardless of the measurement clock:
    # billing accrues in real hours even for a virtual-time run.
    watchdog = RunWatchdog(
        allotment_seconds(budget, spent if billed else Decimal("0"), price),
        RealClock(),
    )
    kernel = resolve_kernel(cell.precision, quant_arg)

    rep_records: list[RepRecord] = []
    try:
        for rep in range(reps):
            # The factory owns its clock domain: mock reps come back on a
            # fresh VirtualClock, live reps on wall time.
            backend, rep_clock = factory(cell, rep)
            load = await measure_rep(
                backend, cell, rep=rep, clock=rep_clock, cancel_event=watchdog.cancel_event
            )
            await backend.aclose()
            rep_records.append(RepRecord(rep=rep, stats=rep_stats(load)))
    finally:
        await watchdog.stop()
    gpu_hours = sum(r.stats.duration_s for r in rep_records) / 3600.0
    cost = hw.cost(gpu_hours, cell.tier) if billed else Decimal("0.0000")
    result = CellResult(
        cell=cell,
        kernel=kernel,
        environment=environment,
        reps=rep_records,
        terminated=watchdog.timed_out,
        cost_usd=cost,
    )

    notes = []
    if not billed:
        notes.append("synthetic run, no billing")
    if watchdog.timed_out:
        notes.append("terminated by watchdog")

    append_spend(
        spend_path,
        SpendEntry(
            ts=now_iso(),
            run_id=f"{cell.id()}@{environment.recorded_at}",
            backend=environment.backend,
            tier=cell.tier,
            gpu_hours=f"{gpu_hours:.6f}",
            cost_usd=str(cost),
            note="; ".join(notes),
        ),
    )

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "cell": asdict(cell),
            "kernel": kernel,
            "environment": asdict(environment),
            "terminated": result.terminated,
            "cost_usd": str(cost),
            "command": cell_command(cell, quant_arg),
            "reps": [
                {"rep": r.rep, **_stats_payload(r.stats)} for r in rep_records
            ],
        }
        path = results_dir / f"{cell.id()}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result


async def measure_rep(
    backend, cell: Cell, *, rep: int, clock: Clock, cancel_event=None
) -> LoadResult:
    grid_seed = stable_seed(cell.id())
    prompts = build_prompts(grid_seed, NUM_PROMPTS, INPUT_TOKENS, cell.prefix_reuse)
    load_kw = {
        "max_tokens": OUTPUT_TOKENS,
        "rate": cell.rate,
        "concurrency": cell.concurrency,
        "timeout_s": REQUEST_TIMEOUT_S,
        "clock": clock,
    }
    await run_load(
        backend,
        prompts=prompts[:WARMUP_REQUESTS],
        seed=rep,
        **load_kw,
    )
    return await run_load(
        backend,
        prompts=prompts,
        seed=10_000 + rep,
        **load_kw,
        cancel_event=cancel_event,
    )


def rep_stats(load: LoadResult) -> RepStats:
    if not load.samples:
        # Nothing launched: cancelled before the first request. Zeros with a
        # zero-length summary, never a fabricated mean.
        empty = Summary(n=0, mean=0.0, stdev=0.0, p50=0.0, p95=0.0, p99=0.0)
        return RepStats(
            duration_s=round(max(load.duration_s, 0.0), 6),
            requests=0,
            errors=0,
            output_tokens=0,
            tok_per_s=0.0,
            ttft_s=empty,
            tpot_s=empty,
        )
    done = load.completed
    ttfts = [s.ttft_s for s in done if s.ttft_s is not None]
    tpots = [s.tpot_s for s in done if s.tpot_s is not None]
    tokens = sum(s.output_tokens for s in done)
    duration = max(load.duration_s, 1e-9)
    return RepStats(
        duration_s=round(duration, 6),
        requests=len(load.samples),
        errors=len(load.samples) - len(done),
        output_tokens=tokens,
        tok_per_s=round(tokens / duration, 2),
        ttft_s=summarize(ttfts),
        tpot_s=summarize(tpots),
    )


def _stats_payload(s: RepStats) -> dict:
    return {
        "duration_s": s.duration_s,
        "requests": s.requests,
        "errors": s.errors,
        "output_tokens": s.output_tokens,
        "tok_per_s": s.tok_per_s,
        "ttft_s": asdict(s.ttft_s),
        "tpot_s": asdict(s.tpot_s),
    }
