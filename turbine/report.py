"""Result files to markdown, gates inline.

The gates fire at render time because the cheapest place to catch a
dishonest table is before it becomes one: unpinned environments, aggregates
under three repetitions, terminated cells and kernel-less quantized rows
are refused rather than footnoted. Mock results are never refused outright,
but a report containing any of them is labelled synthetic in its first
line, and synthetic and measured numbers are never mixed in one document.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from turbine.breakeven import DEFAULT_API_MODELS, SelfHostOption
from turbine.breakeven import render_table as breakeven_table
from turbine.config import (
    CONCURRENCIES,
    INPUT_TOKENS,
    MIN_RUNS,
    NUM_PROMPTS,
    OUTPUT_TOKENS,
    PREFIX_REUSE_SWEEP,
    PREFIX_STUDY_CONCURRENCY,
    REQUEST_RATES,
    WARMUP_REQUESTS,
    Cell,
    rate_label,
)
from turbine.hardware import PRICE_SOURCE_DATE
from turbine.kernels import require_publishable


class Unpublishable(ValueError):
    pass


@dataclass(frozen=True)
class RunFile:
    path: Path
    cell: Cell
    kernel: str
    environment: dict
    terminated: bool
    command: str
    reps: list[dict]

    @property
    def is_mock(self) -> bool:
        return self.environment.get("backend") == "mock"


def load_run(path: Path) -> RunFile:
    raw = json.loads(path.read_text(encoding="utf-8"))
    c = raw["cell"]
    return RunFile(
        path=path,
        cell=Cell(
            tier=c["tier"],
            precision=c["precision"],
            concurrency=int(c["concurrency"]),
            rate=float(c["rate"]),
            prefix_reuse=float(c.get("prefix_reuse", 0.0)),
        ),
        kernel=raw["kernel"],
        environment=raw["environment"],
        terminated=bool(raw.get("terminated", False)),
        command=raw.get("command", ""),
        reps=list(raw["reps"]),
    )


def load_runs(paths: list[Path]) -> list[RunFile]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        else:
            files.append(p)
    return [load_run(f) for f in files if f.name != "manifest.json"]


def validate_publishable(run: RunFile) -> None:
    env = run.environment
    if run.is_mock:
        return
    if not all(str(env.get(k, "")).strip() not in ("", "synthetic") for k in
               ("vllm_version", "cuda_version", "driver_version", "model_id", "model_revision")):
        raise Unpublishable(f"{run.path.name}: environment not fully pinned")
    if run.terminated:
        raise Unpublishable(f"{run.path.name}: terminated by watchdog, incomplete cell")
    if len(run.reps) < MIN_RUNS:
        raise Unpublishable(
            f"{run.path.name}: {len(run.reps)} repetitions is under the {MIN_RUNS}-run floor"
        )
    if run.cell.quantized():
        try:
            require_publishable(run.cell.precision, run.kernel)
        except ValueError as exc:
            raise Unpublishable(f"{run.path.name}: {exc}") from exc


def agg(reps: list[dict], key: str) -> tuple[float, float]:
    from turbine.stats import cross_rep

    values = [rep[key] for rep in reps]
    if len(values) < MIN_RUNS:
        raise Unpublishable(f"aggregating {key} over {len(values)} reps")
    return cross_rep(values)


def agg_summary(reps: list[dict], key: str, stat: str) -> tuple[float, float]:
    from turbine.stats import cross_rep

    values = [rep[key][stat] for rep in reps]
    if len(values) < MIN_RUNS:
        raise Unpublishable(f"aggregating {key}.{stat} over {len(values)} reps")
    return cross_rep(values)


def fmt(value: float, stdev_v: float, digits: int = 2) -> str:
    return f"{value:.{digits}f} +/- {stdev_v:.{digits}f}"


def _tok_per_s_per_usd(tok_per_s: float, tier: str) -> float:
    from turbine.hardware import TIERS

    return round(tok_per_s / float(TIERS[tier].usd_per_hour), 1)


def render_grid(runs: list[RunFile], grid: list[Cell]) -> list[str]:
    by_id = {r.cell.id(): r for r in runs}
    lines = [
        "| cell | kernel | tok/s | TTFT p50 ms | TTFT p95 ms | TTFT p99 ms "
        "| TPOT ms | tok/s/$ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell in grid:
        run = by_id.get(cell.id())
        if run is None:
            lines.append(f"| {cell.id()} | | not measured | | | | | |")
            continue
        tok_m, tok_s = agg(run.reps, "tok_per_s")
        tpot_m, tpot_s = agg_summary(run.reps, "tpot_s", "mean")
        p50_m, p50_s = agg_summary(run.reps, "ttft_s", "p50")
        p95_m, p95_s = agg_summary(run.reps, "ttft_s", "p95")
        p99_m, p99_s = agg_summary(run.reps, "ttft_s", "p99")
        cells = [
            f"{cell.id()}",
            run.kernel,
            fmt(tok_m, tok_s),
            fmt(p50_m * 1000, p50_s * 1000),
            fmt(p95_m * 1000, p95_s * 1000),
            fmt(p99_m * 1000, p99_s * 1000),
            fmt(tpot_m * 1000, tpot_s * 1000),
            str(_tok_per_s_per_usd(tok_m, cell.tier)),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_rate_sweep(runs: list[RunFile]) -> list[str]:
    rows = [r for r in runs if r.cell.concurrency == 8 and r.cell.prefix_reuse == 0]
    if not rows:
        return ["Not part of this report: no concurrency-8 cells."]
    rows.sort(key=lambda r: (r.cell.tier, r.cell.precision, r.cell.rate))
    lines = [
        "| config | rate | tok/s | TTFT p99 ms |",
        "|---|---|---|---|",
    ]
    for r in rows:
        tok = agg(r.reps, "tok_per_s")
        p99 = agg_summary(r.reps, "ttft_s", "p99")
        lines.append(
            f"| {r.cell.tier}-{r.cell.precision} | {rate_label(r.cell.rate)} "
            f"| {fmt(*tok)} | {fmt(*p99)} |"
        )
    return lines


def render_prefix_study(runs: list[RunFile]) -> list[str]:
    import re

    labels = "|".join(re.escape(f"{r:g}") for r in PREFIX_REUSE_SWEEP if r > 0)
    pattern = re.compile(rf"-p({labels})$")
    study = [r for r in runs if pattern.search(r.path.stem)]
    if not study:
        return ["No reuse sweep in this report."]

    # The 0% baseline is the plain concurrency-8 cell on the study tier; its
    # id carries no reuse suffix because zero means the core grid shape.
    base = next(
        (
            r
            for r in runs
            if r.cell.prefix_reuse == 0
            and r.cell.tier == study[0].cell.tier
            and r.cell.precision == study[0].cell.precision
            and r.cell.concurrency == PREFIX_STUDY_CONCURRENCY
            and math.isinf(r.cell.rate)
        ),
        None,
    )
    rows = ([base] if base else []) + sorted(study, key=lambda r: -r.cell.prefix_reuse)
    base_p50 = agg_summary(base.reps, "ttft_s", "p50")[0] if base else None
    lines = [
        "| prefix reuse | tok/s | TTFT p50 ms | TTFT saved vs no reuse |",
        "|---|---|---|---|",
    ]
    for r in rows:
        tok = agg(r.reps, "tok_per_s")
        p50_m, p50_s = agg_summary(r.reps, "ttft_s", "p50")
        saved = ""
        if base_p50 and base_p50 > 0 and r.cell.prefix_reuse > 0:
            saved = f"{100 * (1 - p50_m / base_p50):.1f}%"
        label = "0%" if r.cell.prefix_reuse == 0 else f"{r.cell.prefix_reuse:.0%}"
        lines.append(
            f"| {label} | {fmt(*tok)} | {fmt(p50_m, p50_s)} | {saved} |"
        )
    return lines


def decision_options(runs: list[RunFile]) -> list[SelfHostOption]:
    """Best measured throughput per tier feeds the framework; tiers without
    measurements stay honestly blank."""
    best: dict[str, float] = {}
    for r in runs:
        if r.is_mock or r.cell.rate != math.inf or r.cell.concurrency != 32:
            continue
        tok = agg(r.reps, "tok_per_s")[0]
        best[r.cell.tier] = max(best.get(r.cell.tier, 0.0), tok)
    from turbine.hardware import TIERS

    options = []
    for name, t in TIERS.items():
        tok = best.get(name)
        options.append(
            SelfHostOption(
                label=f"{name} self-hosted",
                tok_per_s=tok,
                usd_per_hour=float(t.usd_per_hour),
                cold_starts_per_day=6.0,
                cold_start_min=2.5,
            )
        )
    return options


def render_report(paths: list[Path], *, grid: list[Cell] | None = None) -> str:
    from datetime import date

    if grid is None:
        from turbine.config import core_grid, prefix_grid

        grid = core_grid() + prefix_grid()
    runs = load_runs(paths)
    if not runs:
        raise Unpublishable("no result files found")
    mocks = {r.is_mock for r in runs}
    if len(mocks) > 1:
        raise Unpublishable("mixed synthetic and measured results in one report")

    for r in runs:
        validate_publishable(r)

    today = date.today().isoformat()
    out: list[str] = ["# Turbine benchmark report", ""]
    if mocks.pop():
        out += [
            "**SYNTHETIC.** Produced by the deterministic mock backend while "
            "verifying the harness. The latency model is invented; nothing on "
            "this page is a measurement.",
            "",
        ]

    envs = {json.dumps(r.environment, sort_keys=True): r for r in runs}.values()
    out += ["## Methodology", ""]
    for r in sorted(envs, key=lambda r: r.command):
        e = r.environment
        out.append(
            f"- {e.get('model_id')} @ {e.get('gpu_host')}: vLLM {e.get('vllm_version')}, "
            f"CUDA {e.get('cuda_version')}, driver {e.get('driver_version')}, "
            f"revision {e.get('model_revision')}, recorded {e.get('recorded_at')}"
        )
    out += [
        "",
        f"Protocol: {NUM_PROMPTS} synthetic requests at {INPUT_TOKENS} in / "
        f"{OUTPUT_TOKENS} out, {WARMUP_REQUESTS} warmups, request rates "
        f"{{{', '.join(rate_label(x) for x in REQUEST_RATES)}}}, concurrencies "
        f"{{{', '.join(map(str, CONCURRENCIES))}}}. Every cell: "
        f"{MIN_RUNS} repetitions, mean and sample stdev reported. TTFT and TPOT "
        f"reported separately. Regenerate with `make bench` (synthetic) or "
        f"`turbine cell` against a live server.",
        "",
        "## Serving grid",
        "",
    ]
    out += render_grid(runs, grid)
    out += ["", "## Request-rate sweep at concurrency 8", ""]
    out += render_rate_sweep(runs)
    out += ["", "## Prefix-caching reuse sweep", ""]
    out += render_prefix_study(runs)
    out += ["", "## Decision framework", ""]
    out += breakeven_table(
        decision_options(runs),
        DEFAULT_API_MODELS,
        tokens_in=INPUT_TOKENS,
        tokens_out=OUTPUT_TOKENS,
        price_date=str(PRICE_SOURCE_DATE),
    )
    out += [
        "",
        "## Threats to validity",
        "",
        "- Single-tenant rented GPUs with noisy-neighbour uncertainty; Modal does "
        "not publish per-tenant contention, so run-to-run variance here includes "
        "neighbours we cannot see.",
        "- Synthetic workload: uniform random tokens at fixed lengths, no real "
        "traffic's burstiness, tool calls, or mixed prompt lengths.",
        "- A small sample of models and precisions; conclusions about AWQ are "
        "about one model family, not quantization in general.",
        "- Cloud thermal and clock variance across runs, partially absorbed by "
        "the repetition floor but never eliminated.",
        "",
        f"Regenerated by `turbine report` on {today}.",
    ]
    return "\n".join(out) + "\n"
