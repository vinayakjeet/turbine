"""CLI. `python -m turbine --help` for the full list."""

from __future__ import annotations

import asyncio
import json
import os
from decimal import Decimal
from pathlib import Path

import typer

import turbine.cpu as cpu
from turbine.backends.openai_compat import OpenAICompatBackend
from turbine.budget import Budget
from turbine.clocks import RealClock
from turbine.config import (
    DEFAULT_PLANNED_MINUTES,
    MIN_RUNS,
    Cell,
    RunEnvironment,
)
from turbine.quality import evaluate_quality
from turbine.report import render_report
from turbine.runner import run_cell
from turbine.spend import load as load_spend
from turbine.spend import month_to_date, total_cost

app = typer.Typer(help="Inference serving benchmarks on free-tier credit.", no_args_is_help=True)

RESULTS_DIR = Path("results")
SPEND_PATH = Path("spend.jsonl")


def _budget() -> Budget:
    return Budget(
        cap_usd=Decimal(os.environ.get("TURBINE_BUDGET_CAP_USD", "30")),
        margin_usd=Decimal(os.environ.get("TURBINE_SAFETY_MARGIN_USD", "5")),
    )


@app.command()
def sweep(
    label: str = typer.Option("mock", help="Name for the results directory."),
    reps: int = typer.Option(MIN_RUNS, min=1),
) -> None:
    """Full synthetic grid: core matrix plus the prefix reuse sweep.

    Mock backend only. Real measurement is per-server-launch; see `cell`.
    """
    from turbine.config import core_grid, prefix_grid
    from turbine.sweep import mock_factory, run_grid

    cells = core_grid() + prefix_grid()
    out = RESULTS_DIR / label
    asyncio.run(
        run_grid(
            cells,
            mock_factory(),
            environment=RunEnvironment.synthetic(),
            results_dir=out,
            budget=_budget(),
            spend_path=SPEND_PATH,
            label=label,
            reps=reps,
        )
    )
    report_path = Path("docs") / f"{label.upper()}-BENCH.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report([out]), encoding="utf-8")
    typer.echo(f"results in {out}, report at {report_path}")


@app.command()
def cell(
    tier: str = typer.Option(...),
    precision: str = typer.Option(...),
    concurrency: int = typer.Option(...),
    rate: float = typer.Option(float("inf")),
    quantization: str = typer.Option(None, help="vLLM --quantization value, e.g. awq_marlin."),
    reps: int = typer.Option(MIN_RUNS, min=1),
    planned_minutes: float = typer.Option(DEFAULT_PLANNED_MINUTES),
    base_url: str = typer.Option("", envvar="TURBINE_BASE_URL"),
    model: str = typer.Option("", envvar="TURBINE_MODEL"),
    api_key: str = typer.Option("", envvar="TURBINE_API_KEY"),
    vllm_version: str = typer.Option(...),
    cuda_version: str = typer.Option(...),
    driver_version: str = typer.Option(...),
    model_revision: str = typer.Option(...),
    gpu_host: str = typer.Option("modal"),
) -> None:
    """One grid position against a live OpenAI-compatible server."""
    if not base_url or not model:
        typer.echo("TURBINE_BASE_URL and TURBINE_MODEL are required for a live run", err=True)
        raise typer.Exit(2)

    cell_cfg = Cell(
        tier=tier,
        precision=precision,
        concurrency=concurrency,
        rate=rate,
    )
    environment = RunEnvironment.live(
        vllm_version=vllm_version,
        cuda_version=cuda_version,
        driver_version=driver_version,
        model_id=model,
        model_revision=model_revision,
        gpu_host=gpu_host,
    )

    def factory(cell: Cell, rep: int):
        clock = RealClock()
        backend = OpenAICompatBackend(
            base_url, model, api_key=api_key or None, clock=clock
        )
        return backend, clock

    out = RESULTS_DIR / "live"
    result = asyncio.run(
        run_cell(
            cell_cfg,
            factory,
            environment=environment,
            results_dir=out,
            budget=_budget(),
            spend_path=SPEND_PATH,
            reps=reps,
            planned_minutes=planned_minutes,
            quant_arg=quantization,
        )
    )
    typer.echo(f"{cell_cfg.id()}: {len(result.reps)} reps, ${result.cost_usd}, results in {out}")


@app.command()
def quality(
    fp16_url: str = typer.Option(...),
    quant_url: str = typer.Option(...),
    model_fp16: str = typer.Option(...),
    model_quant: str = typer.Option(...),
    dataset: Path = typer.Option(None, help="ShipGate-format JSONL eval set."),
    api_key: str = typer.Option("", envvar="TURBINE_API_KEY"),
    reps: int = typer.Option(MIN_RUNS, min=2),
) -> None:
    """Quantization quality delta, exact match on the ShipGate-format set."""
    if dataset is None:
        dataset = Path("datasets/support-intent-quant.jsonl")

    def make(url: str, name: str):
        return lambda rep: OpenAICompatBackend(url, name, api_key=api_key or None)

    result = asyncio.run(
        evaluate_quality(
            make(fp16_url, model_fp16), make(quant_url, model_quant), dataset, reps=reps
        )
    )
    typer.echo(json.dumps(result, indent=2))
    out_dir = RESULTS_DIR / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


@app.command(name="cpu-bench")
def cpu_bench(
    binary: str = typer.Option("llama-bench"),
    model: Path = typer.Option(...),
    threads: int = typer.Option(None),
    pp: int = typer.Option(512),
    tg: int = typer.Option(128),
    mock: bool = typer.Option(False, help="Synthetic rows for pipeline verification."),
) -> None:
    """The CPU chapter via llama-bench CSV output."""
    if not mock and not model.exists():
        typer.echo(f"model file not found: {model}", err=True)
        raise typer.Exit(2)
    if mock:
        rows = cpu.mock_rows()
        measured = False
    else:
        rows = cpu.run_llama_bench(binary, model, pp=pp, tg=tg, threads=threads)
        measured = True
    for line in cpu.render_chapter(rows, measured=measured):
        typer.echo(line)


@app.command()
def report(
    dirs: list[Path] = typer.Argument(None),
    out: Path = typer.Option(None, help="Write markdown here instead of stdout."),
) -> None:
    """Render results directories into the benchmark report markdown."""
    targets = dirs if dirs else [RESULTS_DIR / "mock"]
    text = render_report(targets)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out}")
    else:
        typer.echo(text)


@app.command(name="spend")
def spend_report() -> None:
    """Month-to-date credit ledger with per-run lines."""
    entries = month_to_date(load_spend(SPEND_PATH))
    for e in entries:
        note = f"  {e.note}" if e.note else ""
        typer.echo(f"{e.ts}  {e.run_id}  {e.tier}  {e.gpu_hours}h  ${e.cost_usd}{note}")
    typer.echo(f"month to date: ${total_cost(entries)}")


@app.command(name="budget-check")
def budget_check(
    tier: str = typer.Option(...),
    planned_minutes: float = typer.Option(DEFAULT_PLANNED_MINUTES),
) -> None:
    """Print what the guard would decide for this run, without running it."""
    import turbine.hardware as hw
    from turbine.budget import plan
    from turbine.runner import month_spent

    estimate = hw.cost(Decimal(str(planned_minutes)) / 60, tier)
    decision = plan(_budget(), month_spent(SPEND_PATH), estimate)
    typer.echo(decision.reason)
    if not decision.allowed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
