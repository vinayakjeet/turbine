"""Grid driver.

Mock sweeps exercise the full pipeline in seconds. Live measurement runs
one cell per invocation instead, because a vLLM process is pinned to one
tier, precision and kernel at a time; relaunching the server between cells
is part of the method, not an inconvenience to route around.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from turbine.backends.mock import MockBackend
from turbine.budget import Budget
from turbine.clocks import VirtualClock
from turbine.config import MIN_RUNS, Cell, RunEnvironment, now_iso
from turbine.runner import run_cell, stable_seed


def mock_factory():
    def make(cell: Cell, rep: int) -> tuple[MockBackend, VirtualClock]:
        clock = VirtualClock()
        backend = MockBackend(
            tier=cell.tier,
            precision=cell.precision,
            concurrency=cell.concurrency,
            seed=stable_seed(f"{cell.id()}:{rep}"),
            prefix_reuse=cell.prefix_reuse,
            clock=clock,
        )
        return backend, clock

    return make


async def run_grid(
    cells: list[Cell],
    factory,
    *,
    environment: RunEnvironment,
    results_dir: Path,
    budget: Budget,
    spend_path: Path,
    label: str,
    reps: int = MIN_RUNS,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for cell in cells:
        result = await run_cell(
            cell,
            factory,
            environment=environment,
            results_dir=results_dir,
            budget=budget,
            spend_path=spend_path,
            reps=reps,
        )
        summary.append(
            {
                "id": cell.id(),
                "terminated": result.terminated,
                "cost_usd": str(result.cost_usd),
            }
        )
    manifest = {
        "label": label,
        "created": now_iso(),
        "command": " ".join(sys.argv),
        "environment": asdict(environment),
        "cells": summary,
    }
    path = results_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return results_dir
