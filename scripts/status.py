"""Headline status number for the portfolio's sourced-metric contract.

Prints the best measured tokens-per-second-per-dollar from real,
fully-pinned results at concurrency 8 and infinite request rate, which is
the sustained-serving point the decision framework prices against. With
nothing measured yet it prints value "not measured" and exits 0 anyway: an
honest blank beats an invented number, and the site treats planned
projects accordingly.

    uv run python scripts/status.py [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import turbine.hardware as hw  # noqa: E402
from turbine.config import MIN_RUNS  # noqa: E402
from turbine.report import Unpublishable, load_runs, validate_publishable  # noqa: E402


def headline(results_root: Path) -> dict:
    runs_path = results_root / "live"
    candidates = []
    try:
        runs = load_runs([runs_path]) if runs_path.exists() else []
    except json.JSONDecodeError:
        return _blank("unreadable result files")
    for run in runs:
        if run.is_mock:
            continue
        if math.isinf(run.cell.rate) and run.cell.concurrency == 8:
            try:
                validate_publishable(run)
            except Unpublishable:
                continue
            tok_values = [rep["tok_per_s"] for rep in run.reps]
            tok_per_s = sum(tok_values) / len(tok_values)
            usd_per_hour = float(hw.TIERS[run.cell.tier].usd_per_hour)
            candidates.append((run, tok_per_s / usd_per_hour))

    if len(candidates) == 0:
        return _blank("no pinned concurrency-8 results yet")

    run, value = max(candidates, key=lambda pair: pair[1])
    return {
        "label": f"{run.cell.tier} {run.cell.precision} sustained serving",
        "value": round(value, 1),
        "unit": "tok/s/$",
        "source": "scripts/status.py",
        "n_runs": len(run.reps),
        "min_runs_required": MIN_RUNS,
        "date": run.environment.get("recorded_at", "")[:10],
    }


def _blank(reason: str) -> dict:
    return {
        "label": "best measured throughput per dollar",
        "value": "not measured",
        "source": "scripts/status.py",
        "reason": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = headline(Path("results"))
    if args.json:
        print(json.dumps(payload))
    else:
        value = payload["value"]
        unit = payload.get("unit", "")
        print(f"{payload['label']}: {value}{unit}")
        if payload["value"] == "not measured":
            print(f"({payload['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
