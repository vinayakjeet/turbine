"""Append-only credit ledger.

The spend log lives in the repo on purpose: a $30 budget nobody can audit
is just a number. One JSON object per line, cost carried as a string so the
decimal never drifts through a float.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class SpendEntry:
    ts: str
    run_id: str
    backend: str
    tier: str
    gpu_hours: str
    cost_usd: str
    note: str = ""


def append(path: Path, entry: SpendEntry) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(entry)) + "\n")


def load(path: Path) -> list[SpendEntry]:
    if not path.exists():
        return []
    entries: list[SpendEntry] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            entries.append(
                SpendEntry(
                    ts=raw["ts"],
                    run_id=raw["run_id"],
                    backend=raw["backend"],
                    tier=raw["tier"],
                    gpu_hours=raw["gpu_hours"],
                    cost_usd=raw["cost_usd"],
                    note=raw.get("note", ""),
                )
            )
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path}:{lineno}: malformed spend entry ({exc})") from exc
    return entries


def total_cost(entries: list[SpendEntry]) -> Decimal:
    return sum((Decimal(e.cost_usd) for e in entries), Decimal("0"))


def month_to_date(entries: list[SpendEntry], now: datetime | None = None) -> list[SpendEntry]:
    ref = now or datetime.now(UTC)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=UTC)
    out = []
    for e in entries:
        try:
            ts = datetime.fromisoformat(e.ts)
        except ValueError:
            continue
        ts = ts.astimezone(UTC) if ts.tzinfo is not None else ts.replace(tzinfo=UTC)
        if (ts.year, ts.month) == (ref.year, ref.month):
            out.append(e)

    return out
