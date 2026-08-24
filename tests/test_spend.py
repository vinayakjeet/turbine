from decimal import Decimal
from pathlib import Path

import pytest

from turbine.spend import SpendEntry, append, load, total_cost


def entry(**kw) -> SpendEntry:
    base = dict(
        ts="2026-08-24T10:00:00+00:00",
        run_id="T4-fp16-c1-rinf",
        backend="mock",
        tier="T4",
        gpu_hours="0.010000",
        cost_usd="0.0059",
    )
    base.update(kw)
    return SpendEntry(**base)


def test_round_trip_preserves_decimals_as_strings(tmp_path: Path):
    path = tmp_path / "spend.jsonl"
    append(path, entry(note="first"))
    append(path, entry(cost_usd="0.1000", note="second"))
    entries = load(path)
    assert len(entries) == 2
    assert entries[1].cost_usd == "0.1000"
    assert total_cost(entries) == Decimal("0.1059")


def test_malformed_line_names_itself(tmp_path: Path):
    path = tmp_path / "spend.jsonl"
    append(path, entry())
    with path.open("a", encoding="utf-8") as fh:
        fh.write("not json\n")
    with pytest.raises(ValueError, match=r"spend\.jsonl:2"):
        load(path)


def test_missing_file_is_zero_spend(tmp_path: Path):
    assert load(tmp_path / "absent.jsonl") == []
