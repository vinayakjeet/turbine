import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

import turbine.hardware as hw
from turbine.budget import Budget, RunWatchdog, allotment_seconds, plan
from turbine.clocks import RealClock, VirtualClock


def test_plan_allows_inside_budget():
    b = Budget(cap_usd=Decimal("30"), margin_usd=Decimal("5"))
    d = plan(b, Decimal("0"), Decimal("0.10"))
    assert d.allowed
    assert d.remaining_after_estimate == Decimal("24.90")


def test_plan_refuses_inside_margin_and_names_the_numbers():
    b = Budget(cap_usd=Decimal("30"), margin_usd=Decimal("5"))
    d = plan(b, Decimal("24.95"), Decimal("0.10"))
    assert not d.allowed
    assert "refused" in d.reason
    assert "0.05" in d.reason or "0.1" in d.reason


def test_plan_boundary_is_inclusive():
    # Landing exactly on zero remaining is allowed: the guard refuses what
    # would land *inside* the margin, and the margin is already subtracted.
    b = Budget(cap_usd=Decimal("30"), margin_usd=Decimal("5"))
    assert plan(b, Decimal("24.90"), Decimal("0.10")).allowed is True
    assert plan(b, Decimal("24.95"), Decimal("0.10")).allowed is False


def test_plan_rejects_free_estimates():
    b = Budget(cap_usd=Decimal("30"), margin_usd=Decimal("5"))
    with pytest.raises(ValueError):
        plan(b, Decimal("0"), Decimal("0"))


def test_prices_are_dated_so_stale_tables_announce_themselves():
    assert hw.PRICE_SOURCE_DATE.isoformat() == "2026-08-24"


def test_cost_quantizes_to_four_places():
    assert hw.cost(Decimal("1.5"), "T4") == Decimal("0.8850")
    assert hw.cost(2.0, "A10G") == Decimal("2.2000")


def test_unknown_tier_names_the_known_ones():
    with pytest.raises(ValueError, match="T4"):
        hw.tier("H200")


async def test_watchdog_fires_after_its_limit_on_wall_time():
    w = RunWatchdog(0.01, RealClock())
    await asyncio.wait_for(w.cancel_event.wait(), timeout=2)
    assert w.timed_out
    await w.stop()


async def test_infinite_limit_never_fires_no_matter_how_time_moves():
    clock = VirtualClock()
    w = RunWatchdog(float("inf"), clock)
    await clock.sleep(1_000_000)
    await asyncio.sleep(0)
    assert not w.timed_out
    assert not w.cancel_event.is_set()
    await w.stop()


async def test_zero_limit_means_already_exhausted():
    w = RunWatchdog(0.0, VirtualClock())
    assert w.timed_out
    assert w.cancel_event.is_set()
    await w.stop()


def test_allotment_scales_with_remaining_credit_and_price():
    b = Budget(cap_usd=Decimal("30"), margin_usd=Decimal("5"))
    secs = allotment_seconds(b, Decimal("0"), Decimal("0.59"))
    assert secs == pytest.approx(25 / 0.59 * 3600)
    assert allotment_seconds(b, Decimal("25"), Decimal("0.59")) == 0.0


def test_month_window_filters_other_months():
    from turbine.spend import SpendEntry, month_to_date

    entries = [
        SpendEntry("2026-07-31T00:00:00+00:00", "old", "mock", "T4", "1", "1.0"),
        SpendEntry("2026-08-01T00:00:00+00:00", "this", "mock", "T4", "1", "2.0"),
    ]
    now = datetime(2026, 8, 15, tzinfo=UTC)
    kept = month_to_date(entries, now)
    assert [e.run_id for e in kept] == ["this"]
