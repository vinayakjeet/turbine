
import pytest

from turbine.breakeven import (
    DEFAULT_API_MODELS,
    ApiModel,
    SelfHostOption,
    break_even,
    busy_cost_per_request,
    render_table,
)


def option(tok=100.0, **kw) -> SelfHostOption:
    base = dict(
        label="A10G",
        tok_per_s=tok,
        usd_per_hour=1.10,
        cold_starts_per_day=0.0,
        reserved_hours_per_day=0.0,
    )
    base.update(kw)
    return SelfHostOption(**base)


def model(**kw) -> ApiModel:
    return ApiModel("mini", 0.25, 2.0, "2026-08-24", **kw)


def test_busy_cost_math_by_hand():
    # 1536 tokens at 100 tok/s = 15.36 s; at $1.10/h that is
    # 15.36 / 3600 * 1.10 dollars.
    assert busy_cost_per_request(100.0, 1.10, 1536) == pytest.approx(15.36 / 3600 * 1.10)


def test_scale_to_zero_with_no_fixed_cost_wins_at_any_volume():
    be = break_even(option(tok=1000.0), model(), 1024, 512)
    assert be.verdict == "always"
    assert be.requests_per_day is None



def test_cold_starts_create_a_finite_break_even():
    be = break_even(
        option(tok=1000.0, cold_starts_per_day=6, cold_start_min=2.5),
        model(),
        1024,
        512,
    )
    assert be.verdict == "at"
    # fixed daily = 6 * (2.5/60) * 1.10 = $0.275
    # api/req = 1024/1e6*0.25 + 512/1e6*2 = $0.00128
    # busy/req = round(1024*0.85+512) = 1382 tok -> 1.382 s at 1000 tok/s
    busy = busy_cost_per_request(1000.0, 1.10, 1382)
    assert busy < 0.00128, "test premise: served second must beat the API price"
    rpd = 0.275 / (0.00128 - busy)
    assert abs(be.requests_per_day - rpd) < max(1.0, rpd * 1e-9)


def test_api_cheaper_than_served_second_means_never():
    slow = option(tok=1.0)
    be = break_even(slow, ApiModel("tiny", 0.0000001, 0.0000001, "2026-08-24"), 1024, 512)
    assert be.verdict == "never"


def test_unmeasured_option_stays_honest():
    be = break_even(option(tok=None), model(), 1024, 512)
    assert be.verdict == "not measured"
    assert be.requests_per_day is None


def test_table_has_a_break_even_column_and_blank_rows_for_unmeasured():
    lines = render_table(
        [
            option(tok=1000.0, cold_starts_per_day=6, cold_start_min=2.5),
            SelfHostOption("T4", None, 0.59),
        ],
        DEFAULT_API_MODELS,
        tokens_in=1024,
        tokens_out=512,
        price_date="2026-08-24",
    )
    header = next(line for line in lines if line.startswith("| self-host"))
    assert "break-even" in header
    body = [line for line in lines if line.startswith("| A10G")]
    assert "at" in body[0]
    t4_row = next(line for line in lines if line.startswith("| T4"))
    assert "not measured" in t4_row

