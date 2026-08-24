"""Break-even math for the when-to-self-host table.

Pure functions. Prices are inputs with sources and dates attached; this
module interpolates between them, it never measures anything itself.

Scale-to-zero changes what break-even means. With per-second billing and no
reserved floor, a self-hosted GPU wins on every request it serves and loses
only on cold starts and reserved idle, so the crossover lives in the
fixed-cost column rather than the per-token one. The table makes that
visible instead of hiding it inside an average.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ApiModel:
    name: str
    usd_per_m_input: float
    usd_per_m_output: float
    source_date: str


# Transcribed from provider pricing pages 2026-08-24 for the decision
# framework's default columns. Re-check before publishing: API prices move,
# and a stale input here silently moves every break-even row.
DEFAULT_API_MODELS = (
    ApiModel("frontier flagship", 1.25, 10.0, "2026-08-24"),
    ApiModel("frontier mini", 0.25, 2.0, "2026-08-24"),
)


@dataclass(frozen=True)
class SelfHostOption:
    label: str
    tok_per_s: float | None
    usd_per_hour: float
    utilization: float = 0.85
    reserved_hours_per_day: float = 0.0
    cold_starts_per_day: float = 0.0
    cold_start_min: float = 0.0


def api_cost_per_request(model: ApiModel, tokens_in: int, tokens_out: int) -> float:
    return tokens_in / 1e6 * model.usd_per_m_input + tokens_out / 1e6 * model.usd_per_m_output


def busy_cost_per_request(tok_per_s: float, usd_per_hour: float, tokens_per_request: int) -> float:
    """GPU-second cost of one request while the card is serving."""
    seconds = tokens_per_request / tok_per_s
    return seconds / 3600.0 * usd_per_hour


def fixed_daily_cost(option: SelfHostOption) -> float:
    reserved = option.reserved_hours_per_day * option.usd_per_hour
    colds = option.cold_starts_per_day * (option.cold_start_min / 60.0) * option.usd_per_hour
    return reserved + colds


@dataclass(frozen=True)
class BreakEven:
    verdict: str
    requests_per_day: float | None
    note: str


def break_even(
    option: SelfHostOption, model: ApiModel, tokens_in: int, tokens_out: int
) -> BreakEven:
    if option.tok_per_s is None or option.tok_per_s <= 0:
        return BreakEven("not measured", None, "no throughput measured yet")

    tpr = round(tokens_in * option.utilization + tokens_out)
    api = api_cost_per_request(model, tokens_in, tokens_out)
    busy = busy_cost_per_request(option.tok_per_s, option.usd_per_hour, tpr)
    fixed = fixed_daily_cost(option)

    if api <= busy:
        return BreakEven(
            "never",
            None,
            "API is cheaper than the served second already; no volume helps",
        )
    if fixed <= 0:
        return BreakEven(
            "always",
            None,
            "scale-to-zero with no reserved floor: every served request wins",
        )
    rpd = fixed / (api - busy)
    return BreakEven("at", round(rpd), f"fixed ${fixed:.3f}/day against ${api - busy:.6f}/req")


def render_table(
    options: list[SelfHostOption],
    models: tuple[ApiModel, ...],
    *,
    tokens_in: int,
    tokens_out: int,
    price_date: str,
) -> list[str]:
    lines = [
        f"Request shape: {tokens_in} in / {tokens_out} out. Prices transcribed {price_date}.",
        "",
        "| self-host option | tok/s | $/1k requests | break-even vs "
        + ", ".join(m.name for m in models)
        + " (requests/day) |",
        "|---|---|---|---|",
    ]
    for opt in options:
        if opt.tok_per_s is None:
            lines.append(f"| {opt.label} | not measured | not measured | not measured |")
            continue
        cost_1k = (
            busy_cost_per_request(opt.tok_per_s, opt.usd_per_hour, tokens_in + tokens_out)
            * 1000.0
        )
        cells = []
        for model in models:
            be = break_even(opt, model, tokens_in, tokens_out)
            if be.verdict == "not measured":
                cells.append("not measured")
            elif be.requests_per_day is None:
                cells.append(be.verdict)
            else:
                cells.append(f"{be.verdict} {be.requests_per_day:,.0f}")
        lines.append(
            f"| {opt.label} | {opt.tok_per_s:.1f} | {cost_1k:.4f} | {'; '.join(cells)} |"
        )
    return lines
