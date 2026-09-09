from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValuationResult:
    current_eps: float
    growth_rate: float          # decimal, e.g. 0.15 for 15%
    future_pe: float
    future_eps: float
    future_price: float
    value_price: float
    mos_price: float
    discount_rate: float
    horizon_years: int
    growth_source: str          # "big5", "analyst", or "min(big5, analyst)"


def value_price(
    current_eps: float,
    big5_eps_growth: float | None,
    analyst_growth: float | None,
    historical_pe: float | None,
    custom_growth: float | None = None,
    mos: float = 0.5,
    discount: float = 0.15,
    years: int = 10,
) -> ValuationResult | None:
    """Value Price and Margin-of-Safety buy price.

    Returns None when we can't produce an honest number (missing EPS,
    missing every growth estimate, or non-positive inputs).

    Method (from Value):
      1. growth = min(big5 EPS growth, analyst 5yr estimate) — take the lower.
         Cap at 15% to stay conservative even for great businesses.
      2. future_pe = min(historical avg PE, 2 * growth_rate_as_percent).
      3. future_eps = current_eps * (1 + growth) ** years.
      4. future_price = future_eps * future_pe.
      5. value = future_price discounted back `years` at `discount` (15%).
      6. mos_price = value * (1 - mos), default half of value.
    """
    if current_eps is None or current_eps <= 0:
        return None

    # Use custom growth when provided; otherwise use the lower
    # of Big5 and analyst growth estimates.
    if custom_growth is not None and custom_growth > 0:
        growth = custom_growth
        source = "custom"
    else:
        candidates = [
            g for g in (big5_eps_growth, analyst_growth)
            if g is not None and g > 0
        ]

        if not candidates:
            return None

        growth = min(candidates)

        if big5_eps_growth is not None and analyst_growth is not None:
            source = "min(big5, analyst)"
        elif big5_eps_growth is not None:
            source = "big5"
        else:
            source = "analyst"

    # Cap growth at 15%
    #growth = min(growth, 0.15) #RG

    growth_pct = growth * 100

    default_pe = growth_pct * 2
    if historical_pe is not None and historical_pe > 0:
        future_pe = min(historical_pe, default_pe)
    else:
        future_pe = default_pe

    future_eps = current_eps * (1 + growth) ** years
    future_price = future_eps * future_pe
    value = future_price / ((1 + discount) ** years)
    mos_price = value * (1 - mos)

    return ValuationResult(current_eps=current_eps,
        growth_rate=growth,
        future_pe=future_pe,
        future_eps=future_eps,
        future_price=future_price,
        value_price=value,
        mos_price=mos_price,
        discount_rate=discount,
        horizon_years=years,
        growth_source=source,
    )


def payback_time(
    fcf_ttm: float | None,
    growth_rate: float | None,
    market_cap: float | None,
    max_years: int = 30,
) -> float | None:
    """Years to recoup the market cap from cumulative growing owner earnings.

    Town's rule from *Payback Time*: pass if <= 8 years. Uses simple annual
    compounding of FCF at `growth_rate` and sums until cumulative FCF meets
    market cap. Returns None if any input is missing or FCF is non-positive.
    Returns `max_years` if it never pays back within the cap.
    """
    if fcf_ttm is None or fcf_ttm <= 0:
        return None
    if market_cap is None or market_cap <= 0:
        return None
    if growth_rate is None:
        growth_rate = 0.0
    # Cap growth for the payback projection — keeps the number sane for
    # young high-growth companies where extrapolating 30%+ forever is silly.
    growth_rate = min(max(growth_rate, 0.0), 0.20)

    cumulative = 0.0
    for year in range(1, max_years + 1):
        cumulative += fcf_ttm * (1 + growth_rate) ** year
        if cumulative >= market_cap:
            return float(year)
    return float(max_years)


def debt_to_fcf(long_term_debt: float | None, fcf_ttm: float | None) -> float | None:
    """Long-term debt / TTM free cash flow. Pass < 3."""
    if long_term_debt is None or fcf_ttm is None:
        return None
    if fcf_ttm <= 0:
        return None
    if long_term_debt <= 0:
        return 0.0
    return long_term_debt / fcf_ttm
