from __future__ import annotations

from dataclasses import dataclass
import statistics

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

MAX_ANALYST_GROWTH_FORWARD = 0.40
MAX_ANALYST_GROWTH_TRAILING = 0.25
MAX_BIG5_GROWTH = 0.30
MIN_BIG5_GROWTH = 0.02

def get_model_growth(fin, big5_eps_values: dict) -> float | None:
    """The 'Model' growth rate: lower of (capped Big 5 EPS) and (capped Analyst).
    Used by both the Stock Analyzer and My Investments pages.
    """
    # 1. Get capped analyst growth
    _ag = None
    _fg = getattr(fin, "analyst_forward_growth", None)
    if _fg is not None:
        _ag = min(_fg, MAX_ANALYST_GROWTH_FORWARD)
    elif getattr(fin, "analyst_5yr_growth", None) is not None:
        _ag = min(fin.analyst_5yr_growth, MAX_ANALYST_GROWTH_TRAILING)

    # 2. Get capped Big 5 EPS growth (median of 10y, 5y, 3y)
    _bg = None
    stable_eps = [v for w, v in big5_eps_values.items() if w in (5, 3, 10) and v is not None]
    if stable_eps:
        _bg = min(max(statistics.median(stable_eps), MIN_BIG5_GROWTH), MAX_BIG5_GROWTH)
    else:
        _fallback = big5_eps_values.get(1) or big5_eps_values.get(3) or big5_eps_values.get(5)
        if _fallback is not None:
            _bg = min(max(_fallback, MIN_BIG5_GROWTH), MAX_BIG5_GROWTH)

    # 3. The Model chooses the lower of the positive Big 5 and Analyst
    cands = [g for g in (_bg, _ag) if g is not None and g > 0]
    return min(cands) if cands else None

# ============================================================
# CENTRALIZED VALUATION METHODS
# ============================================================

from .intrinsic import dcf_two_stage, peter_lynch_fair, graham_number, graham_formula, peg_fair_value

def calculate_median_valuation(
    current_eps: float | None,
    fcf_ttm: float | None,
    shares_outstanding: float | None,
    book_value_per_share: float | None,
    dividend_yield: float | None,
    current_price: float | None,
    growth_rate: float | None,
    historical_pe: float | None = None,  # <-- ADD THIS LINE
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025,
    aaa_bond_yield: float = 0.045,
    mos: float = 0.25,
    **kwargs,                            # <-- ADD THIS LINE to catch anything else
) -> dict:
    """Calculates the 5 standard valuation methods and returns the median."""
    vals = []
    details = {}

    if current_eps and growth_rate:
        # 1. DCF
        if fcf_ttm is not None and fcf_ttm > 0 and shares_outstanding:
            dcf = dcf_two_stage(
                fcf_ttm=fcf_ttm, shares_out=shares_outstanding, current_price=current_price,
                growth_rate=growth_rate, discount_rate=discount_rate, terminal_growth=terminal_growth,
            )
            if dcf and dcf.fair_value:
                vals.append(("DCF", dcf.fair_value))
                details["DCF"] = dcf

        # 2. Peter Lynch
        lynch = peter_lynch_fair(
            current_eps=current_eps, growth_rate=growth_rate,
            dividend_yield=dividend_yield, current_price=current_price, mos=mos,
        )
        if lynch and lynch.fair_value:
            vals.append(("Lynch", lynch.fair_value))
            details["Lynch"] = lynch

        # 3. Buffett DCF (Replaces Graham Number)
        b_dcf = buffett_dcf(
            fcf_ttm=fcf_ttm,
            shares_outstanding=shares_outstanding,
            current_price=current_price,
            growth_rate=growth_rate,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            mos=mos,
        )
        if b_dcf and b_dcf.fair_value:
            vals.append(("Buffett DCF", b_dcf.fair_value))
            details["Buffett DCF"] = b_dcf

        # 4. Graham Formula
        gf = graham_formula(
            current_eps=current_eps, growth_rate=growth_rate,
            current_price=current_price, aaa_bond_yield=aaa_bond_yield, mos=mos,
        )
        if gf and gf.fair_value:
            vals.append(("Graham F", gf.fair_value))
            details["Graham F"] = gf

        # 5. PEG
        peg = peg_fair_value(
            current_eps=current_eps, growth_rate=growth_rate,
            current_price=current_price, mos=mos,
        )
        if peg and peg.fair_value:
            vals.append(("PEG", peg.fair_value))
            details["PEG"] = peg

    if len(vals) >= 2:
        intrinsic = statistics.median(v for _, v in vals)
        method = f"median of {len(vals)}"
    elif len(vals) == 1:
        intrinsic = vals[0][1]
        method = vals[0][0]
    else:
        intrinsic = None
        # If EPS is missing completely, it's likely an ETF. If EPS is just negative, it's a stock we can't value.
        if current_eps is None:
            method = "ETF / Fund"
        elif current_eps <= 0:
            method = "Negative EPS"
        else:
            method = "No models applicable"

    return {"intrinsic_value": intrinsic, "method": method, "details": details}


def calculate_model_valuation(
    current_eps: float | None,
    fcf_ttm: float | None,
    shares_outstanding: float | None,
    book_value_per_share: float | None,
    dividend_yield: float | None,
    current_price: float | None,
    growth_rate: float | None,
    historical_pe: float | None = None,
    **kwargs
) -> dict:
    """The 'Value Price' model strategy (what the Stock Analyzer top card uses)."""
    if not current_eps or current_eps <= 0 or not growth_rate:
        return {"intrinsic_value": None, "method": "ETF / Fund" if current_eps is None else "No growth", "details": {}}

    val = value_price(
        current_eps=current_eps,
        big5_eps_growth=None,
        analyst_growth=None,
        historical_pe=historical_pe,
        custom_growth=growth_rate,
    )
    
    if val and val.value_price:
        return {"intrinsic_value": val.value_price, "method": "Value Price Model", "details": {"Value Price": val}}
    return {"intrinsic_value": None, "method": "Model Failed", "details": {}}


# ============================================================
# 🎯 THE ROUTER — Change this one line to switch strategies!
# ============================================================
VALUATION_METHOD = "model"  # Options: "median" or "model"

def calculate_intrinsic_value(
    current_eps: float | None,
    fcf_ttm: float | None,
    shares_outstanding: float | None,
    book_value_per_share: float | None,
    dividend_yield: float | None,
    current_price: float | None,
    growth_rate: float | None,
    **kwargs
) -> dict:
    """The central entry point for all valuation."""
    if VALUATION_METHOD == "median":
        return calculate_median_valuation(
            current_eps=current_eps, fcf_ttm=fcf_ttm, shares_outstanding=shares_outstanding,
            book_value_per_share=book_value_per_share, dividend_yield=dividend_yield,
            current_price=current_price, growth_rate=growth_rate, **kwargs
        )
    elif VALUATION_METHOD == "model":
        return calculate_model_valuation(
            current_eps=current_eps, fcf_ttm=fcf_ttm, shares_outstanding=shares_outstanding,
            book_value_per_share=book_value_per_share, dividend_yield=dividend_yield,
            current_price=current_price, growth_rate=growth_rate, **kwargs
        )
    else:
        raise ValueError(f"Unknown VALUATION_METHOD: {VALUATION_METHOD}")
def buffett_dcf(
    fcf_ttm: float | None,
    shares_outstanding: float | None,
    current_price: float | None,
    growth_rate: float | None,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025,
    mos: float = 0.25,
):
    """Warren Buffett style Owner Earnings DCF."""
    if not fcf_ttm or fcf_ttm <= 0 or not shares_outstanding or shares_outstanding <= 0:
        return None

    # Buffett never assumes high growth forever. Cap it at 20% for the DCF.
    g = min(max(growth_rate or 0.0, 0.0), 0.20)
    tg = terminal_growth
    r = discount_rate

    # Project 10 years of Free Cash Flow and discount to present value
    pv = 0.0
    for year in range(1, 11):
        fcf_year = fcf_ttm * (1 + g) ** year
        pv += fcf_year / (1 + r) ** year

    # Terminal Value (Gordon Growth Model)
    fcf_year_11 = fcf_ttm * (1 + g) ** 10 * (1 + tg)
    tv = fcf_year_11 / (r - tg)
    pv_tv = tv / (1 + r) ** 10

    total_equity_value = pv + pv_tv
    fair_value = total_equity_value / shares_outstanding
    mos_price = fair_value * (1 - mos)

    upside_pct = (fair_value - current_price) / current_price if current_price else None
    if upside_pct is None:
        verdict = "N/A"
    elif upside_pct >= 0.15:
        verdict = "BUY"
    elif upside_pct >= -0.05:
        verdict = "WATCH"
    else:
        verdict = "AVOID"

    from .intrinsic import MethodResult
    return MethodResult(
        name="Buffett DCF",
        fair_value=fair_value,
        mos_price=mos_price,
        upside_pct=upside_pct,
        verdict=verdict,
        assumptions=(
            f"FCF: ${fcf_ttm/1e9:.2f}B | Growth: {g*100:.1f}% | Discount: {r*100:.0f}% | "
            f"Terminal: {tg*100:.1f}% | Margin of Safety: {mos*100:.0f}%"
        )
    )

def get_ui_growth_rate(mode: str, fin, big5_eps_g: float | None, custom_pct: float = 0.0) -> float | None:
    """Translates the UI radio button selection into a growth rate."""
    if mode == "Model (lower of Big 5 & analyst)":
        return get_model_growth(fin, {"5": big5_eps_g}) # (Passing a dict to match the signature)
    if mode == "Value conservative":
        return 0.15
    if mode == "Big 5 EPS growth":
        return big5_eps_g
    if mode == "Analyst 5Y growth":
        return getattr(fin, "analyst_5yr_growth", None)
    if mode == "Custom":
        return custom_pct / 100.0 if custom_pct > 0 else None
    return None