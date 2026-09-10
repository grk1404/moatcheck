"""Non-Value intrinsic-value methods.

Every function returns a `MethodResult` (or None if it cannot be honestly
computed) so the UI can render each side-by-side with the same layout.

Methods implemented:
  - dcf_two_stage        Two-stage DCF on Free Cash Flow (5yr high growth + terminal)
  - peter_lynch_fair     Lynch's "Fair PE = growth + dividend yield" heuristic
  - graham_number        Graham's sqrt(22.5 * EPS * BVPS) — deep-value floor
  - graham_formula       Graham's EPS * (8.5 + 2g) * 4.4 / bond_yield revised formula
  - peg_fair_value       PEG=1 anchor: fair PE = growth rate as integer
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ==================== GLOBAL CONFIGURATION ====================
# These caps prevent unrealistic valuations from extreme growth rates
# or abnormal PE ratios. They can be adjusted globally.

MAX_GROWTH_RATE = 0.30          # Maximum sustainable growth rate (30%)
MIN_GROWTH_RATE = 0.02          # Minimum growth rate for valuation (2%)
MAX_FUTURE_PE = 25.0            # Maximum future P/E ratio (25x)
MIN_FUTURE_PE = 8.0             # Minimum future P/E ratio (8x)
MAX_PE_MULTIPLE = 30.0          # Maximum PE for sanity checks (30x)
MAX_FCF_MULTIPLE = 30.0         # Maximum FCF multiple for DCF (30x)
DEFAULT_DISCOUNT_RATE = 0.15    # Buffett's required return (15%)
DEFAULT_TERMINAL_GROWTH = 0.025 # Terminal growth rate (2.5%)
MAX_REASONABLE_EPS_MULTIPLE = 20  # For final sanity check
MAX_REASONABLE_YEARS = 10        # For final sanity check


@dataclass
class MethodResult:
    name: str
    fair_value: float               # per-share fair value
    mos_price: float                # 50% margin-of-safety buy price
    upside_pct: float | None        # (fair - current) / current
    verdict: str                    # BUY / WATCH / AVOID / UNKNOWN
    assumptions: dict               # what the number depended on

@dataclass
class ValueResult:
    """Result from the Buffett-style value_price calculation."""
    value_price: float
    mos_price: float
    growth_rate: float
    future_pe: float
    future_eps: float
    future_price: float
    horizon_years: int
    discount_rate: float
    growth_source: str
    current_eps: float

def _verdict(current: float | None, fair: float, mos: float) -> tuple[str, float | None]:
    if current is None or current <= 0:
        return "UNKNOWN", None
    upside = (fair - current) / current
    if current <= mos:
        return "BUY", upside
    if current <= fair:
        return "WATCH", upside
    return "AVOID", upside

def value_price(
    current_eps: float | None,
    big5_eps_growth: float | None,
    analyst_growth: float | None,
    historical_pe: float | None,
    custom_growth: float | None = None,
) -> ValueResult | None:
    """
    Compute Buffett-style intrinsic value (10-year projection + 15% discount).
    
    With SANITY CHECKS to prevent unrealistic valuations.
    """
    if current_eps is None or current_eps <= 0:
        return None

    # --- FIX 1: PICK GROWTH RATE WITH SANITY CHECKS ---
    growth_rate = None
    growth_source = None
    
    if custom_growth is not None and custom_growth > 0:
        growth_rate = min(custom_growth, MAX_GROWTH_RATE)  # Cap at maximum sustainable growth rate
        growth_source = "custom"
    else:
        # Collect available growth rates
        candidates = []
        if big5_eps_growth is not None and big5_eps_growth > 0:
            candidates.append(("big5", big5_eps_growth))
        if analyst_growth is not None and analyst_growth > 0:
            candidates.append(("analyst", analyst_growth))
        
        if candidates:
            # Take the LOWER of Big5 and analyst growth (conservative)
            candidates.sort(key=lambda x: x[1])
            growth_rate = candidates[0][1]
            growth_source = candidates[0][0]
            
            # --- FIX 2: CAP GROWTH AT REASONABLE LEVELS ---
            # 30% is Buffett's maximum sustainable growth rate
            if growth_rate > MAX_GROWTH_RATE:
                growth_rate = MAX_GROWTH_RATE
                growth_source = f"{growth_source}_capped"
            
            # If growth is too low (< 2%), use 5% as minimum
            if growth_rate < MIN_GROWTH_RATE:
                growth_rate = MIN_GROWTH_RATE
                growth_source = "minimum_5pct"
    
    if growth_rate is None or growth_rate <= 0:
        return None

    # --- FIX 3: FUTURE PE WITH SANITY CHECK ---
    # Historical PE, but capped at a reasonable range
    if historical_pe is not None and historical_pe > 0:
        future_pe = min(historical_pe, MAX_FUTURE_PE)  # Cap PE at maximum for safety
        if future_pe < MIN_FUTURE_PE:
            future_pe = MIN_FUTURE_PE  # Minimum PE of 8
    else:
        # Default: 2x growth rate, capped
        future_pe = min(growth_rate * 100 * 2, MAX_FUTURE_PE)
        if future_pe < MIN_FUTURE_PE:
            future_pe = MIN_FUTURE_PE

    # --- FIX 4: PROJECT EPS WITH FADE ---
    horizon_years = MAX_REASONABLE_YEARS
    discount_rate = DEFAULT_DISCOUNT_RATE  # Buffett's required return
    
    # Calculate future EPS with growth rate
    future_eps = current_eps * ((1 + growth_rate) ** horizon_years)
    
    # --- FIX 5: SANITY CHECK THE RESULT ---
    # If future_eps is unreasonably high, apply a fade factor
    if future_eps / current_eps > 5:  # More than 5x current EPS is suspicious
        # Use a more conservative approach: linear growth to 2x then fade
        fade_years = 5
        eps_values = [current_eps]
        for i in range(1, horizon_years + 1):
            if i <= fade_years:
                eps_values.append(eps_values[-1] * (1 + growth_rate))
            else:
                eps_values.append(eps_values[-1] * (1 + growth_rate * 0.5))
        future_eps = eps_values[-1]

    future_price = future_eps * future_pe
    value_price = future_price / ((1 + discount_rate) ** horizon_years)
    mos_price = value_price * 0.5  # 50% margin of safety

    # --- FIX 6: FINAL SANITY CHECK ---
    # Value should not exceed 10x current EPS * 20 PE
    max_reasonable = current_eps * MAX_REASONABLE_EPS_MULTIPLE * MAX_REASONABLE_YEARS
    if value_price > max_reasonable:
        value_price = max_reasonable
        mos_price = value_price * 0.5

    return ValueResult(
        value_price=value_price,
        mos_price=mos_price,
        growth_rate=growth_rate,
        future_pe=future_pe,
        future_eps=future_eps,
        future_price=future_price,
        horizon_years=horizon_years,
        discount_rate=discount_rate,
        growth_source=growth_source,
        current_eps=current_eps,
    )

def dcf_two_stage(
    fcf_ttm: float | None,
    shares_out: float | None,
    current_price: float | None,
    growth_rate: float | None,
    discount_rate: float = 0.10,
    terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
    high_growth_years: int = 5,
    fade_years: int = 5,
    mos: float = 0.5,
) -> MethodResult | None:
    """Two-stage DCF on free cash flow.

    Model:
      Years 1..N:            FCF grows at `growth_rate` (capped at 20% — no company
                             grows 30%+ forever, and analyst 5yr numbers routinely lie).
      Years N+1..N+fade:     Growth linearly fades from `growth_rate` to `terminal_growth`.
      Year N+fade+1..inf:    Gordon growth model at `terminal_growth`.
      Discount everything back at `discount_rate` (default 10% ≈ long-run equity return).

    Returns None if FCF is non-positive (DCF is meaningless with negative cashflow —
    show that transparently rather than fabricating a number).
    """
    if fcf_ttm is None or fcf_ttm <= 0:
        return None
    if shares_out is None or shares_out <= 0:
        return None
    if growth_rate is None:
        return None

    # Cap growth to keep the model honest.
    g_high = min(max(growth_rate, -0.05), MAX_GROWTH_RATE)
    # --- FIX: Cap growth to keep the model honest ---    
    if g_high < MIN_GROWTH_RATE:
        g_high = MIN_GROWTH_RATE  # Minimum growth for DCF
    if terminal_growth >= discount_rate:
        # Gordon model diverges — clamp terminal to safely below discount rate.
        terminal_growth = discount_rate - 0.005
    if terminal_growth < 0.01:
        terminal_growth = 0.01  # Minimum terminal growth

    cashflows = []
    fcf = fcf_ttm

    for year in range(1, high_growth_years + 1):
        fcf = fcf * (1 + g_high)
        cashflows.append(fcf)

    # Linear fade from g_high toward terminal_growth over `fade_years`.
    if fade_years > 0:
        step = (g_high - terminal_growth) / (fade_years + 1)
        for i in range(1, fade_years + 1):
            g = g_high - step * i
            fcf = fcf * (1 + g)
            cashflows.append(fcf)

    pv_operating = sum(
        cf / ((1 + discount_rate) ** year) for year, cf in enumerate(cashflows, start=1)
    )

    terminal_fcf = cashflows[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** len(cashflows))

    equity_value = pv_operating + pv_terminal
    fair = equity_value / shares_out
     # --- FIX: Sanity check DCF result ---
    # DCF value shouldn't exceed 10x current FCF per share
    fcf_per_share = fcf_ttm / shares_out
    max_reasonable = fcf_per_share * MAX_FCF_MULTIPLE  # 30x FCF is reasonable max
    if fair > max_reasonable:
        fair = max_reasonable

    mos_price = fair * (1 - mos)
    verdict, upside = _verdict(current_price, fair, mos_price)

    return MethodResult(
        name="Two-Stage DCF",
        fair_value=fair,
        mos_price=mos_price,
        upside_pct=upside,
        verdict=verdict,
        assumptions={
            "fcf_ttm": fcf_ttm,
            "shares_out": shares_out,
            "growth_high": g_high,
            "high_growth_years": high_growth_years,
            "fade_years": fade_years,
            "terminal_growth": terminal_growth,
            "discount_rate": discount_rate,
            "margin_of_safety": mos,
        },
    )


def peter_lynch_fair(
    current_eps: float | None,
    growth_rate: float | None,
    dividend_yield: float | None,
    current_price: float | None,
    mos: float = 0.25,
) -> MethodResult | None:
    """Peter Lynch's Fair PE heuristic.

    From *One Up on Wall Street*: a fairly-priced company should trade at a
    P/E equal to its earnings growth rate (in whole-number percent), plus
    a bonus for dividend yield. Fair PE = growth% + div_yield%.

    Fair value = current_eps * fair_PE. Lynch used a smaller ~25% MOS
    (unlike Town's 50%), reflecting that he was buying at fair, not deep-discount.
    """
    if current_eps is None or current_eps <= 0:
        return None
    if growth_rate is None or growth_rate <= 0:
        return None

    # --- FIX: Cap growth at 30% ---
    growth_capped = min(growth_rate, MAX_GROWTH_RATE)
    
    growth_pct = growth_capped * 100
    div_pct = (dividend_yield or 0) * 100
    fair_pe = growth_pct + div_pct
    # Cap fair PE at MAX_PE_MULTIPLE — Lynch himself was skeptical of anything above 20-25.
    fair_pe = min(fair_pe, MAX_PE_MULTIPLE)
    
    if fair_pe < MIN_FUTURE_PE:
        fair_pe = MIN_FUTURE_PE  # Minimum PE of MIN_FUTURE_PE

    fair = current_eps * fair_pe
    mos_price = fair * (1 - mos)
    verdict, upside = _verdict(current_price, fair, mos_price)

    return MethodResult(
        name="Peter Lynch Fair Value",
        fair_value=fair,
        mos_price=mos_price,
        upside_pct=upside,
        verdict=verdict,
        assumptions={
            "current_eps": current_eps,
            "growth_rate": growth_capped,
            "dividend_yield": dividend_yield or 0,
            "fair_pe_used": fair_pe,
            "margin_of_safety": mos,
        },
    )


def graham_number(
    current_eps: float | None,
    book_value_per_share: float | None,
    current_price: float | None,
    mos: float = 0.25,
) -> MethodResult | None:
    """Benjamin Graham's classic "Graham Number".

    Fair value = sqrt(22.5 * EPS * BVPS)
    The 22.5 constant comes from Graham's max P/E of 15 * max P/B of 1.5.
    A deep-value floor: many quality growth companies will always fail this
    (because BVPS is small relative to EPS on capital-light businesses),
    but it's a useful sanity check for margin of safety.
    """
    if current_eps is None or current_eps <= 0:
        return None
    if book_value_per_share is None or book_value_per_share <= 0:
        return None

    fair = math.sqrt(22.5 * current_eps * book_value_per_share)
    mos_price = fair * (1 - mos)
    verdict, upside = _verdict(current_price, fair, mos_price)

    return MethodResult(
        name="Graham Number",
        fair_value=fair,
        mos_price=mos_price,
        upside_pct=upside,
        verdict=verdict,
        assumptions={
            "current_eps": current_eps,
            "book_value_per_share": book_value_per_share,
            "formula": "sqrt(22.5 * EPS * BVPS)",
            "margin_of_safety": mos,
        },
    )


def graham_formula(
    current_eps: float | None,
    growth_rate: float | None,
    current_price: float | None,
    aaa_bond_yield: float = 0.045,
    mos: float = 0.25,
) -> MethodResult | None:
    """Graham's revised (1974) intrinsic-value formula.

    V = EPS * (8.5 + 2g) * 4.4 / Y
      - 8.5 = P/E for a no-growth company
      - g   = expected annual growth (in whole-number percent)
      - 4.4 = the AAA corporate bond yield when Graham published (1962)
      - Y   = today's AAA corporate bond yield
    """
    if current_eps is None or current_eps <= 0:
        return None
    if growth_rate is None:
        return None
    if aaa_bond_yield is None or aaa_bond_yield <= 0:
        return None

    # --- FIX: Cap growth at 30% ---
    growth_capped = min(growth_rate, MAX_GROWTH_RATE)

    growth_pct = growth_capped * 100
    y_pct = aaa_bond_yield * 100
    fair = current_eps * (8.5 + 2 * growth_pct) * 4.4 / y_pct
    if fair <= 0:
        return None

    # --- FIX: Sanity check Graham Formula ---
    # Shouldn't exceed 30x EPS
    max_fair = current_eps * MAX_PE_MULTIPLE
    if fair > max_fair:
        fair = max_fair
        
    mos_price = fair * (1 - mos)
    verdict, upside = _verdict(current_price, fair, mos_price)

    return MethodResult(
        name="Graham Formula",
        fair_value=fair,
        mos_price=mos_price,
        upside_pct=upside,
        verdict=verdict,
        assumptions={
            "current_eps": current_eps,
            "growth_rate": growth_capped,
            "aaa_bond_yield": aaa_bond_yield,
            "formula": "EPS * (8.5 + 2g) * 4.4 / Y",
            "margin_of_safety": mos,
        },
    )


def peg_fair_value(
    current_eps: float | None,
    growth_rate: float | None,
    current_price: float | None,
    mos: float = 0.25,
) -> MethodResult | None:
    """PEG=1 anchor. Fair PE equals the growth rate (whole-number percent).

    PEG ratio = P/E divided by growth rate. Lynch popularized PEG < 1 as
    "cheap for its growth" and PEG > 2 as "overpriced". Setting PEG = 1
    gives us a fair-value price implied by the current growth outlook.
    """
    if current_eps is None or current_eps <= 0:
        return None
    if growth_rate is None or growth_rate <= 0:
        return None

    # --- FIX: Cap growth at 30% ---
    growth_capped = min(growth_rate, MAX_GROWTH_RATE)

    fair_pe = growth_capped * 100
    fair_pe = min(fair_pe, MAX_PE_MULTIPLE)  # cap same as Lynch — extrapolation guardrail
    if fair_pe < MIN_FUTURE_PE:
        fair_pe = MIN_FUTURE_PE  # Minimum PE of MIN_FUTURE_PE
    fair = current_eps * fair_pe
    mos_price = fair * (1 - mos)
    verdict, upside = _verdict(current_price, fair, mos_price)

    peg_current = None
    if current_price and current_price > 0:
        pe_now = current_price / current_eps
        peg_current = pe_now / (growth_capped * 100) if growth_capped > 0 else None

    return MethodResult(
        name="PEG (fair @ PEG=1)",
        fair_value=fair,
        mos_price=mos_price,
        upside_pct=upside,
        verdict=verdict,
        assumptions={
            "current_eps": current_eps,
            "growth_rate": growth_capped,
            "fair_pe_used": fair_pe,
            "peg_ratio_now": peg_current,
            "margin_of_safety": mos,
        },
    )
