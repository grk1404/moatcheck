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


@dataclass
class MethodResult:
    name: str
    fair_value: float               # per-share fair value
    mos_price: float                # 50% margin-of-safety buy price
    upside_pct: float | None        # (fair - current) / current
    verdict: str                    # BUY / WATCH / AVOID / UNKNOWN
    assumptions: dict               # what the number depended on


def _verdict(current: float | None, fair: float, mos: float) -> tuple[str, float | None]:
    if current is None or current <= 0:
        return "UNKNOWN", None
    upside = (fair - current) / current
    if current <= mos:
        return "BUY", upside
    if current <= fair:
        return "WATCH", upside
    return "AVOID", upside


def dcf_two_stage(
    fcf_ttm: float | None,
    shares_out: float | None,
    current_price: float | None,
    growth_rate: float | None,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025,
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
    g_high = min(max(growth_rate, -0.05), 0.20)
    if terminal_growth >= discount_rate:
        # Gordon model diverges — clamp terminal to safely below discount rate.
        terminal_growth = discount_rate - 0.005

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

    growth_pct = growth_rate * 100
    div_pct = (dividend_yield or 0) * 100
    fair_pe = growth_pct + div_pct
    # Cap fair PE at 30 — Lynch himself was skeptical of anything above 20-25.
    fair_pe = min(fair_pe, 30)

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
            "growth_rate": growth_rate,
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

    growth_pct = growth_rate * 100
    y_pct = aaa_bond_yield * 100
    fair = current_eps * (8.5 + 2 * growth_pct) * 4.4 / y_pct
    if fair <= 0:
        return None
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
            "growth_rate": growth_rate,
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

    fair_pe = growth_rate * 100
    fair_pe = min(fair_pe, 30)  # cap same as Lynch — extrapolation guardrail
    fair = current_eps * fair_pe
    mos_price = fair * (1 - mos)
    verdict, upside = _verdict(current_price, fair, mos_price)

    peg_current = None
    if current_price and current_price > 0:
        pe_now = current_price / current_eps
        peg_current = pe_now / (growth_rate * 100) if growth_rate > 0 else None

    return MethodResult(
        name="PEG (fair @ PEG=1)",
        fair_value=fair,
        mos_price=mos_price,
        upside_pct=upside,
        verdict=verdict,
        assumptions={
            "current_eps": current_eps,
            "growth_rate": growth_rate,
            "fair_pe_used": fair_pe,
            "peg_ratio_now": peg_current,
            "margin_of_safety": mos,
        },
    )
