from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import math

from .fetcher import Financials
from .growth import cagr, window_average

WINDOWS = (10, 5, 3, 1)
PASS_THRESHOLD = 0.10  # 10% is the bar for "wonderful" growth rate in any metric.
CONSISTENCY_VOLATILITY_CEILING = 0.6  # avg coefficient-of-variation at/above this -> consistency = 0 RG


# ============================================================
# Threshold resolution
# ============================================================

# Canonical per-metric threshold keys. Each maps to a metric in Big5Result.
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "revenue": {"min": 0.05, "max": 0.50},
    "eps":     {"min": 0.05, "max": 0.50},
    "equity":  {"min": 0.05, "max": 0.50},
    "ocf":     {"min": 0.05, "max": 0.50},
    "roic":    {"min": 0.05, "max": 0.50},
    # New floors — single-sided (min only). max=1.0 disables the upper bound.
    "cash_conversion": {"min": 0.80, "max": 999.0},
    "discount_fcf":    {"min": 0.30, "max": 999.0},
    "discount_np":     {"min": 0.30, "max": 999.0},
}


def resolve_thresholds(user_thresholds: dict | None) -> dict[str, float]:
    """Merge user-supplied thresholds over the defaults.

    Accepts keys: 'revenue', 'eps', 'equity', 'ocf', 'roic'.
    Unknown keys are ignored. Values must be floats in decimal form (0.10 = 10%).
    """
    resolved = dict(DEFAULT_THRESHOLDS)
    if not user_thresholds:
        return resolved
    for k, v in user_thresholds.items():
        if k in resolved and v is not None:
            try:
                resolved[k] = float(v)
            except (TypeError, ValueError):
                continue
    return resolved


# ============================================================
# Result containers
# ============================================================

@dataclass
class MetricResult:
    label: str
    values: dict[int, float | None]  # window years -> rate
    passes: bool
    unit: str = "pct"  # "pct" or "ratio" (ROIC displayed as pct too)

    def as_row(self) -> dict[str, Any]:
        row = {"metric": self.label}
        for w in WINDOWS:
            row[f"{w}yr"] = self.values.get(w)
        row["pass"] = self.passes
        return row


@dataclass
class WonderfulnessScore:
    """A 0-10 scale answer to 'is this a wonderful company?'.

    Composed of three sub-scores that each capture a different quality dimension:
      - pass_rate: how many of the 20 checks pass the threshold (breadth)
      - magnitude: how far above the threshold the metrics actually sit (height)
      - consistency: how tight the range is across windows (durability)

    The overall score is a weighted blend. 10 is theoretically possible
    only for a compounder that beats the threshold by a wide margin in every
    window with almost no variance — very rare (MSFT-class businesses).
    """
    overall: float
    label: str
    color: str
    pass_rate: float
    magnitude: float
    consistency: float
    consistency_reason: str
    checks_passed: int
    checks_total: int
    strengths: list[str]
    weaknesses: list[str]
    trend: float


def _label_for_score(s: float) -> tuple[str, str]:
    if s >= 9.0:
        return ("Wonderful", "bargain")
    if s >= 7.0:
        return ("Very Good", "green")
    if s >= 5.0:
        return ("Decent", "orange")
    if s >= 3.0:
        return ("Mediocre", "orange")
    return ("Poor", "red")


import math


def _growth_score(values: list[float], threshold: dict[str, float]) -> float:
    """Convert a metric's average growth rate into a 0-10 score.

    Scoring anchors:
      - 0 at avg_growth = 0
      - 5 at avg_growth = rmin
      - 10 at avg_growth = rmax (or 2×rmin for min-only metrics)

    Special cases:
      - rmax >= 900 is treated as "no upper bound" (sentinel used by
        min-only metrics like Cash Conversion, Discount FCF, Discount NP).
        In that case the score ramps from 5.0 at rmin to 10.0 at 2×rmin.
        Any value at or above 2×rmin scores 10.
      - rmin == 0 and rmax == 0 returns 0.0 (no meaningful target).
      - rmin == 0 with rmax > 0 scores linearly from 0 at zero growth
        to 10 at rmax.
      - NaN values are filtered before scoring.

    The min-only ramp fixes the "Infinity Squash": previously a sentinel
    rmax of 999.0 caused every value between rmin and 999 to score ~5.0,
    which capped the magnitude sub-score and dragged the overall score
    down for every company with strong quality metrics.
    """
    if not values:
        return 0.0

    # Filter NaN — a NaN passes `is not None` but breaks every comparison
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return 0.0

    avg_growth = sum(values) / len(values)
    if avg_growth <= 0:
        return 0.0

    rmin = threshold.get("min", 0.0)
    rmax = threshold.get("max", 0.0)

    # ---- Degenerate: no meaningful threshold ----
    if rmin <= 0 and rmax <= 0:
        return 0.0

    # ---- Min-only metric (sentinel rmax) ----
    if rmax >= 900.0:
        if rmin <= 0:
            return 0.0
        cap = 2.0 * rmin
        if avg_growth >= cap:
            return 10.0
        if avg_growth >= rmin:
            return 5.0 + (avg_growth - rmin) / (cap - rmin) * 5.0
        return 5.0 * (avg_growth / rmin) ** 2      # ← change

    # ---- Bounded metric ----
    if rmax <= rmin:
        if rmin <= 0:
            return 0.0
        return 10.0 if avg_growth >= rmin else avg_growth / rmin * 10.0

    # NEW: rmin == 0 with rmax > 0 — linear ramp from 0 to 10
    if rmin <= 0:
        return min(10.0, avg_growth / rmax * 10.0)

    # Standard bounded scaling (rmin > 0 and rmax > rmin)
    if avg_growth >= rmax:
        return 10.0
    if avg_growth >= rmin:
        return 5.0 + (avg_growth - rmin) / (rmax - rmin) * 5.0
    return 5.0 * (avg_growth / rmin) ** 2


def _trend_penalty(values: dict[int, float]) -> float:
    """Score 0-10 based on whether growth is accelerating or decelerating."""
    ordered = [values[w] for w in sorted(values, reverse=True) if values[w] is not None]
    if len(ordered) < 2:
        return 5.0
    declines = sum(1 for a, b in zip(ordered, ordered[1:]) if b < a)
    fraction_decelerating = declines / (len(ordered) - 1)
    return max(0.0, min(10.0, (1 - fraction_decelerating) * 10))


# ============================================================
# Wonderfulness scoring
# ============================================================

def score_wonderfulness(big5: "Big5Result") -> WonderfulnessScore:
    """Compute a 0-10 wonderfulness score from a Big5Result.

    Thresholds are read from big5.thresholds (set by compute_big5).
    """
    metrics = [big5.roic, big5.sales, big5.eps, big5.equity, big5.fcf]
    for extra in (big5.cash_conversion, big5.discount_fcf, big5.discount_np):
        if extra is not None:
            metrics.append(extra)
    thresholds = getattr(big5, "thresholds", DEFAULT_THRESHOLDS)

    # Map metric object -> its threshold key
    metric_threshold_map = {
        "roic":            thresholds.get("roic",            DEFAULT_THRESHOLDS["roic"]),
        "sales":           thresholds.get("revenue",         DEFAULT_THRESHOLDS["revenue"]),
        "eps":             thresholds.get("eps",             DEFAULT_THRESHOLDS["eps"]),
        "equity":          thresholds.get("equity",          DEFAULT_THRESHOLDS["equity"]),
        "fcf":             thresholds.get("ocf",             DEFAULT_THRESHOLDS["ocf"]),
        "cash_conversion": thresholds.get("cash_conversion", DEFAULT_THRESHOLDS["cash_conversion"]),
        "discount_fcf":    thresholds.get("discount_fcf",    DEFAULT_THRESHOLDS["discount_fcf"]),
        "discount_np":     thresholds.get("discount_np",     DEFAULT_THRESHOLDS["discount_np"]),
    }
    metric_key_order = ["roic", "sales", "eps", "equity", "fcf",
                        "cash_conversion", "discount_fcf", "discount_np"]

    # ---------- Pass rate ----------
    computed = []
    for i, m in enumerate(metrics):
        key = metric_key_order[i]
        thr = metric_threshold_map[key]
        for w, v in m.values.items():
            if v is not None and not math.isnan(v):
                computed.append((m, w, v, thr))
    checks_total = len(computed)
    checks_passed = sum(1 for _, _, v, thr in computed if thr["min"] <= v <= thr["max"])
    pass_rate = (checks_passed / checks_total * 10) if checks_total else 0.0

    # ---------- Magnitude ----------
    metric_scores = []
    for i, m in enumerate(metrics):
        key = metric_key_order[i]
        thr = metric_threshold_map[key]
        vals = [v for v in m.values.values() if v is not None and not math.isnan(v)]
        if vals:
            metric_scores.append(_growth_score(vals, threshold=thr))
    magnitude = (sum(metric_scores) / len(metric_scores)) if metric_scores else 0.0

        # ---------- Consistency ----------
    per_metric_consistency = []
    for i, m in enumerate(metrics):
        key = metric_key_order[i]
        thr = metric_threshold_map[key]
        vals = [v for v in m.values.values() if v is not None and not math.isnan(v)]
        if len(vals) >= 2:
            failing_windows = [v for v in vals if not (thr["min"] <= v <= thr["max"])]
            if not failing_windows:
                per_metric_consistency.append(10.0)
            else:
                shortfalls = []
                for v in failing_windows:
                    if v < thr["min"]:
                        shortfalls.append(thr["min"] - v)
                    else:
                        shortfalls.append(v - thr["max"])
                        # ↓↓↓ THESE THREE LINES ARE THE BUG ↓↓↓
                        avg_shortfall = sum(shortfalls) / len(vals) if shortfalls else 0.0
                        bound_range = thr["max"] - thr["min"]
                        score = max(0.0, (1.0 - (avg_shortfall / bound_range)) * 10.0) if bound_range > 0 else 0.0
                        per_metric_consistency.append(score)
                    # ↑↑↑ THEY'RE INSIDE THE `else`, SO THEY ONLY RUN FOR ABOVE-MAX FAILURES ↑↑↑
    consistency = (sum(per_metric_consistency) / len(per_metric_consistency)) if per_metric_consistency else 0.0
    # Human-readable consistency reason (based on the computed consistency score)
    if consistency >= 9.0:
        c_reason = "Exceptional stability across all 4 time windows (10yr, 5yr, 3yr, 1yr)."
    elif consistency >= 7.0:
        c_reason = "Strong growth consistency with only minor single-period dips below threshold."
    elif consistency >= 4.0:
        c_reason = "Moderate stability; growth slowed down or varied in shorter windows."
    else:
        c_reason = "Uneven growth trajectory; multiple time windows missed the benchmark."

    # ---------- Trend ----------
    trend_scores = []
    for m in metrics:
        if len([v for v in m.values.values() if v is not None]) >= 2:
            trend_scores.append(_trend_penalty(m.values))
    trend = (sum(trend_scores) / len(trend_scores)) if trend_scores else 5.0

    # ---------- Overall ----------
    overall = 0.55 * magnitude + 0.25 * pass_rate + 0.10 * consistency + 0.10 * trend
    label, color = _label_for_score(overall)

    # ---------- Strengths / weaknesses ----------
    metric_scores_by_name = []
    for i, m in enumerate(metrics):
        key = metric_key_order[i]
        thr = metric_threshold_map[key]
        vals = [v for v in m.values.values() if v is not None and not math.isnan(v)]
        if vals:
            metric_scores_by_name.append((m.label, sum(vals) / len(vals), thr))
    metric_scores_by_name.sort(key=lambda t: t[1], reverse=True)
    strengths = [
        f"{name} avg {avg*100:.0f}%"
        for name, avg, thr in metric_scores_by_name[:2]
        if thr["min"] <= avg <= thr["max"]
    ]
    weaknesses = [
        f"{name} avg {avg*100:.0f}%"
        for name, avg, thr in metric_scores_by_name[-2:][::-1]
        if not (thr["min"] <= avg <= thr["max"])
    ]

    return WonderfulnessScore(
        overall=round(overall, 1),
        label=label,
        color=color,
        pass_rate=round(pass_rate, 1),
        magnitude=round(magnitude, 1),
        consistency=round(consistency, 1),
        consistency_reason=c_reason,
        checks_passed=checks_passed,
        checks_total=checks_total,
        strengths=strengths,
        weaknesses=weaknesses,
        trend=round(trend, 1),
    )


# ============================================================
# Big5Result
# ============================================================

@dataclass
class Big5Result:
    roic: MetricResult
    sales: MetricResult
    eps: MetricResult
    equity: MetricResult
    fcf: MetricResult
    roic_by_year: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    # New quality metrics
    cash_conversion: MetricResult | None = None
    discount_fcf: MetricResult | None = None
    discount_np: MetricResult | None = None

    def all_pass(self) -> bool:
        base = (self.roic, self.sales, self.eps, self.equity, self.fcf)
        extras = tuple(m for m in (self.cash_conversion, self.discount_fcf, self.discount_np) if m is not None)
        return all(m.passes for m in base + extras)

    def as_dataframe(self) -> pd.DataFrame:
        metrics = [self.roic, self.sales, self.eps, self.equity, self.fcf]
        for extra in (self.cash_conversion, self.discount_fcf, self.discount_np):
            if extra is not None:
                metrics.append(extra)
        rows = [m.as_row() for m in metrics]
        return pd.DataFrame(rows)

    def wonderfulness(self) -> WonderfulnessScore:
        return score_wonderfulness(self)


# ============================================================
# Metric constructors
# ============================================================

def _roic_per_year(fin: Financials) -> pd.Series:
    """ROIC = NOPAT / (Equity + Long-term Debt)."""
    if fin.ebit.empty or fin.tax_rate.empty:
        nopat = fin.net_income
    else:
        aligned = pd.concat([fin.ebit, fin.tax_rate], axis=1, join="inner")
        aligned.columns = ["ebit", "tax"]
        nopat = aligned["ebit"] * (1 - aligned["tax"])
        if nopat.empty:
            nopat = fin.net_income

    if fin.long_term_debt.empty:
        invested = fin.equity
    else:
        invested = fin.equity.add(fin.long_term_debt, fill_value=0)

    both = pd.concat([nopat, invested], axis=1, join="inner")
    both.columns = ["nopat", "invested"]
    both = both[both["invested"] > 0]
    if both.empty:
        return pd.Series(dtype=float)
    return both["nopat"] / both["invested"]

def _cash_conversion_ratio(fin: Financials, years: int = 10) -> float | None:
    """
    Average OCF / Net Income over the last `years` fiscal years.

    A ratio >= 1.0 means the company turns every dollar of accounting profit
    into at least a dollar of operating cash. Ratios below ~0.8 suggest
    earnings quality issues (aggressive accruals, receivables buildup).
    Returns None if either series is empty or all net income is <= 0.
    """
    if fin.operating_cash_flow.empty or fin.net_income.empty:
        return None

    aligned = pd.concat(
        [fin.operating_cash_flow, fin.net_income],
        axis=1,
        join="inner",
    )
    aligned.columns = ["ocf", "ni"]
    aligned = aligned[aligned["ni"] > 0].tail(years)

    if aligned.empty:
        return None

    ratios = aligned["ocf"] / aligned["ni"]
    ratios = ratios.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if ratios.empty:
        return None

    return float(ratios.mean())


def _discount_to_fair_value(
    fin: Financials,
    growth_rate: float | None,
    terminal_multiple: float = 15.0,
    discount_rate: float = 0.12,
    growth_cap: float = 0.20,
) -> float | None:
    """
    Simplified intrinsic-value estimate. Returns the discount to fair value
    as a positive number (0.30 = current price is 30% below fair value),
    or None if the inputs don't allow a computation.

    Methodology (matches the StockInvestorIQ "Discount to Fair Value" FAQ):
      - Take the growth rate (historical CAGR or NP growth), cap at 20%
      - Project the current per-share figure forward 10 years at that rate
      - Apply a terminal multiple of 15x to get the future per-share value
      - Discount back 10 years at 12% to get the present fair value
      - Compare against the current stock price
    """
    if fin.current_price is None or fin.current_price <= 0:
        return None
    if growth_rate is None:
        return None

    g = min(growth_rate, growth_cap)
    if g <= 0:
        return None

    # Determine the per-share base: prefer FCF/share if caller passed FCF growth;
    # caller is responsible for passing the right growth rate.
    # Use EPS TTM as the base in both cases — it's the cleanest per-share figure
    # and matches how the site's FCF/NP discounts converge at the fair-value line.
    if fin.eps.empty:
        return None
    base = float(fin.eps.iloc[-1])
    if base <= 0:
        # Fall back to 3-year positive average
        positive = fin.eps[fin.eps > 0]
        if positive.empty:
            return None
        base = float(positive.tail(3).mean())

    future_per_share = base * ((1 + g) ** 10) * terminal_multiple
    fair_value = future_per_share / ((1 + discount_rate) ** 10)

    discount = (fair_value - fin.current_price) / fair_value
    return float(discount)

def _metric_from_cagr(label: str, series: pd.Series, threshold: dict[str, float]) -> MetricResult:
    raw = {w: cagr(series, w) for w in WINDOWS}
    # Normalize NaN to None so downstream filters ("is not None") catch it
    values = {w: (None if v is None or math.isnan(v) else v) for w, v in raw.items()}
    computed = [v for v in values.values() if v is not None]
    rmin = threshold["min"]
    rmax = threshold["max"]
    passes = bool(computed) and all(rmin <= v <= rmax for v in computed)
    return MetricResult(label=label, values=values, passes=passes)


def _metric_from_average(label: str, series: pd.Series, threshold: dict[str, float]) -> MetricResult:
    raw = {w: window_average(series, w) for w in WINDOWS}
    values = {w: (None if v is None or math.isnan(v) else v) for w, v in raw.items()}
    computed = [v for v in values.values() if v is not None]
    rmin = threshold["min"]
    rmax = threshold["max"]
    passes = bool(computed) and all(rmin <= v <= rmax for v in computed)
    return MetricResult(label=label, values=values, passes=passes)


def _bvps(fin: Financials) -> pd.Series:
    """Book value per share = equity / diluted shares."""
    if hasattr(fin, 'bvps_series') and not fin.bvps_series.empty:
        return fin.bvps_series

    if fin.shares.empty:
        return fin.equity
    aligned = pd.concat([fin.equity, fin.shares], axis=1, join="inner")
    aligned.columns = ["eq", "sh"]
    aligned = aligned[aligned["sh"] > 0]
    if aligned.empty:
        return fin.equity
    return aligned["eq"] / aligned["sh"]


# ============================================================
# Public entry point
# ============================================================

def compute_big5(
    fin: Financials,
    thresholds: dict | None = None,
) -> Big5Result:
    """
    Compute the Big 5 growth metrics.

    Parameters
    ----------
    fin : Financials
        Fetched company data.
    thresholds : dict, optional
        Per-metric pass thresholds in decimal form.
        Keys: 'revenue', 'eps', 'equity', 'ocf', 'roic'.
        Example: {'revenue': 0.05, 'eps': 0.10, ...}
        Missing keys default to PASS_THRESHOLD (10%).

    Returns
    -------
    Big5Result with metric rows, ROIC series, and resolved thresholds.
    """
    resolved = resolve_thresholds(thresholds)

    roic_series = _roic_per_year(fin)
    bvps_series = _bvps(fin)

# Big 5 core metrics
    roic_m = _metric_from_average("ROIC growth rate", roic_series, threshold=resolved["roic"])
    sales_m = _metric_from_cagr("Sales growth rate", fin.revenue, threshold=resolved["revenue"])
    eps_m = _metric_from_cagr("EPS growth rate", fin.eps, threshold=resolved["eps"])
    equity_m = _metric_from_cagr("Equity (BVPS) growth rate", bvps_series, threshold=resolved["equity"])
    fcf_m = _metric_from_cagr("Free Cash Flow growth rate", fin.free_cash_flow, threshold=resolved["ocf"])

    # Cash Conversion Ratio (10yr average)
    cash_conv_value = _cash_conversion_ratio(fin, years=10)
    if cash_conv_value is not None and math.isnan(cash_conv_value):
        cash_conv_value = None
    cash_conv_metric = None
    if cash_conv_value is not None:
        cc_thr = resolved["cash_conversion"]
        cc_passes = cc_thr["min"] <= cash_conv_value <= cc_thr["max"]
        cash_conv_metric = MetricResult(
            label="Cash Conversion Ratio",
            values={10: cash_conv_value, 5: cash_conv_value, 3: cash_conv_value, 1: cash_conv_value},
            passes=cc_passes,
            unit="ratio",
        )

    # Discount to Fair Value (FCF) — uses FCF CAGR from the Big 5 as the growth input
    fcf_growth = fcf_m.values.get(10) or fcf_m.values.get(5) or fcf_m.values.get(3)
    disc_fcf_value = _discount_to_fair_value(fin, growth_rate=fcf_growth)
    if disc_fcf_value is not None and math.isnan(disc_fcf_value):
        disc_fcf_value = None
    disc_fcf_metric = None
    if disc_fcf_value is not None:
        df_thr = resolved["discount_fcf"]
        df_passes = df_thr["min"] <= disc_fcf_value <= df_thr["max"]
        disc_fcf_metric = MetricResult(
            label="Discount to Fair Value (FCF)",
            values={10: disc_fcf_value, 5: disc_fcf_value, 3: disc_fcf_value, 1: disc_fcf_value},
            passes=df_passes,
            unit="ratio",
        )

    # Discount to Fair Value (NP) — uses EPS CAGR as the growth input
    np_growth = eps_m.values.get(10) or eps_m.values.get(5) or eps_m.values.get(3)
    disc_np_value = _discount_to_fair_value(fin, growth_rate=np_growth)
    if disc_np_value is not None and math.isnan(disc_np_value):
        disc_np_value = None
    disc_np_metric = None
    if disc_np_value is not None:
        dn_thr = resolved["discount_np"]
        dn_passes = dn_thr["min"] <= disc_np_value <= dn_thr["max"]
        disc_np_metric = MetricResult(
            label="Discount to Fair Value (NP)",
            values={10: disc_np_value, 5: disc_np_value, 3: disc_np_value, 1: disc_np_value},
            passes=dn_passes,
            unit="ratio",
        )

    return Big5Result(
        roic=roic_m,
        sales=sales_m,
        eps=eps_m,
        equity=equity_m,
        fcf=fcf_m,
        roic_by_year=roic_series,
        thresholds=resolved,
        cash_conversion=cash_conv_metric,
        discount_fcf=disc_fcf_metric,
        discount_np=disc_np_metric,
    )