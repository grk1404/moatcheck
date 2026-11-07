from __future__ import annotations

import pandas as pd


def _year_labelled(s: pd.Series) -> pd.Series | None:
    """Return `s` indexed by int fiscal year (sorted, de-duplicated), or None if the
    index is not a plain year index (then callers fall back to positional logic)."""
    if s.empty:
        return s
    try:
        years = [int(x) for x in s.index]
        if not all(1900 <= y <= 2200 and float(x) == y for x, y in zip(s.index, years)):
            return None
    except (TypeError, ValueError):
        return None
    out = pd.Series(s.to_numpy(dtype=float), index=pd.Index(years))
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def cagr(series: pd.Series, years: int, min_base_ratio: float = 0.15) -> float | None:
    """Compounded annual growth rate between the latest fiscal year and the year
    exactly `years` earlier.

    The start year is looked up by its *label*, not by position, so a series with a
    missing middle year (e.g. NVDA free cash flow 2012 -> 2022) can never silently
    stretch a "5-year" window across 14 calendar years.

    Returns None when the window can't be computed honestly:
      - the start year (latest - years) is not in the series
      - start value <= 0 (CAGR is undefined; a sign flip is not "growth")
      - end value <= 0 (company went negative; treat as no meaningful CAGR)
      - the start value is a tiny fraction of the series' typical size
    """
    if years < 1:
        return None
    s = series.dropna()
    if s.empty:
        return None

    labelled = _year_labelled(s)
    if labelled is not None:
        end_year = int(labelled.index[-1])
        start_year = end_year - years
        if start_year not in labelled.index:
            return None
        start = float(labelled.loc[start_year])
        end = float(labelled.iloc[-1])
        typical = float(labelled.abs().median())
    else:
        # Non-year index (rare): positional fallback, as before.
        if len(s) < years + 1:
            return None
        start = float(s.iloc[-(years + 1)])
        end = float(s.iloc[-1])
        typical = float(s.abs().median())

    if start <= 0 or end <= 0:
        return None
    # Guard against a distorted base: if start is a small fraction of the series'
    # typical magnitude, this window's CAGR isn't trustworthy.
    if typical > 0 and start < min_base_ratio * typical:
        return None
    return (end / start) ** (1 / years) - 1


def window_average(series: pd.Series, years: int) -> float | None:
    """Simple mean over the last `years` fiscal years. Used for ROIC.

    Returns None unless every one of those `years` fiscal years has a value.
    """
    s = series.dropna()
    if s.empty or years < 1:
        return None
    labelled = _year_labelled(s)
    if labelled is not None:
        end_year = int(labelled.index[-1])
        window = labelled[labelled.index > end_year - years]
        if len(window) < years:
            return None
        return float(window.mean())
    if len(s) < years:
        return None
    return float(s.iloc[-years:].mean())
