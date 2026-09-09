from __future__ import annotations

import pandas as pd


def cagr(series: pd.Series, years: int, min_base_ratio: float = 0.15) -> float | None:
    """Compounded annual growth rate over the last `years` full periods.

    Returns None when the window can't be computed honestly:
      - fewer than `years + 1` observations
      - start value <= 0 (CAGR is undefined; a sign flip is not "growth")
      - end value <= 0 (company went negative — treat as no meaningful CAGR)
    """
    s = series.dropna()
    if len(s) < years + 1:
        return None
    #print(f"DEBUG cagr(years={years}) full series after dropna:\n{s}")
    #print(f"DEBUG cagr(years={years}) index dtype={s.index.dtype}, is_monotonic_increasing={s.index.is_monotonic_increasing}, has_duplicates={s.index.duplicated().any()}")
    start = float(s.iloc[-(years + 1)])
    end = float(s.iloc[-1])
    #print(f"DEBUG cagr years={years}: start_idx={s.index[-(years+1)]} start={start}  end_idx={s.index[-1]} end={end}")
    if start <= 0 or end <= 0:
        return None
    # Guard against a distorted base: compare start to the series' median
    # magnitude. If start is a small fraction of the "typical" value, this
    # window's CAGR isn't trustworthy.
    typical = float(s.abs().median())
    if typical > 0 and start < min_base_ratio * typical:
        return None    
    result = (end / start) ** (1 / years) - 1
    print(f"DEBUG cagr(years={years}) result={result}")
    return result


def window_average(series: pd.Series, years: int) -> float | None:
    """Simple mean of the last `years` observations. Used for ROIC.

    Returns None if fewer than `years` observations are available.
    """
    s = series.dropna()
    if len(s) < years:
        return None
    return float(s.iloc[-years:].mean())
