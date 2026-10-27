# synthetic.py
"""
Synthetic TQQQ/SQQQ reconstruction from NDX or QQQ daily returns.

Uses the standard daily-reset formula:
    levered_return_t = leverage * underlying_return_t - daily_cost

For long 3× (TQQQ), leverage = +3.
For inverse 3× (SQQQ), leverage = -3.

The daily cost term is the expense ratio / 252 plus the financing spread
that real 3× ETFs pay. TQQQ's expense ratio is ~0.95%; SQQQ's is ~0.95%.
Financing cost (the swap rate TQQQ pays to hold 3× exposure) is not in
the ER — it's a separate ~overnight-rate cost. Historical overnight
rates ranged 0.25% (2010-2015) to 5.5% (2023-2024). We default to 2%
average. The caller can override.
"""
from __future__ import annotations

import pandas as pd


def synthetic_levered_series(
    underlying_returns: pd.Series,
    leverage: float,
    expense_ratio: float = 0.0095,
    financing_spread: float = 0.02,
    initial_nav: float = 100.0,
) -> pd.Series:
    """Reconstruct a daily-reset levered ETF price series.

    underlying_returns: daily simple returns of the underlying (e.g. QQQ)
    leverage: +3 for TQQQ-style, -3 for SQQQ-style
    expense_ratio: annual ER expressed as a fraction (0.0095 = 0.95%)
    financing_spread: annual financing cost as a fraction (0.02 = 2%)
    """
    # Combine ER and financing into one daily drag.
    annual_cost = expense_ratio + financing_spread
    daily_cost = annual_cost / 252.0

    levered_returns = leverage * underlying_returns - daily_cost
    levered_nav = initial_nav * (1.0 + levered_returns).cumprod()
    return levered_nav.rename(f"synth_{int(leverage)}x")