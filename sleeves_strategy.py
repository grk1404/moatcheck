"""
Sleeve strategy: trend-regime sleeve on ^NDX -> TQQQ / SQQQ / cash target.

Sleeve weight is +1 above both 50d and 250d MA, -1 below both, +0.5 between.
The weight is scaled by a volatility overlay and moved toward its target by at
most MAX_STEP per day, so the live number needs the recent path, not one day.
Backtest (^NDX synthetic 3x, 1986-2026, close fill, long-or-cash): ~+20% CAGR,
-56% max drawdown at TARGET_VOL 0.45. Not a prediction; see sleeves_backtest.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data_provider import get_ticker

W_TREND = 0.80        # capital share of the regime sleeve (max TQQQ before vol scaling)
TARGET_VOL = 0.45     # annualized vol target for the 3x position
MAX_STEP = 0.10       # max change in net position per day
HISTORY_DAYS = 1500   # warmup for the 250d MA, vol and the smoothing path
SYMBOL = "^NDX"


def _target_series(close: pd.Series, band: float, use_vol: bool) -> pd.Series:
    s50 = close.rolling(50).mean()
    s250 = close.rolling(250).mean()
    bull = (close > s50 * (1 + band)) & (s50 > s250)
    bear = (close < s50 * (1 - band)) & (s50 < s250)

    regime = pd.Series(0.5, index=close.index)
    regime[bull] = 1.0
    regime[bear] = -1.0
    target = (W_TREND * regime).where(s250.notna())

    if use_vol:
        rv = close.pct_change().rolling(20).std() * np.sqrt(252)
        scale = (TARGET_VOL / (rv * 3.0)).clip(upper=1.0).fillna(1.0)
        target = target * scale
    return target


def _smooth(target: pd.Series) -> pd.Series:
    t = target.fillna(0.0).to_numpy()
    out = np.zeros(len(t))
    for i in range(1, len(t)):
        out[i] = out[i - 1] + np.clip(t[i] - out[i - 1], -MAX_STEP, MAX_STEP)
    return pd.Series(out, index=target.index)


def build_sleeve_targets(close: pd.Series, short_scale: float = 0.0,
                         band: float = 0.0, use_vol: bool = True) -> pd.DataFrame:
    """Daily tqqq/sqqq/cash targets for a whole price history (for backtests).

    Same rules as get_sleeve_target; pass the full history so the smoothing
    state is warm by the time the evaluation window starts.
    """
    target = _target_series(close, band, use_vol)
    pos = _smooth(target)
    net = pos.where(pos >= 0, pos * short_scale).clip(-1.0, 1.0)
    tq, sq = net.clip(lower=0.0), (-net).clip(lower=0.0)
    out = pd.DataFrame({"tqqq": tq, "sqqq": sq, "cash": 1.0 - tq - sq})
    return out[target.notna()]


def get_sleeve_target(short_scale: float = 0.0, band: float = 0.0, use_vol: bool = True) -> dict:
    """Return today's TQQQ / SQQQ / cash target as percentages of the sleeve.

    short_scale: scales negative positions into SQQQ. 0.0 = long-or-cash, the
    best-tested setting; 0.5 was worse in backtests.
    """
    cash_only = {"tqqq_pct": 0.0, "sqqq_pct": 0.0, "cash_pct": 100.0,
                 "regime": "UNKNOWN", "as_of": None, "raw_target": 0.0}
    try:
        start = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
        df = get_ticker(SYMBOL).history(start=start)
        close = df["Close"].dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
    except Exception as e:
        return {**cash_only, "error": f"{SYMBOL} fetch failed: {e}"}

    if len(close) < 300:
        return {**cash_only, "error": f"Only {len(close)} rows of {SYMBOL}; need 300+"}

    target = _target_series(close, band, use_vol)
    pos = _smooth(target)
    raw = float(target.iloc[-1]) if pd.notna(target.iloc[-1]) else 0.0
    net = float(pos.iloc[-1])
    if net < 0:
        net *= short_scale
    net = max(-1.0, min(1.0, net))

    s50, s250, px = close.rolling(50).mean().iloc[-1], close.rolling(250).mean().iloc[-1], close.iloc[-1]
    if px > s50 > s250:
        regime = "BULL"
    elif px < s50 < s250:
        regime = "BEAR"
    else:
        regime = "NEUTRAL"

    tq, sq = max(net, 0.0), max(-net, 0.0)
    return {
        "tqqq_pct": tq * 100,
        "sqqq_pct": sq * 100,
        "cash_pct": (1.0 - tq - sq) * 100,
        "regime": regime,
        "as_of": close.index[-1].date(),
        "raw_target": raw,
    }
