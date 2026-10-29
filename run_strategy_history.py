"""
Replay the TQQQ/SQQQ strategy day by day, respecting the time-delay rule.
Updated to use the exact same logic as the live app (no duplicate code).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Import the live strategy coordinator and indicator helpers
from tqqq_sqqq_strategies import run_all_strategies, _compute_rsi, _compute_atr

def _build_data_for_date(qqq_slice: pd.DataFrame) -> dict | None:
    """Build the data dict using the new 250 SMA regime logic."""
    if len(qqq_slice) < 250:  # Need 250 rows for the new SMA
        return None

    qqq = qqq_slice.dropna(subset=["Close"]).copy()
    close = qqq["Close"]
    
    price = float(close.iloc[-1])
    sma_50 = float(close.rolling(50).mean().iloc[-1])
    sma_200 = float(close.rolling(200).mean().iloc[-1])
    sma_250 = float(close.rolling(250).mean().iloc[-1])
    bb_mid = float(close.rolling(20).mean().iloc[-1])
    bb_std = float(close.rolling(20).std().iloc[-1])

    ind = {
        "price": price,
        "sma_20": bb_mid,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "sma_250": sma_250,
        "ema_12": float(close.ewm(span=12, adjust=False).mean().iloc[-1]),
        "ema_26": float(close.ewm(span=26, adjust=False).mean().iloc[-1]),
        "bb_mid": bb_mid,
        "bb_std": bb_std,
        "rsi_14": _compute_rsi(close, 14),
        "atr_14": _compute_atr(qqq, 14),
    }
    
    # New Regime Logic
    ind["sma_regime"] = ind["sma_250"]
    ind["dist_from_sma50"] = (price - sma_50) / sma_50 if sma_50 else 0.0
    ind["dist_from_sma200"] = (price - sma_200) / sma_200 if sma_200 else 0.0
    ind["dist_from_sma250"] = (price - sma_250) / sma_250 if sma_250 else 0.0

    ind["bb_upper"] = bb_mid + 2 * bb_std
    ind["bb_lower"] = bb_mid - 2 * bb_std
    bb_range = ind["bb_upper"] - ind["bb_lower"]
    denom_width = bb_mid if pd.notna(bb_mid) and bb_mid != 0 else 1.0
    denom_pos = bb_range if pd.notna(bb_range) and bb_range != 0 else 1.0

    ind["bb_width"] = (bb_range if pd.notna(bb_range) else 0.0) / denom_width
    ind["bb_position"] = ((price - ind["bb_lower"]) / denom_pos * 100) if pd.notna(price) else 50.0

    if price > sma_50 > ind["sma_regime"]:
        ind["regime"] = "BULL"
    elif price < sma_50 < ind["sma_regime"]:
        ind["regime"] = "BEAR"
    else:
        ind["regime"] = "NEUTRAL"

    return {"QQQ": qqq, "_indicators": ind}

def build_daily_targets(qqq: pd.DataFrame, start_date: str | None = None) -> pd.DataFrame:
    """Walk forward through qqq, emitting {tqqq, sqqq, cash} for each day."""
    if start_date is None:
        dates = qqq.index[200:]  # Start after 250-day warmup
    else:
        dates = qqq.index[qqq.index >= pd.Timestamp(start_date)]
        if len(dates) == 0:
            return pd.DataFrame()

    rows = []
    for d in dates:
        qqq_slice = qqq.loc[:d]
        data = _build_data_for_date(qqq_slice)
        if data is None:
            continue
            
        # CALL THE LIVE STRATEGY DIRECTLY
        result = run_all_strategies(data=data)
        
        if "error" in result:
            continue
            
        rows.append({
            "date": d,
            "tqqq": result["target_tqqq_pct"] / 100.0,
            "sqqq": result["target_sqqq_pct"] / 100.0,
            "cash": result["target_cash_pct"] / 100.0,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")