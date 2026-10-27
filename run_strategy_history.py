# run_strategy_history.py
"""
Replay the TQQQ/SQQQ strategy day by day, respecting the time-delay rule:
- Decision at close of day t
- Execution at open of day t+1
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tqqq_sqqq_strategies import (
    STRATEGIES,
    _compute_rsi,
    _compute_atr,
)


def _build_data_for_date(qqq_slice: pd.DataFrame) -> dict | None:
    """Build the data dict the strategy functions expect, using only bars
    up to and including qqq_slice's last row."""
    if len(qqq_slice) < 200:
        return None

    qqq = qqq_slice.dropna(subset=["Close"]).copy()
    close = qqq["Close"]

    ind = {
        "price": float(close.iloc[-1]),
        "sma_20": float(close.rolling(20).mean().iloc[-1]),
        "sma_50": float(close.rolling(50).mean().iloc[-1]),
        "sma_200": float(close.rolling(200).mean().iloc[-1]),
        "ema_12": float(close.ewm(span=12, adjust=False).mean().iloc[-1]),
        "ema_26": float(close.ewm(span=26, adjust=False).mean().iloc[-1]),
        "bb_mid": float(close.rolling(20).mean().iloc[-1]),
        "bb_std": float(close.rolling(20).std().iloc[-1]),
        "rsi_14": _compute_rsi(close, 14),
        "atr_14": _compute_atr(qqq, 14),
        "dist_from_sma50": float(
            (close.iloc[-1] - close.rolling(50).mean().iloc[-1])
            / close.rolling(50).mean().iloc[-1]
        ),
        "dist_from_sma200": float(
            (close.iloc[-1] - close.rolling(200).mean().iloc[-1])
            / close.rolling(200).mean().iloc[-1]
        ),
    }
    ind["bb_upper"] = ind["bb_mid"] + 2 * ind["bb_std"]
    ind["bb_lower"] = ind["bb_mid"] - 2 * ind["bb_std"]

    bb_range = ind["bb_upper"] - ind["bb_lower"]
    denom_width = ind["bb_mid"] if pd.notna(ind["bb_mid"]) and ind["bb_mid"] != 0 else 1.0
    denom_pos = bb_range if pd.notna(bb_range) and bb_range != 0 else 1.0
    ind["bb_width"] = (bb_range if pd.notna(bb_range) else 0.0) / denom_width
    ind["bb_position"] = (
        (ind["price"] - ind["bb_lower"]) / denom_pos * 100
        if pd.notna(ind["price"]) else 50.0
    )

    if ind["price"] > ind["sma_50"] > ind["sma_200"]:
        ind["regime"] = "BULL"
    elif ind["price"] < ind["sma_50"] < ind["sma_200"]:
        ind["regime"] = "BEAR"
    else:
        ind["regime"] = "NEUTRAL"

    return {"QQQ": qqq, "_indicators": ind}


def run_strategy_vote(data: dict) -> dict:
    """Run all 7 strategies and aggregate into a {tqqq, sqqq, cash} target.

    One-sided structure (matches Mallik's documented behavior):
    the residual of the signal goes to CASH, never to the opposite ETF.
    His own example: '30% TQQQ, 0% SQQQ, 70% cash' — never both legs at once.
    """
    ind = data["_indicators"]
    votes = []
    for fn in STRATEGIES:
        try:
            r = fn(data)
            votes.append(float(r["target_tqqq"]))
        except Exception:
            continue

    if not votes:
        return {"tqqq": 0.0, "sqqq": 0.0, "cash": 1.0}

    base_tqqq = sum(votes) / len(votes)
    deviation = base_tqqq - 0.50

    if ind["regime"] in ("BULL", "BEAR") and ind["bb_width"] > 0.10:
        position_multiplier = 1.2
    else:
        position_multiplier = 0.8

    scaled = 0.50 + deviation * position_multiplier
    scaled = max(0.0, min(1.0, scaled))

    # One-sided: residual goes to CASH, not the opposite ETF.
    if scaled >= 0.5:
        tqqq = scaled
        sqqq = 0.0
        cash = 1.0 - scaled
    else:
        tqqq = 0.0
        sqqq = 1.0 - scaled
        cash = 1.0 - sqqq

    return {"tqqq": tqqq, "sqqq": sqqq, "cash": cash}


def build_daily_targets(qqq: pd.DataFrame, start_date: str | None = None) -> pd.DataFrame:
    """Walk forward through qqq, emitting {tqqq, sqqq, cash} for each day
    from `start_date` (or from the first day with 200 bars of history).

    Slow: ~1ms per day × ~10,000 days for 40 years = ~10 seconds.
    """
    if start_date is None:
        dates = qqq.index[200:]
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
        target = run_strategy_vote(data)
        rows.append({"date": d, **target})

    if not rows:
        return pd.DataFrame()

    df_diag = pd.DataFrame(rows).set_index("date")

    # --- DIAGNOSTIC DUMP (temporary; remove after analysis) ---
    # print(f"\n[DIAG] Target distribution over {len(df_diag)} days:")
    # print(f"  TQQQ target:  min={df_diag['tqqq'].min():.3f}  "
    #       f"p25={df_diag['tqqq'].quantile(0.25):.3f}  "
    #       f"median={df_diag['tqqq'].median():.3f}  "
    #       f"p75={df_diag['tqqq'].quantile(0.75):.3f}  "
    #       f"max={df_diag['tqqq'].max():.3f}")
    # print(f"  SQQQ target:  min={df_diag['sqqq'].min():.3f}  "
    #       f"p25={df_diag['sqqq'].quantile(0.25):.3f}  "
    #       f"median={df_diag['sqqq'].median():.3f}  "
    #       f"p75={df_diag['sqqq'].quantile(0.75):.3f}  "
    #       f"max={df_diag['sqqq'].max():.3f}")
    # print(f"  Cash target:  min={df_diag['cash'].min():.3f}  "
    #       f"p25={df_diag['cash'].quantile(0.25):.3f}  "
    #       f"median={df_diag['cash'].median():.3f}  "
    #       f"p75={df_diag['cash'].quantile(0.75):.3f}  "
    #       f"max={df_diag['cash'].max():.3f}")
    # print(f"  Days with TQQQ > 0.50:  {(df_diag['tqqq'] > 0.50).sum()}")
    # print(f"  Days with TQQQ > 0.70:  {(df_diag['tqqq'] > 0.70).sum()}")
    # print(f"  Days with SQQQ > 0.30:  {(df_diag['sqqq'] > 0.30).sum()}")
    # print(f"  Days with cash  < 0.30: {(df_diag['cash'] < 0.30).sum()}")
    # print(f"  Days with cash  > 0.70: {(df_diag['cash'] > 0.70).sum()}")
    # --- END DIAGNOSTIC ---

    return df_diag