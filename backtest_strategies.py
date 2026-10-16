"""
Backtest strategies for MoatCheck.

Each strategy is a function that takes QQQ (and optionally TQQQ) price history
and returns a pd.Series of 0/1 signals indexed the same as the input.

The strategy code must NOT look ahead. For each day i, the signal is computed
using only data up to and including day i's close. The backtest engine then
applies that signal on day i+1.

Currently available:
  - signal_buy_hold   : always long (baseline — should match the benchmark)
  - signal_7step      : your live TQQQ/SQQQ 7-step framework, binary long/cash
  - signal_1pct_week  : TASC "One Percent A Week" weekly swing on TQQQ
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# STRATEGY 1 — Buy and hold (baseline)
# ============================================================

def signal_buy_hold(qqq_close: pd.Series, tqqq_close: pd.Series, **params) -> pd.Series:
    """Always long. Used to verify the engine matches the benchmark."""
    return pd.Series(1, index=qqq_close.index, dtype=int)


# ============================================================
# STRATEGY 2 — 7-step TQQQ framework (live strategy)
# ============================================================

def _compute_indicators_at(qqq_window: pd.DataFrame) -> dict:
    """Compute the indicators the 7-step strategies need, on a slice of QQQ.

    qqq_window must have columns Open, High, Low, Close, Volume.
    Returns the same shape as strategies.py's _indicators dict, plus
    None placeholders when not enough history is available.
    """
    close = qqq_window["Close"]

    def _last(series, default=None):
        try:
            v = series.iloc[-1]
            return float(v) if pd.notna(v) else default
        except Exception:
            return default

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    # RSI (Wilder)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rsi_series = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

    price = float(close.iloc[-1])
    sma_50_v = _last(sma_50)
    sma_200_v = _last(sma_200)
    bb_mid_v = _last(bb_mid)
    bb_std_v = _last(bb_std)

    bb_upper = bb_mid_v + 2 * bb_std_v if (bb_mid_v is not None and bb_std_v is not None) else None
    bb_lower = bb_mid_v - 2 * bb_std_v if (bb_mid_v is not None and bb_std_v is not None) else None

    bb_range = (bb_upper - bb_lower) if (bb_upper is not None and bb_lower is not None) else None
    denom_width = bb_mid_v if (bb_mid_v and bb_mid_v != 0) else 1.0
    denom_pos = bb_range if (bb_range and bb_range != 0) else 1.0

    bb_width = (bb_range / denom_width) if bb_range is not None else 0.0
    bb_position = ((price - bb_lower) / denom_pos * 100) if (bb_lower is not None) else 50.0

    if sma_50_v is not None and sma_200_v is not None:
        if price > sma_50_v > sma_200_v:
            regime = "BULL"
        elif price < sma_50_v < sma_200_v:
            regime = "BEAR"
        else:
            regime = "NEUTRAL"
    else:
        regime = "NEUTRAL"

    dist_sma50 = ((price - sma_50_v) / sma_50_v) if sma_50_v else 0.0

    return {"price": price, "sma_20": _last(sma_20), "sma_50": sma_50_v, "sma_200": sma_200_v, "bb_mid": bb_mid_v, "bb_std": bb_std_v, "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_width": bb_width, "bb_position": bb_position, "rsi_14": _last(rsi_series, 50.0), "regime": regime, "dist_from_sma50": dist_sma50}


def _vote_7step(ind: dict) -> float:
    """Run the 7 sub-strategy votes on a single day's indicators.

    Mirrors strategies.py's strategy_momentum_long, strategy_mean_reversion,
    strategy_bb_reversion, strategy_bb_width, strategy_momentum_short,
    strategy_short_reversion, strategy_bear_bounce_fade.

    Returns the average target_tqqq in [0, 1].
    """
    price = ind["price"]
    sma_50 = ind["sma_50"]
    regime = ind["regime"]
    bb_pos = ind["bb_position"]
    bb_width = ind["bb_width"]
    dist = ind["dist_from_sma50"]

    votes = []

    # S1 — Momentum Long
    if regime == "BULL" and (sma_50 is None or price > sma_50):
        votes.append(1.0)
    elif regime == "BEAR":
        votes.append(0.0)
    else:
        votes.append(0.3)

    # S2 — Mean Reversion
    if dist < -0.08 and regime != "BEAR":
        votes.append(0.8)
    elif dist > 0.10:
        votes.append(0.2)
    else:
        votes.append(0.5)

    # S3 — Bollinger Band Reversion
    if bb_pos < 15:
        votes.append(0.9)
    elif bb_pos > 85:
        votes.append(0.1)
    else:
        votes.append(0.5)

    # S4 — BB Width Momentum Confirmation
    if regime == "BULL" and bb_width > 0.08:
        votes.append(0.9)
    elif regime == "BULL":
        votes.append(0.5)
    else:
        votes.append(0.2)

    # S5 — Momentum Short
    if regime == "BEAR":
        votes.append(0.0)
    else:
        votes.append(0.7)

    # S6 — Short Overbought Fade
    if dist > 0.12 and regime != "BULL":
        votes.append(0.1)
    else:
        votes.append(0.6)

    # S7 — Bear Bounce Fade
    if regime == "BEAR" and dist > -0.02:
        votes.append(0.0)
    elif regime == "BEAR" and dist < -0.10:
        votes.append(0.3)
    else:
        votes.append(0.5)

    return float(sum(votes) / len(votes))


def signal_7step(qqq_close: pd.Series, tqqq_close: pd.Series, min_history: int = 200, threshold: float = 0.60, **params) -> pd.Series:
    """Run the 7-step framework day-by-day, return 0/1 signals.

    Parameters:
        qqq_close: full QQQ close series (used to build the OHLC window).
        tqqq_close: full TQQQ close series (kept for symmetry; unused here).
        min_history: minimum QQQ bars required before producing a signal.
        threshold: average vote >= threshold -> long. Below -> cash.

    Returns: pd.Series of 0/1, indexed like qqq_close.
    """
    signals = pd.Series(0, index=qqq_close.index, dtype=int)
    n = len(qqq_close)

    # We need Open/High/Low to compute ATR and other things in the real
    # strategy, but for these 7 votes, Close is enough. Use Close as a stand-in
    # for the OHLC window.
    qqq_df = pd.DataFrame({"Close": qqq_close})

    for i in range(min_history, n):
        window = qqq_df.iloc[: i + 1]
        ind = _compute_indicators_at(window)
        avg_vote = _vote_7step(ind)
        signals.iloc[i] = 1 if avg_vote >= threshold else 0

    return signals


# ============================================================
# STRATEGY 3 — One Percent A Week (TASC weekly swing)
# ============================================================

def signal_1pct_week(qqq_close: pd.Series, tqqq_close: pd.Series, dip_pct: float = 0.01, profit_pct: float = 0.01, breakeven_pct: float = 0.005, **params) -> pd.Series:
    """The TASC 'One Percent A Week' strategy, daily bar equivalent.

    Rules (simplified for daily bars):
      - On each Monday (or first trading day of the week), note the open price.
      - During the week, if price drops dip_pct% below Monday's open, enter long.
      - Exit at +profit_pct% from entry, or at breakeven if a loss of
        breakeven_pct% is hit first, or at Friday close.
      - Long-only, no shorts. Return 0/1 daily.

    Since the daily engine can only trade at next-day open, this strategy
    computes the entry trigger intraday and fires the signal the next day.
    The engine then buys at the following open.
    """
    signals = pd.Series(0, index=qqq_close.index, dtype=int)

    # Group by ISO week
    idx = qqq_close.index
    week_id = idx.isocalendar().week.values
    year_id = idx.isocalendar().year.values

    # Track state per week
    position = 0
    entry_price = None
    week_open_price = None
    current_week = None

    for i, date in enumerate(idx):
        this_week = (year_id[i], week_id[i])

        # New week — reset
        if this_week != current_week:
            current_week = this_week
            week_open_price = float(qqq_close.iloc[i])
            position = 0
            entry_price = None

        price = float(qqq_close.iloc[i])

        if position == 0:
            # Look for entry: price dropped dip_pct below week's open
            if week_open_price is not None and price <= week_open_price * (1 - dip_pct):
                # Fire signal for tomorrow's open (engine will apply it next day)
                position = 1
                entry_price = price
                signals.iloc[i] = 1
                continue

        elif position == 1:
            # In position — check exit conditions on this day's close
            # (signal fires for tomorrow, engine sells at tomorrow's open)
            if entry_price is not None:
                gain = price / entry_price - 1.0
                if gain >= profit_pct:
                    position = 0
                    entry_price = None
                    signals.iloc[i] = 0
                    continue
                if gain <= -breakeven_pct:
                    position = 0
                    entry_price = None
                    signals.iloc[i] = 0
                    continue

            # Friday close — force exit
            if date.weekday() == 4:  # Friday
                position = 0
                entry_price = None
                signals.iloc[i] = 0
                continue

            # Stay in position
            signals.iloc[i] = 1

    return signals


def signal_15pct_swing(
    qqq_close: pd.Series,
    tqqq_close: pd.Series,
    threshold: float = 0.15,
    lookback: int | None = None,
    **params,
) -> pd.Series:
    """Buy every 15% drop. Sell only when the position is +15% from entry.

    If `lookback` is None (default): the BUY reference is the running high
    since the last exit — an unbounded swing high. This is the original
    Design B behavior.

    If `lookback` is an integer N: the BUY reference is the rolling max of
    the last N closes. This is Design A. Use for parameter sweeps.

    In both modes:
      - SELL fires only when price is +threshold above the actual entry.
      - No pyramiding: while long, further drops do not trigger new buys.
    """
    signals = pd.Series(0, index=tqqq_close.index, dtype=int)
    prices = tqqq_close.dropna()
    if len(prices) < 2:
        return signals

    # Precompute rolling max if we're in Design A mode
    if lookback is not None:
        rolling_max = prices.rolling(lookback).max()
    else:
        rolling_max = None

    in_position = False
    entry_price = None
    running_high = float(prices.iloc[0])

    for date, px in prices.iloc[1:].items():
        px = float(px)

        if not in_position:
            # Determine the reference high
            if rolling_max is not None:
                ref_high = rolling_max.loc[date]
                if pd.isna(ref_high):
                    signals.iloc[signals.index.get_loc(date)] = 0
                    continue
                ref_high = float(ref_high)
            else:
                # Running high: only rises while flat
                if px > running_high:
                    running_high = px
                ref_high = running_high

            # Enter when price drops `threshold` below the reference high
            if px <= ref_high * (1 - threshold):
                in_position = True
                entry_price = px
        else:
            # In position — exit only at +threshold from entry
            if entry_price is not None and px >= entry_price * (1 + threshold):
                in_position = False
                entry_price = None
                # Reset running high to exit price (Design B only)
                if rolling_max is None:
                    running_high = px

        signals.iloc[signals.index.get_loc(date)] = 1 if in_position else 0

    return signals


# ============================================================
# STRATEGY REGISTRY
# ============================================================
STRATEGIES = {
    "Buy and Hold (baseline)": signal_buy_hold,
    "7-Step TQQQ (live)": signal_7step,
    "One Percent A Week (TASC^)": signal_1pct_week,
    "Swing - since last exit": signal_15pct_swing,
    "Swing - 20-day window": lambda q, t, **p: signal_15pct_swing(q, t, lookback=20, **p),
    "Swing - 60-day window": lambda q, t, **p: signal_15pct_swing(q, t, lookback=60, **p),
}