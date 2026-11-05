"""
Backtest strategies for the MoatCheck Back Test page.

Every strategy has the signature

    fn(prices, trade_start=None, swing_pct=0.15, leverage=1.0, **_) -> StrategySignal

`prices` comes from backtest.load_backtest_data (columns: open, close,
signal_close, qqq_open, qqq_close). `open`/`close` are the TRADED ticker.
`signal_close` is the series the trend/regime logic reads: the ticker's own
price, or QQQ when the user picks it (default for TQQQ, matching the live app).

The returned weight is the fraction of the portfolio to hold from the NEXT
bar's open (the engine applies the shift), so nothing can look ahead.

`trade_start`: history before it only warms up indicators (SMAs, rolling
highs). The weight is 0 before it, and all trade state (in-position flag,
running high, entry price, Sleeves position ramp) starts clean at it, so
results do not depend on how much warmup history the loader fetched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StrategySignal:
    weight: pd.Series                         # target weight 0..1, decided at each bar's close
    exit_at_close: pd.Series | None = None    # True = sell at this bar's close, not the next open


def _start_pos(index: pd.Index, trade_start) -> int:
    """Integer position of the first bar on/after trade_start (0 if None)."""
    if trade_start is None:
        return 0
    return int(index.searchsorted(pd.Timestamp(trade_start), side="left"))


def _zeros(index: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=index)


# ============================================================
# 1. Buy and hold (baseline)
# ============================================================

def signal_buy_hold(prices: pd.DataFrame, trade_start=None, **_) -> StrategySignal:
    w = pd.Series(1.0, index=prices.index)
    w.iloc[: _start_pos(prices.index, trade_start)] = 0.0
    return StrategySignal(w)


# ============================================================
# 2. 7-Step dial (live rules) - the same vote functions the live app uses
# ============================================================

def _dial_indicators(close: pd.Series, regime_ma: int) -> pd.DataFrame:
    """Per-day version of tqqq_sqqq_strategies.load_strategy_data's indicator block."""
    sma_50 = close.rolling(50).mean()
    sma_reg = close.rolling(regime_ma).mean()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper, bb_lower = bb_mid + 2 * bb_std, bb_mid - 2 * bb_std
    bb_range = bb_upper - bb_lower
    denom_w = bb_mid.where(bb_mid.notna() & (bb_mid != 0), 1.0)
    denom_p = bb_range.where(bb_range.notna() & (bb_range != 0), 1.0)
    return pd.DataFrame({
        "price": close, "sma_50": sma_50, "sma_regime": sma_reg,
        "dist_from_sma50": (close - sma_50) / sma_50,
        "bb_width": bb_range / denom_w,
        "bb_position": (close - bb_lower) / denom_p * 100,
    })


def signal_dial(prices: pd.DataFrame, trade_start=None, **_) -> StrategySignal:
    """Live 7-vote dial -> TQQQ weight, long-or-cash (the short leg is not traded).

    Calls the live strategy_* functions and dial_to_allocation, so the Back Test
    cannot drift from the app. Settings come from data/dial_settings.json.
    """
    import tqqq_sqqq_strategies as live

    ind = _dial_indicators(prices["signal_close"], live.REGIME_LONG_MA)
    start_i = _start_pos(prices.index, trade_start)
    w = np.zeros(len(ind))
    cols = ind.to_dict("list")
    for i in range(start_i, len(ind)):
        if pd.isna(cols["sma_regime"][i]) or pd.isna(cols["bb_position"][i]):
            continue
        px, s50, sreg = cols["price"][i], cols["sma_50"][i], cols["sma_regime"][i]
        regime = "BULL" if px > s50 > sreg else "BEAR" if px < s50 < sreg else "NEUTRAL"
        d = {"price": px, "sma_50": s50, "sma_regime": sreg, "regime": regime,
             "dist_from_sma50": cols["dist_from_sma50"][i],
             "bb_width": cols["bb_width"][i], "bb_position": cols["bb_position"][i]}
        data = {"_indicators": d}
        votes = [float(fn(data)["target_tqqq"]) for fn in live.STRATEGIES]
        deviation = sum(votes) / len(votes) - 0.5
        mult = live.SETTINGS["high_vol_multiplier"] if (regime in ("BULL", "BEAR") and d["bb_width"] > 0.10) else live.SETTINGS["low_vol_multiplier"]
        dial = max(0.0, min(1.0, 0.5 + deviation * mult))
        w[i] = live.dial_to_allocation(dial)["tqqq"]
    return StrategySignal(pd.Series(w, index=prices.index))


# ============================================================
# 3. Sleeves (the strategy traded live)
# ============================================================

def signal_sleeves(prices: pd.DataFrame, trade_start=None, leverage: float = 1.0, **_) -> StrategySignal:
    """Trend-regime sleeve from sleeves_strategy.py, long-or-cash.

    The 50/250-day MAs and 20-day vol are computed on `signal_close` using all
    warmup history. The 10%/day position ramp starts from cash at trade_start,
    like someone starting that day. `leverage` is the traded ticker's leverage
    relative to the signal series (3 for TQQQ driven by QQQ) and scales the vol overlay.
    """
    import sleeves_strategy as sl

    target = sl._target_series(prices["signal_close"], band=0.0, use_vol=True, leverage=leverage)
    start_i = _start_pos(prices.index, trade_start)
    w = pd.Series(0.0, index=prices.index)
    if start_i < len(target):
        pos = sl._smooth(target.iloc[start_i:])
        w.iloc[start_i:] = pos.clip(lower=0.0, upper=1.0).to_numpy()
    return StrategySignal(w)


# ============================================================
# 4. One Percent A Week (TASC) - daily-bar version
# ============================================================

def signal_1pct_week(prices: pd.DataFrame, trade_start=None, dip_pct: float = 0.01,
                     profit_pct: float = 0.01, stop_pct: float = 0.005, **_) -> StrategySignal:
    """Weekly dip-buy on the TRADED ticker.

    - Reference = the week's first OPEN. A close >= dip_pct below it signals a buy, filled at the next open.
    - Exit: close >= entry + profit_pct (sold at the next open), close <= entry - stop_pct (stop, next open),
      or the week's last trading day, sold AT THAT DAY'S CLOSE (never held over the weekend).
    - No entries on a week's last trading day. Long only, one position at a time.
    Daily bars cannot see intraday touches, so triggers are checked on closes.
    """
    idx = prices.index
    o, c = prices["open"].to_numpy(float), prices["close"].to_numpy(float)
    iso = idx.isocalendar()
    wk = (iso.year.astype(int) * 100 + iso.week.astype(int)).to_numpy()
    n = len(idx)
    start_i = _start_pos(idx, trade_start)

    w = np.zeros(n)
    exit_close = np.zeros(n, dtype=bool)
    in_pos = False       # holding, or filled at the next open
    entry_px = None      # actual fill (open of the bar after the signal)
    entry_bar = -1
    week_open = None
    cur_wk = None

    for i in range(start_i, n):
        if wk[i] != cur_wk:
            cur_wk, week_open = wk[i], o[i]
            if in_pos:   # defensive: position never survives the weekly close
                in_pos, entry_px = False, None
        last_of_week = (i + 1 >= n) or (wk[i + 1] != wk[i])

        if in_pos:
            if entry_px is None and i == entry_bar:      # fill happens at this bar's open
                entry_px = o[i]
            if last_of_week:
                exit_close[i], in_pos, entry_px = True, False, None
                continue
            gain = c[i] / entry_px - 1.0
            if gain >= profit_pct or gain <= -stop_pct:
                in_pos, entry_px = False, None
                continue
            w[i] = 1.0
        else:
            if not last_of_week and c[i] <= week_open * (1.0 - dip_pct) and i + 1 < n:
                in_pos, entry_px, entry_bar = True, None, i + 1
                w[i] = 1.0
    return StrategySignal(pd.Series(w, index=idx), pd.Series(exit_close, index=idx))


# ============================================================
# 5. Swing family
# ============================================================

def _swing(prices: pd.DataFrame, trade_start, threshold: float, lookback: int | None, regime_filter: bool) -> StrategySignal:
    """Buy a `threshold` drop from the reference high, sell `threshold` above the fill price.

    Reference high: running high since the last exit (lookback None) or the
    rolling `lookback`-bar high. regime_filter: only buy while signal_close > its 200-day SMA.
    The fill (entry price) is the NEXT bar's open, taken from the traded ticker's own open.
    """
    idx = prices.index
    c, o = prices["close"].to_numpy(float), prices["open"].to_numpy(float)
    n = len(idx)
    if n < 2:
        return StrategySignal(_zeros(idx))
    start_i = min(_start_pos(idx, trade_start), n - 1)

    roll_hi = None
    if lookback is not None:
        roll_hi = prices["close"].rolling(lookback, min_periods=1).max().to_numpy(float)
    if regime_filter:
        sma = prices["signal_close"].rolling(200).mean().to_numpy(float)
        sig_px = prices["signal_close"].to_numpy(float)
        bull = np.where(np.isnan(sma), False, sig_px > sma)
    else:
        bull = np.ones(n, dtype=bool)

    w = np.zeros(n)
    in_pos, entry_px = False, None
    entry_bar = -1
    running_high = c[start_i]

    for i in range(start_i, n):
        if not in_pos:
            running_high = max(running_high, c[i])
            ref = roll_hi[i] if roll_hi is not None else running_high
            if c[i] <= ref * (1.0 - threshold) and bull[i] and i + 1 < n:
                in_pos, entry_px, entry_bar = True, None, i + 1
                w[i] = 1.0
        else:
            if entry_px is None:
                entry_px = o[entry_bar]          # actual fill: the open after the signal
            w[i] = 1.0
            if i + 1 < n and c[i] >= entry_px * (1.0 + threshold):
                in_pos, entry_px = False, None   # sell at the next open
                running_high = c[i]
                w[i] = 0.0
    return StrategySignal(pd.Series(w, index=idx))


def signal_swing_since_exit(prices, trade_start=None, swing_pct=0.15, **_):
    return _swing(prices, trade_start, swing_pct, None, False)


def signal_swing_20d(prices, trade_start=None, swing_pct=0.15, **_):
    return _swing(prices, trade_start, swing_pct, 20, False)


def signal_swing_60d(prices, trade_start=None, swing_pct=0.15, **_):
    return _swing(prices, trade_start, swing_pct, 60, False)


def signal_swing_regime(prices, trade_start=None, swing_pct=0.15, **_):
    return _swing(prices, trade_start, swing_pct, None, True)


# ============================================================
# STRATEGY REGISTRY
# ============================================================
STRATEGIES = {
    "Buy and Hold (baseline)": signal_buy_hold,
    "Sleeves (traded live)": signal_sleeves,
    "7-Step Dial (live rules)": signal_dial,
    "One Percent A Week (TASC^)": signal_1pct_week,
    "Swing - since last exit": signal_swing_since_exit,
    "Swing - 20-day window": signal_swing_20d,
    "Swing - 60-day window": signal_swing_60d,
    "Regime-Aware Swing": signal_swing_regime,
}
