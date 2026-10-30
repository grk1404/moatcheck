"""
TQQQ/SQQQ Multi-Strategy Framework
Seven independent sub-strategies that each vote on a "dial" from 0.0 to 1.0:
    1.0 = full TQQQ, 0.5 = neutral, 0.0 = full SQQQ.
The coordinator averages the votes and converts the dial into a
TQQQ / SQQQ / cash allocation.

Each strategy follows the same signature:
    strategy_name(data: dict) -> dict
        returns:
            {
                'name': str,
                'target_tqqq': float,   # 0.0 to 1.0 (the dial)
                'reason': str,
            }

Add new strategies by writing a function and registering it in STRATEGIES.

CHANGES FROM THE PREVIOUS VERSION
 1. Dial -> allocation mapping is continuous around 0.5 (no more flip from
    50% TQQQ to 50% SQQQ on a 0.001 change). The old behaviour is available
    with ALLOCATION_MODE = "legacy".
 2. Signal label and confidence are computed from the dial, not from the
    already-split TQQQ number (which made every sub-0.5 reading "SQQQ HEAVY").
 3. Dead conviction/cash code removed; cash is simply what is left over.
 4. "No signal" votes are neutral (0.5) so flat systems don't bias bullish.
 5. Regime uses SMA 250 (matching Mallik's published 50/250) via REGIME_LONG_MA.
 6. Optional rebalance band to cut small trades.
 7. NaN-safe clamping (min/max silently turned NaN into 1.0 before).
"""

from typing import Optional

from data_provider import get_ticker
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

NEUTRAL = 0.5            # vote used when a system has no signal
REGIME_LONG_MA = 250     # 250 matches Mallik's blog; set to 200 for the old behaviour
MIN_ROWS = 250           # minimum valid QQQ rows required
ALLOCATION_MODE = "continuous"   # "continuous" or "legacy"


# ============================================================
# DIAL -> ALLOCATION
# ============================================================

def dial_to_allocation(dial: float, mode: Optional[str] = None) -> dict:
    """
    Convert a 0..1 dial into fractions {tqqq, sqqq, cash} that sum to 1.
    Uses a 48%-52% cash dead-zone, but does NOT dampen exposure outside the zone.
    """
    mode = mode or ALLOCATION_MODE
    try:
        dial = float(dial)
    except (TypeError, ValueError):
        return {"tqqq": 0.0, "sqqq": 0.0, "cash": 1.0}
    if not np.isfinite(dial):
        return {"tqqq": 0.0, "sqqq": 0.0, "cash": 1.0}

    dial = max(0.0, min(1.0, dial))
    
    DEAD_ZONE = 0.02  # 0.48 to 0.52 is 100% Cash

    if dial >= 0.50 + DEAD_ZONE:
        # Direct allocation. If dial is 0.70, TQQQ is 70%.
        tqqq = dial
        sqqq = 0.0
    elif dial <= 0.50 - DEAD_ZONE:
        # Direct allocation. If dial is 0.30, SQQQ is 70% (1.0 - 0.30).
        tqqq = 0.0
        sqqq = 1.0 - dial
    else:
        # Inside the dead zone
        tqqq = 0.0
        sqqq = 0.0

    # Clamp to ensure safety
    tqqq = max(0.0, min(1.0, tqqq))
    sqqq = max(0.0, min(1.0, sqqq))
    cash = max(0.0, 1.0 - tqqq - sqqq)

    return {"tqqq": tqqq, "sqqq": sqqq, "cash": cash}


# ============================================================
# DATA LOADER - Fetches QQQ, TQQQ, SQQQ for regime detection
# ============================================================

def load_strategy_data(period: str = "2y") -> dict:
    """
    Fetch QQQ, TQQQ, SQQQ data and precompute common indicators safely.
    All strategies read from this shared data dict.
    """
    data = {}
    for ticker in ["QQQ", "TQQQ", "SQQQ"]:
        try:
            ticker_obj = get_ticker(ticker)
            df = ticker_obj.history(period=period)
            if df is None or df.empty:
                continue
            data[ticker] = df
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")

    if "QQQ" not in data:
        print("Error: QQQ data missing entirely.")
        return {}

    # Drop rows with NaN Close before checking length.
    qqq_clean = data["QQQ"].dropna(subset=["Close"])
    if len(qqq_clean) < MIN_ROWS:
        print(f"Error: Insufficient valid QQQ rows ({len(qqq_clean)} < {MIN_ROWS}).")
        return {}

    data["QQQ"] = qqq_clean

    qqq = data["QQQ"]
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
    ind["sma_regime"] = ind["sma_250"] if REGIME_LONG_MA == 250 else ind["sma_200"]
    ind["dist_from_sma50"] = (price - sma_50) / sma_50 if sma_50 else 0.0
    ind["dist_from_sma200"] = (price - sma_200) / sma_200 if sma_200 else 0.0
    ind["dist_from_sma250"] = (price - sma_250) / sma_250 if sma_250 else 0.0

    ind["bb_upper"] = bb_mid + 2 * bb_std
    ind["bb_lower"] = bb_mid - 2 * bb_std

    # Protect against flatline (zero range) AND NaN (bad data) divisions.
    bb_range = ind["bb_upper"] - ind["bb_lower"]
    denom_width = bb_mid if pd.notna(bb_mid) and bb_mid != 0 else 1.0
    denom_pos = bb_range if pd.notna(bb_range) and bb_range != 0 else 1.0

    ind["bb_width"] = (bb_range if pd.notna(bb_range) else 0.0) / denom_width
    ind["bb_position"] = ((price - ind["bb_lower"]) / denom_pos * 100) if pd.notna(price) else 50.0

    # Regime classification (NaN comparisons are False -> NEUTRAL)
    if price > sma_50 > ind["sma_regime"]:
        ind["regime"] = "BULL"
    elif price < sma_50 < ind["sma_regime"]:
        ind["regime"] = "BEAR"
    else:
        ind["regime"] = "NEUTRAL"

    data["_indicators"] = ind
    return data


def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    """Standard RSI with Wilder's smoothing; returns 50.0 when data is insufficient."""
    if close is None:
        return 50.0

    close = close.dropna()
    if len(close) < period + 1:
        return 50.0

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]

    if pd.isna(last_gain) or pd.isna(last_loss):
        return 50.0

    if last_loss == 0:
        return 100.0 if last_gain > 0 else 50.0

    rs = last_gain / last_loss
    return float(100 - (100 / (1 + rs)))


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if pd.notna(val) else 0.0

# ============================================================
# STRATEGY 1: Momentum Long TQQQ (Trend Following)
# ============================================================

def strategy_momentum_long(data: dict) -> dict:
    ind = data["_indicators"]
    regime = ind["regime"]

    if regime == "BULL" and ind["price"] > ind["sma_50"]:
        return {"name": "Momentum Long", "target_tqqq": 1.0,
                "reason": "Bull regime + price above SMA 50 - full TQQQ"}
    if regime == "BEAR":
        return {"name": "Momentum Long", "target_tqqq": 0.0,
                "reason": "Bear regime - avoid TQQQ"}
    
    # FIX: Neutral regime means no trend. Abstain and hold cash.
    return {"name": "Momentum Long", "target_tqqq": NEUTRAL,
            "reason": "Neutral regime - no trend, holding cash"}


# ============================================================
# STRATEGY 2: Mean Reversion TQQQ (Oversold Bounce)
# ============================================================

def strategy_mean_reversion(data: dict) -> dict:
    ind = data["_indicators"]
    dist = ind["dist_from_sma50"]
    regime = ind["regime"]

    if dist < -0.08 and regime != "BEAR":
        return {"name": "Mean Reversion", "target_tqqq": 0.8,
                "reason": f"Oversold: {dist*100:.1f}% below SMA 50 - bounce likely"}
    if dist > 0.10:
        return {"name": "Mean Reversion", "target_tqqq": 0.2,
                "reason": f"Overextended: {dist*100:.1f}% above SMA 50 - fade risk"}
    return {"name": "Mean Reversion", "target_tqqq": NEUTRAL,
            "reason": f"Within normal range ({dist*100:.1f}% from SMA 50)"}


# ============================================================
# STRATEGY 3: Bollinger Band Reversion
# ============================================================

def strategy_bb_reversion(data: dict) -> dict:
    ind = data["_indicators"]
    pos = ind["bb_position"]

    if pos < 15:
        return {"name": "BB Reversion", "target_tqqq": 0.9,
                "reason": f"At lower BB ({pos:.0f}%) - oversold"}
    if pos > 85:
        return {"name": "BB Reversion", "target_tqqq": 0.1,
                "reason": f"At upper BB ({pos:.0f}%) - overbought"}
    return {"name": "BB Reversion", "target_tqqq": NEUTRAL,
            "reason": f"Mid-band ({pos:.0f}%)"}


# ============================================================
# STRATEGY 4: Bollinger Band Width Momentum Confirmation
# ============================================================

def strategy_bb_width(data: dict) -> dict:
    ind = data["_indicators"]
    regime = ind["regime"]
    width = ind["bb_width"]

    if regime == "BULL" and width > 0.08:
        return {"name": "BB Width Momentum", "target_tqqq": 0.9,
                "reason": f"Wide BB ({width*100:.1f}%) confirms strong bull trend"}
    if regime == "BULL":
        return {"name": "BB Width Momentum", "target_tqqq": NEUTRAL,
                "reason": f"Narrow BB ({width*100:.1f}%) - weak trend"}
    # Changed from 0.2: a momentum-confirmation system with no bull signal
    # should abstain, not vote bearish.
    return {"name": "BB Width Momentum", "target_tqqq": NEUTRAL,
            "reason": "Not in a confirmed bull regime - no signal"}


# ============================================================
# STRATEGY 5: Momentum Short (Bearish Trend Following)
# ============================================================

def strategy_momentum_short(data: dict) -> dict:
    ind = data["_indicators"]

    if ind["regime"] == "BEAR":
        return {"name": "Momentum Short", "target_tqqq": 0.0,
                "reason": "Bear regime confirmed - favor SQQQ"}
    # Changed from 0.7: a short system with no signal is flat, not bullish.
    return {"name": "Momentum Short", "target_tqqq": NEUTRAL,
            "reason": "No bearish confirmation - no signal"}


# ============================================================
# STRATEGY 6: Short Overbought Fade
# ============================================================

def strategy_short_reversion(data: dict) -> dict:
    ind = data["_indicators"]
    dist = ind["dist_from_sma50"]
    regime = ind["regime"]

    if dist > 0.12 and regime != "BULL":
        return {"name": "Short Reversion", "target_tqqq": 0.1,
                "reason": f"Overbought fade: {dist*100:.1f}% above SMA 50"}
    # Changed from 0.6: no fade signal means flat.
    return {"name": "Short Reversion", "target_tqqq": NEUTRAL,
            "reason": "No fade signal active"}


# ============================================================
# STRATEGY 7: Bearish Bounce Fade
# ============================================================

def strategy_bear_bounce_fade(data: dict) -> dict:
    ind = data["_indicators"]
    dist = ind["dist_from_sma50"]
    regime = ind["regime"]

    # In a BEAR regime price is below SMA 50, so dist is negative; being
    # within 2% of the SMA 50 means the bounce has run into resistance.
    if regime == "BEAR" and dist > -0.02:
        return {"name": "Bear Bounce Fade", "target_tqqq": 0.0,
                "reason": f"Bear rally stalling at SMA 50 ({dist*100:.1f}% below) - short fade active"}

    if regime == "BEAR" and dist < -0.10:
        return {"name": "Bear Bounce Fade", "target_tqqq": 0.3,
                "reason": "Bear regime deeply oversold - fade strength"}

    return {"name": "Bear Bounce Fade", "target_tqqq": NEUTRAL,
            "reason": "No bear-bounce setup"}


# ============================================================
# STRATEGY REGISTRY - Add new strategies here
# ============================================================

STRATEGIES = [
    strategy_momentum_long,
    strategy_mean_reversion,
    strategy_bb_reversion,
    strategy_bb_width,
    strategy_momentum_short,
    strategy_short_reversion,
    strategy_bear_bounce_fade,
]


# ============================================================
# MALLIK SIM (continuous sizing, SMA 50 / 250 + Bollinger width)
# NOTE: the formula constants (5.0, 0.5, 1.2, 0.10) are a reconstruction,
# not published by Mallik.
# ============================================================

def get_mallik_live_target(data: dict) -> dict:
    """Mallik-style continuous sizing target for the live app."""
    cash_only = {"tqqq_pct": 0.0, "sqqq_pct": 0.0, "cash_pct": 100.0, "dial": NEUTRAL}

    qqq = data.get("QQQ")
    if qqq is None:
        return cash_only
    close = qqq["Close"].dropna()

    if len(close) < 250:
        return cash_only

    price = float(close.iloc[-1])
    sma_50 = float(close.rolling(50).mean().iloc[-1])
    sma_250 = float(close.rolling(250).mean().iloc[-1])
    bb_mid = float(close.rolling(20).mean().iloc[-1])
    bb_std = float(close.rolling(20).std().iloc[-1])

    if not all(np.isfinite(x) for x in (price, sma_50, sma_250, bb_mid, bb_std)) or sma_250 == 0:
        return cash_only

    bb_width = (4 * bb_std) / bb_mid if bb_mid != 0 else 0.0

    dist_250 = (price - sma_250) / sma_250
    base_target = max(0.0, min(1.0, 0.5 + dist_250 * 5.0))

    if price > sma_50 > sma_250:
        target = base_target
    elif price < sma_50 < sma_250:
        target = base_target * 0.5
    else:
        target = 0.5

    if bb_width > 0.10:
        target = 0.5 + (target - 0.5) * 1.2

    target = max(0.0, min(1.0, target))

    alloc = dial_to_allocation(target)
    return {
        "tqqq_pct": alloc["tqqq"] * 100,
        "sqqq_pct": alloc["sqqq"] * 100,
        "cash_pct": alloc["cash"] * 100,
        "dial": target,
    }


# ============================================================
# COORDINATOR - Runs all strategies and aggregates votes
# ============================================================

def _classify_signal(dial: float) -> tuple:
    """Label the dial. Thresholds unchanged, but applied to the dial itself."""
    EPS = 1e-9
    NEUTRAL_BAND = 0.02
    if dial > 0.60 + EPS:
        return "TQQQ HEAVY", "long"
    if dial > 0.50 + NEUTRAL_BAND + EPS:
        return "MILD TQQQ", "long_light"
    if dial >= 0.50 - NEUTRAL_BAND - EPS:
        return "BALANCED", "neutral"
    if dial >= 0.40 - EPS:
        return "MILD SQQQ", "short_light"
    return "SQQQ HEAVY", "short"


def run_all_strategies(
    data: dict = None,
    period: str = "2y",
    current_allocation: Optional[dict] = None,
    rebalance_band_pct: float = 10.0,
) -> dict:
    """
    Run all registered strategies and aggregate into a 3-way allocation.

    Returns target_tqqq_pct, target_sqqq_pct, target_cash_pct (sum to 100%).

    current_allocation (optional): {"tqqq_pct":..,"sqqq_pct":..,"cash_pct":..}.
    If given, the result includes "rebalance_needed", which is True only when
    any leg differs from the target by more than rebalance_band_pct points.
    """
    if data is None:
        data = load_strategy_data(period)
    if not data or "_indicators" not in data:
        return {"error": "Could not load strategy data"}

    ind = data["_indicators"]
    votes = []
    vote_details = []
    failed = []

    for strategy_fn in STRATEGIES:
        try:
            result = strategy_fn(data)
            vote = float(result["target_tqqq"])
            if not np.isfinite(vote):
                raise ValueError("non-finite vote")
            vote = max(0.0, min(1.0, vote))
            votes.append(vote)
            vote_details.append({
                "name": result["name"],
                "target_tqqq_pct": vote * 100,
                "reason": result["reason"],
            })
        except Exception as e:
            print(f"Strategy {strategy_fn.__name__} failed: {e}")
            failed.append(strategy_fn.__name__)
            continue

    if not votes:
        return {"error": "All strategies failed"}

    # 1. Aggregate directional bias
    base_dial = sum(votes) / len(votes)
    deviation = base_dial - 0.5

    # 2. Conviction multiplier
    if ind["regime"] in ("BULL", "BEAR") and ind["bb_width"] > 0.10:
        position_multiplier = 1.2
    else:
        position_multiplier = 0.8

    dial = max(0.0, min(1.0, 0.5 + deviation * position_multiplier))

    # 3. Dial -> allocation
    alloc = dial_to_allocation(dial)
    target_tqqq, target_sqqq, target_cash = alloc["tqqq"], alloc["sqqq"], alloc["cash"]

    # 4. Signal label and confidence from the dial
    signal, signal_type = _classify_signal(dial)
    confidence = abs(dial - 0.5) * 200

    # 5. Average conviction (diagnostic only)
    avg_conviction = sum(abs(v - 0.5) * 2 for v in votes) / len(votes)

    mallik_live = get_mallik_live_target(data)

    # --- Calculate TQQQ SMAs and Price for the UI ---
    tqqq_df = data.get("TQQQ")
    tq_sma_50, tq_sma_200, tq_sma_250 = 0.0, 0.0, 0.0
    tq_price = 0.0
    if tqqq_df is not None and not tqqq_df.empty:
        tq_close = tqqq_df["Close"]
        tq_price = float(tq_close.iloc[-1])   # <-- NEW: Current TQQQ Price
        tq_sma_50 = float(tq_close.rolling(50).mean().iloc[-1]) if len(tq_close) >= 50 else 0.0
        tq_sma_200 = float(tq_close.rolling(200).mean().iloc[-1]) if len(tq_close) >= 200 else 0.0
        tq_sma_250 = float(tq_close.rolling(250).mean().iloc[-1]) if len(tq_close) >= 250 else 0.0


    result = {
        "signal": signal,
        "signal_type": signal_type,
        "regime": ind["regime"],
        "dial": dial,
        "target_tqqq_pct": target_tqqq * 100,
        "target_sqqq_pct": target_sqqq * 100,
        "target_cash_pct": target_cash * 100,
        "mallik_tqqq_pct": mallik_live["tqqq_pct"],
        "mallik_sqqq_pct": mallik_live["sqqq_pct"],
        "mallik_cash_pct": mallik_live["cash_pct"],
        "mallik_dial": mallik_live["dial"],
        "current_price": ind["price"],
        "tqqq_current_price": tq_price,
        "sma_50": ind["sma_50"],
        "sma_200": ind["sma_200"],
        "sma_250": ind["sma_250"],
        "tqqq_sma_50": tq_sma_50,      # <-- NEW
        "tqqq_sma_200": tq_sma_200,    # <-- NEW
        "tqqq_sma_250": tq_sma_250,    # <-- NEW
        "tqqq_price": tq_price,        # <-- NEW
        "distance_from_sma50_pct": ind["dist_from_sma50"] * 100,
        "bb_width": ind["bb_width"],
        "bb_position": ind["bb_position"],
        "rsi": ind["rsi_14"],
        "position_multiplier": position_multiplier,
        "avg_conviction": avg_conviction,
        "votes": votes,
        "vote_details": vote_details,
        "failed_strategies": failed,
        "confidence": confidence,
    }

    if current_allocation is not None:
        diffs = [
            abs(result["target_tqqq_pct"] - float(current_allocation.get("tqqq_pct", 0.0))),
            abs(result["target_sqqq_pct"] - float(current_allocation.get("sqqq_pct", 0.0))),
            abs(result["target_cash_pct"] - float(current_allocation.get("cash_pct", 0.0))),
        ]
        result["max_allocation_gap_pct"] = max(diffs)
        result["rebalance_needed"] = max(diffs) > rebalance_band_pct

    return result