"""
TQQQ/SQQQ Multi-Strategy Framework
Seven independent sub-strategies that each vote on target allocation.
The coordinator averages votes to produce a daily target allocation.

Each strategy follows the same signature:
    strategy_name(data: dict) -> dict
        returns:
            {
                'name': str,
                'target_tqqq': float,   # 0.0 to 1.0 (1.0 = 100% TQQQ)
                'reason': str,
                'weight': float,        # optional weight (default 1.0)
            }

Add new strategies by writing a function and registering it in STRATEGIES.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ============================================================
# DATA LOADER - Fetches QQQ, TQQQ, SQQQ for regime detection
# ============================================================

def load_strategy_data(period: str = "2y") -> dict:
    """
    Fetch QQQ, TQQQ, SQQQ data and precompute common indicators.
    All strategies read from this shared data dict.
    """
    data = {}
    for ticker in ["QQQ", "TQQQ", "SQQQ"]:
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df.empty:
                continue
            data[ticker] = df
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")

    if "QQQ" not in data:
        return {}

    qqq = data["QQQ"]
    close = qqq["Close"]

    # Precompute shared indicators on QQQ
    data["_indicators"] = {
        "price": close.iloc[-1],
        "sma_20": close.rolling(20).mean().iloc[-1],
        "sma_50": close.rolling(50).mean().iloc[-1],
        "sma_200": close.rolling(200).mean().iloc[-1],
        "ema_12": close.ewm(span=12, adjust=False).mean().iloc[-1],
        "ema_26": close.ewm(span=26, adjust=False).mean().iloc[-1],
        "bb_mid": close.rolling(20).mean().iloc[-1],
        "bb_std": close.rolling(20).std().iloc[-1],
        "rsi_14": _compute_rsi(close, 14),
        "atr_14": _compute_atr(qqq, 14),
        "dist_from_sma50": (close.iloc[-1] - close.rolling(50).mean().iloc[-1]) / close.rolling(50).mean().iloc[-1],
        "dist_from_sma200": (close.iloc[-1] - close.rolling(200).mean().iloc[-1]) / close.rolling(200).mean().iloc[-1],
    }

    ind = data["_indicators"]
    ind["bb_upper"] = ind["bb_mid"] + 2 * ind["bb_std"]
    ind["bb_lower"] = ind["bb_mid"] - 2 * ind["bb_std"]
    ind["bb_width"] = (ind["bb_upper"] - ind["bb_lower"]) / ind["bb_mid"]
    ind["bb_position"] = (ind["price"] - ind["bb_lower"]) / (ind["bb_upper"] - ind["bb_lower"]) * 100

    # Regime classification
    if ind["price"] > ind["sma_50"] > ind["sma_200"]:
        ind["regime"] = "BULL"
    elif ind["price"] < ind["sma_50"] < ind["sma_200"]:
        ind["regime"] = "BEAR"
    else:
        ind["regime"] = "NEUTRAL"

    return data


def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return (100 - 100 / (1 + rs)).iloc[-1]


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]


# ============================================================
# STRATEGY 1: Momentum Long TQQQ (Trend Following)
# ============================================================

def strategy_momentum_long(data: dict) -> dict:
    ind = data["_indicators"]
    regime = ind["regime"]
    price = ind["price"]
    sma_50 = ind["sma_50"]

    if regime == "BULL" and price > sma_50:
        return {"name": "Momentum Long", "target_tqqq": 1.0,
                "reason": "Bull regime + price above SMA 50 — full TQQQ"}
    if regime == "BEAR":
        return {"name": "Momentum Long", "target_tqqq": 0.0,
                "reason": "Bear regime — avoid TQQQ"}
    return {"name": "Momentum Long", "target_tqqq": 0.3,
            "reason": "Neutral regime — reduced TQQQ exposure"}


# ============================================================
# STRATEGY 2: Mean Reversion TQQQ (Oversold Bounce)
# ============================================================

def strategy_mean_reversion(data: dict) -> dict:
    ind = data["_indicators"]
    dist = ind["dist_from_sma50"]
    regime = ind["regime"]

    if dist < -0.08 and regime != "BEAR":
        return {"name": "Mean Reversion", "target_tqqq": 0.8,
                "reason": f"Oversold: {dist*100:.1f}% below SMA 50 — bounce likely"}
    if dist > 0.10:
        return {"name": "Mean Reversion", "target_tqqq": 0.2,
                "reason": f"Overextended: {dist*100:.1f}% above SMA 50 — fade risk"}
    return {"name": "Mean Reversion", "target_tqqq": 0.5,
            "reason": f"Within normal range ({dist*100:.1f}% from SMA 50)"}


# ============================================================
# STRATEGY 3: Bollinger Band Reversion
# ============================================================

def strategy_bb_reversion(data: dict) -> dict:
    ind = data["_indicators"]
    pos = ind["bb_position"]

    if pos < 15:
        return {"name": "BB Reversion", "target_tqqq": 0.9,
                "reason": f"At lower BB ({pos:.0f}%) — oversold"}
    if pos > 85:
        return {"name": "BB Reversion", "target_tqqq": 0.1,
                "reason": f"At upper BB ({pos:.0f}%) — overbought"}
    return {"name": "BB Reversion", "target_tqqq": 0.5,
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
        return {"name": "BB Width Momentum", "target_tqqq": 0.5,
                "reason": f"Narrow BB ({width*100:.1f}%) — weak trend"}
    return {"name": "BB Width Momentum", "target_tqqq": 0.2,
            "reason": "Not in a confirmed bull regime"}


# ============================================================
# STRATEGY 5: Momentum Short (Bearish Trend Following)
# ============================================================

def strategy_momentum_short(data: dict) -> dict:
    ind = data["_indicators"]
    regime = ind["regime"]

    if regime == "BEAR":
        return {"name": "Momentum Short", "target_tqqq": 0.0,
                "reason": "Bear regime confirmed — favor SQQQ"}
    return {"name": "Momentum Short", "target_tqqq": 0.7,
            "reason": "No bearish confirmation — lean TQQQ"}


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
    return {"name": "Short Reversion", "target_tqqq": 0.6,
            "reason": "No fade signal active"}


# ============================================================
# STRATEGY 7: Bearish Bounce Fade
# ============================================================

def strategy_bear_bounce_fade(data: dict) -> dict:
    ind = data["_indicators"]
    dist = ind["dist_from_sma50"]
    regime = ind["regime"]

    if regime == "BEAR" and dist < -0.10:
        return {"name": "Bear Bounce Fade", "target_tqqq": 0.3,
                "reason": "Bear regime oversold bounce — fade strength"}
    return {"name": "Bear Bounce Fade", "target_tqqq": 0.5,
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
# COORDINATOR - Runs all strategies and aggregates votes
# ============================================================

def run_all_strategies(data: dict = None, period: str = "2y") -> dict:
    """
    Run all registered strategies and aggregate their votes.
    Returns the target allocation plus per-strategy vote details.
    """
    if data is None:
        data = load_strategy_data(period)

    if not data or "_indicators" not in data:
        return {"error": "Could not load strategy data"}

    ind = data["_indicators"]
    votes = []
    vote_details = []

    for strategy_fn in STRATEGIES:
        try:
            result = strategy_fn(data)
            votes.append(result["target_tqqq"])
            vote_details.append({
                "name": result["name"],
                "target_tqqq_pct": result["target_tqqq"] * 100,
                "reason": result["reason"],
            })
        except Exception as e:
            print(f"Strategy {strategy_fn.__name__} failed: {e}")
            continue

    if not votes:
        return {"error": "All strategies failed"}

    # Average vote = base allocation
    target_tqqq = sum(votes) / len(votes)

    # Position multiplier based on BB width (trend strength)
    if ind["regime"] in ("BULL", "BEAR") and ind["bb_width"] > 0.10:
        position_multiplier = 1.2
    else:
        position_multiplier = 0.8

    target_tqqq = min(1.0, target_tqqq * position_multiplier)
    target_sqqq = 1.0 - target_tqqq

    # Signal classification
    if target_tqqq > 0.6:
        signal, signal_type = "TQQQ HEAVY", "long"
    elif target_tqqq > 0.4:
        signal, signal_type = "MILD TQQQ", "long_light"
    elif target_sqqq > 0.6:
        signal, signal_type = "SQQQ HEAVY", "short"
    elif target_sqqq > 0.4:
        signal, signal_type = "MILD SQQQ", "short_light"
    else:
        signal, signal_type = "BALANCED", "neutral"

    return {
        "signal": signal,
        "signal_type": signal_type,
        "regime": ind["regime"],
        "target_tqqq_pct": target_tqqq * 100,
        "target_sqqq_pct": target_sqqq * 100,
        "current_price": ind["price"],
        "sma_50": ind["sma_50"],
        "sma_200": ind["sma_200"],
        "distance_from_sma50_pct": ind["dist_from_sma50"] * 100,
        "bb_width": ind["bb_width"],
        "bb_position": ind["bb_position"],
        "rsi": ind["rsi_14"],
        "position_multiplier": position_multiplier,
        "votes": votes,
        "vote_details": vote_details,
        "confidence": abs(target_tqqq - 0.5) * 200,
    }