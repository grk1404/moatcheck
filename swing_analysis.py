"""
Swing analysis — find meaningful up/down moves in a price series.

A "swing" is a peak-to-trough or trough-to-peak move of at least `threshold`
(e.g. 15%). Moves smaller than that threshold are ignored, so a series of
small daily wiggles doesn't spam the report.

State machine:
    We alternate between looking for a DOWN move (from a peak) and an UP
    move (from a trough). While looking for a DOWN move, we track the highest
    price seen; if price falls `threshold` from that peak, we log a DOWN move
    and flip state. Symmetrically for UP moves.
"""
from __future__ import annotations

import pandas as pd


def find_swings(prices: pd.Series, threshold: float = 0.15) -> pd.DataFrame:
    """Find peak-to-trough and trough-to-peak moves of at least `threshold`.

    Args:
        prices: Daily close series, indexed by date.
        threshold: minimum move fraction (0.15 = 15%).

    Returns:
        DataFrame with columns:
            date, direction, from_date, from_price, to_price, move_pct, days_since_prior
        where direction is "UP" or "DOWN". Empty if no qualifying swings.
    """
    if prices is None or prices.empty:
        return pd.DataFrame()

    prices = prices.dropna()
    if len(prices) < 2:
        return pd.DataFrame()

    moves = []
    ref_type = "peak"                          # start assuming first point is a peak
    running_extreme_date = prices.index[0]
    running_extreme_price = float(prices.iloc[0])

    for date, px in prices.iloc[1:].items():
        px = float(px)

        if ref_type == "peak":
            # Looking for a DOWN move of >= threshold from the running peak
            if px > running_extreme_price:
                running_extreme_date = date
                running_extreme_price = px
            elif px <= running_extreme_price * (1 - threshold):
                moves.append({"date": date, "direction": "DOWN", "from_date": running_extreme_date, "from_price": running_extreme_price, "to_price": px, "move_pct": (px - running_extreme_price) / running_extreme_price})
                ref_type = "trough"
                running_extreme_date = date
                running_extreme_price = px
        else:
            # Looking for an UP move of >= threshold from the running trough
            if px < running_extreme_price:
                running_extreme_date = date
                running_extreme_price = px
            elif px >= running_extreme_price * (1 + threshold):
                moves.append({"date": date, "direction": "UP", "from_date": running_extreme_date, "from_price": running_extreme_price, "to_price": px, "move_pct": (px - running_extreme_price) / running_extreme_price})
                ref_type = "peak"
                running_extreme_date = date
                running_extreme_price = px

    if not moves:
        return pd.DataFrame()

    df = pd.DataFrame(moves)
    df["days_since_prior"] = df["date"].diff().dt.days
    return df