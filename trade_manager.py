"""
Trade Manager
- Calculates target share counts (including fractional) from signal + capital
- Logs every trade to a CSV file for later accuracy analysis
"""

import os
import csv
import pandas as pd
from datetime import datetime

TRADES_FILE = "trade_log.csv"

TRADE_COLUMNS = [
    "timestamp", "ticker", "action", "shares", "price",
    "total_value", "signal", "tqqq_target_pct",
    "sqqq_target_pct", "account_value", "notes",
]


def compute_target_shares(account_value: float,
                          tqqq_target_pct: float,
                          tqqq_price: float,
                          sqqq_price: float) -> dict:
    """
    Compute fractional target share counts for both legs.

    Returns:
        {
            'tqqq_target_value': float,
            'sqqq_target_value': float,
            'tqqq_target_shares': float,
            'sqqq_target_shares': float,
        }
    """
    sqqq_target_pct = 100.0 - tqqq_target_pct

    tqqq_value = account_value * (tqqq_target_pct / 100.0)
    sqqq_value = account_value * (sqqq_target_pct / 100.0)

    tqqq_shares = tqqq_value / tqqq_price if tqqq_price > 0 else 0.0
    sqqq_shares = sqqq_value / sqqq_price if sqqq_price > 0 else 0.0

    return {
        "tqqq_target_value": tqqq_value,
        "sqqq_target_value": sqqq_value,
        "tqqq_target_shares": tqqq_shares,
        "sqqq_target_shares": sqqq_shares,
    }


def compute_rebalance(current_tqqq_shares: float,
                      current_sqqq_shares: float,
                      tqqq_target_shares: float,
                      sqqq_target_shares: float,
                      tqqq_price: float,
                      sqqq_price: float) -> dict:
    """
    Compute the BUY/SELL actions needed to reach target.
    Positive = BUY, Negative = SELL.
    """
    tqqq_delta = tqqq_target_shares - current_tqqq_shares
    sqqq_delta = sqqq_target_shares - current_sqqq_shares

    return {
        "tqqq_delta_shares": tqqq_delta,
        "sqqq_delta_shares": sqqq_delta,
        "tqqq_delta_value": tqqq_delta * tqqq_price,
        "sqqq_delta_value": sqqq_delta * sqqq_price,
    }


def log_trade(ticker: str, action: str, shares: float, price: float,
              signal: str, tqqq_target_pct: float,
              account_value: float, notes: str = "") -> None:
    """Append a trade to the CSV log."""
    exists = os.path.exists(TRADES_FILE)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker.upper(),
        "action": action.upper(),
        "shares": round(shares, 4),
        "price": round(price, 4),
        "total_value": round(shares * price, 2),
        "signal": signal,
        "tqqq_target_pct": tqqq_target_pct,
        "sqqq_target_pct": round(100.0 - tqqq_target_pct, 2),
        "account_value": round(account_value, 2),
        "notes": notes,
    }

    with open(TRADES_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_trades() -> pd.DataFrame:
    """Load the trade log as a DataFrame (empty if no file yet)."""
    if not os.path.exists(TRADES_FILE):
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.read_csv(TRADES_FILE)


def clear_trades() -> None:
    """Delete the trade log file."""
    if os.path.exists(TRADES_FILE):
        os.remove(TRADES_FILE)