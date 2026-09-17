"""
Trade Manager
- Calculates target share counts (including fractional) from signal + capital
- Logs every trade to a CSV file for later accuracy analysis
"""

from data_provider import get_ticker
import csv
import json
import pandas as pd
from datetime import datetime

from pathlib import Path

DATA_DIR = Path("data")
TRADES_FILE = DATA_DIR / "trade_log.csv"
POSITION_FILE = DATA_DIR / "current_position.json"
CAPITAL_FILE = DATA_DIR / "capital.json"

TRADE_COLUMNS = [
    "timestamp", "ticker", "action", "shares", "price",
    "total_value", "signal", "tqqq_target_pct",
    "sqqq_target_pct", "account_value", "notes",
]



def load_capital() -> dict:
    """Read the capital tracker. Initializes with defaults if missing."""
    if not CAPITAL_FILE.exists():
        return {
            "starting_capital": 0.0,
            "deposits": [],
            "withdrawals": [],
            "started_at": None,
        }
    try:
        return json.loads(CAPITAL_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {
            "starting_capital": 0.0,
            "deposits": [],
            "withdrawals": [],
            "started_at": None,
        }


def save_capital(data: dict) -> None:
    CAPITAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    CAPITAL_FILE.write_text(json.dumps(data, indent=2))


def net_capital() -> float:
    """Total capital committed to the strategy = starting + deposits − withdrawals."""
    d = load_capital()
    deposits = sum(x["amount"] for x in d.get("deposits", []))
    withdrawals = sum(x["amount"] for x in d.get("withdrawals", []))
    return d.get("starting_capital", 0.0) + deposits - withdrawals

def load_position() -> dict:
    """Read the current TQQQ/SQQQ position. Returns zeros if missing."""
    if not POSITION_FILE.exists():
        return {"tqqq_shares": 0.0, "sqqq_shares": 0.0, "updated": None}
    try:
        return json.loads(POSITION_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {"tqqq_shares": 0.0, "sqqq_shares": 0.0, "updated": None}


def save_position(tqqq_shares: float, sqqq_shares: float) -> None:
    """Persist the current position to disk."""
    POSITION_FILE.parent.mkdir(parents=True, exist_ok=True)
    POSITION_FILE.write_text(json.dumps({
        "tqqq_shares": float(tqqq_shares),
        "sqqq_shares": float(sqqq_shares),
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))

def compute_running_pnl() -> dict:
    """
    Compute the current cumulative P&L for the sleeve.

    Returns:
        {
            'net_capital': float,       # starting + deposits − withdrawals
            'sleeve_value': float,      # current TQQQ+SQQQ market value
            'pnl_dollars': float,       # sleeve_value − net_capital
            'pnl_pct': float,           # pnl_dollars / net_capital * 100
            'updated': str,             # timestamp of the position file
        }
    Returns zeros if no capital or position data exists yet.
    """
    cap = load_capital()
    starting = cap.get("starting_capital", 0.0)
    deposits = sum(x["amount"] for x in cap.get("deposits", []))
    withdrawals = sum(x["amount"] for x in cap.get("withdrawals", []))
    net = starting + deposits - withdrawals

    pos = load_position()
    tqqq_shares = pos.get("tqqq_shares", 0.0)
    sqqq_shares = pos.get("sqqq_shares", 0.0)

    # Fetch current prices to value the sleeve
    try:        
        tqqq_price = get_ticker("TQQQ").history(period="1d")["Close"].iloc[-1]
        sqqq_price = get_ticker("SQQQ").history(period="1d")["Close"].iloc[-1]
    except Exception:
        tqqq_price = sqqq_price = 0.0

    sleeve_value = tqqq_shares * tqqq_price + sqqq_shares * sqqq_price

    pnl = sleeve_value - net
    pnl_pct = (pnl / net * 100) if net > 0 else 0.0

    return {
        "net_capital": net,
        "sleeve_value": sleeve_value,
        "pnl_dollars": pnl,
        "pnl_pct": pnl_pct,
        "updated": pos.get("updated"),
    }


def compute_realized_pnl() -> pd.DataFrame:
    """
    Compute realized P&L per trade by matching buys and sells chronologically
    using average-cost accounting.

    For each ticker (TQQQ, SQQQ), maintain a running position:
      - BUY  → add shares at that price to the running cost basis
      - SELL → realize P&L = (sell_price − avg_cost) × shares_sold
               and reduce the position's cost basis proportionally

    Returns the trade log (chronological order) with new columns:
      avg_cost_before, realized_pnl_this_trade, cumulative_realized_pnl
    """
    df = load_trades()
    if df.empty:
        return df

    # Ensure chronological order
    df = df.sort_values("timestamp").reset_index(drop=True)

    positions = {
        "TQQQ": {"shares": 0.0, "cost_basis": 0.0},
        "SQQQ": {"shares": 0.0, "cost_basis": 0.0},
    }

    avg_cost_before = []
    realized_this_trade = []
    cumulative_realized = []
    running_total = 0.0

    for _, row in df.iterrows():
        ticker = row["ticker"].upper()
        action = row["action"].upper()
        shares = float(row["shares"])
        price = float(row["price"])

        pos = positions.setdefault(ticker, {"shares": 0.0, "cost_basis": 0.0})
        prev_avg_cost = (pos["cost_basis"] / pos["shares"]) if pos["shares"] > 0 else 0.0

        if action == "BUY":
            pos["shares"] += shares
            pos["cost_basis"] += shares * price
            realized = 0.0
        elif action == "SELL":
            # Realize P&L on the shares being sold, using current avg cost
            shares_sold = min(shares, pos["shares"])
            realized = (price - prev_avg_cost) * shares_sold

            # Reduce position proportionally
            if pos["shares"] > 0:
                cost_per_share = pos["cost_basis"] / pos["shares"]
                pos["cost_basis"] -= cost_per_share * shares_sold
                pos["shares"] -= shares_sold
                if pos["shares"] <= 1e-9:
                    pos["shares"] = 0.0
                    pos["cost_basis"] = 0.0
        else:
            realized = 0.0

        running_total += realized

        avg_cost_before.append(round(prev_avg_cost, 4))
        realized_this_trade.append(round(realized, 2))
        cumulative_realized.append(round(running_total, 2))

    df["avg_cost_before"] = avg_cost_before
    df["realized_pnl_this_trade"] = realized_this_trade
    df["cumulative_realized_pnl"] = cumulative_realized
    return df

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
    """Append a trade to the CSV log with total_value computed from shares × price."""
    TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    exists = TRADES_FILE.exists()

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
    if not TRADES_FILE.exists():
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.read_csv(TRADES_FILE)


def clear_trades() -> None:
    """Delete the trade log file."""
    if TRADES_FILE.exists():
        TRADES_FILE.unlink()