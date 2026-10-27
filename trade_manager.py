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
    "sqqq_target_pct", "cash_target_pct", "account_value", "notes",
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


def save_position(tqqq_shares: float, sqqq_shares: float, tqqq_avg_cost: float = None, sqqq_avg_cost: float = None, cash_balance: float = None) -> None:
    """Persist the current position to disk, including cash on hand."""
    data = {
        "tqqq_shares": float(tqqq_shares),
        "sqqq_shares": float(sqqq_shares),
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if tqqq_avg_cost is not None:
        data["tqqq_avg_cost"] = float(tqqq_avg_cost)
    if sqqq_avg_cost is not None:
        data["sqqq_avg_cost"] = float(sqqq_avg_cost)
    if cash_balance is not None:
        data["cash_balance"] = float(cash_balance)
    POSITION_FILE.write_text(json.dumps(data, indent=2))

def compute_running_pnl(account_value: float = None) -> dict:
    """Compute P&L. Prefers the live account_value from the UI, 
    falls back to shares + cash if not provided."""
    cap = load_capital()
    starting = cap.get("starting_capital", 0.0)
    deposits = sum(x["amount"] for x in cap.get("deposits", []))
    withdrawals = sum(x["amount"] for x in cap.get("withdrawals", []))
    net = starting + deposits - withdrawals

    pos = load_position()
    
    if account_value is not None:
        sleeve_value = account_value
    else:
        tqqq_shares = pos.get("tqqq_shares", 0.0)
        sqqq_shares = pos.get("sqqq_shares", 0.0)
        cash_balance = pos.get("cash_balance", 0.0)
        try:        
            tqqq_price = get_ticker("TQQQ").history(period="1d")["Close"].iloc[-1]
            sqqq_price = get_ticker("SQQQ").history(period="1d")["Close"].iloc[-1]
        except Exception:
            tqqq_price = sqqq_price = 0.0
        sleeve_value = tqqq_shares * tqqq_price + sqqq_shares * sqqq_price + cash_balance

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

def compute_target_shares(account_value, tqqq_target_pct, tqqq_price, sqqq_price,
                          cash_target_pct: float = 0.0):
    """
    Compute fractional target share counts for both legs, reserving cash.

    cash_target_pct is a PERCENT (0-100), matching tqqq_target_pct's units.
    It reduces the investable base; the residual cash is NOT reallocated
    to TQQQ/SQQQ.

    Returns:
        {
            'tqqq_target_value': float,
            'sqqq_target_value': float,
            'tqqq_target_shares': float,
            'sqqq_target_shares': float,
            'cash_value': float,       # ← NEW
            'invested_value': float,   # ← NEW
        }
    """
    assert 0.0 <= cash_target_pct <= 100.0, (
        f"cash_target_pct must be a percent (0-100), got {cash_target_pct}"
    )

    invested_value = account_value * (1.0 - cash_target_pct / 100.0)

    tqqq_value = invested_value * (tqqq_target_pct / 100.0)
    sqqq_value = invested_value - tqqq_value   # residual, not re-derived from pct

    tqqq_shares = tqqq_value / tqqq_price if tqqq_price > 0 else 0.0
    sqqq_shares = sqqq_value / sqqq_price if sqqq_price > 0 else 0.0

    return {
        "tqqq_target_value": tqqq_value,
        "sqqq_target_value": sqqq_value,
        "tqqq_target_shares": tqqq_shares,
        "sqqq_target_shares": sqqq_shares,
        "cash_value": account_value - invested_value,
        "invested_value": invested_value,
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
              signal: str, tqqq_target_pct: float, sqqq_target_pct: float, cash_target_pct: float,
              account_value: float, notes: str = "") -> None:
    """Append a trade to the CSV log."""
    import os
    TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    exists = TRADES_FILE.exists()

    needs_leading_newline = False
    if exists:
        try:
            with open(TRADES_FILE, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size > 0:
                    f.seek(-1, 2)
                    last = f.read(1)
                    if last not in (b"\n", b"\r"):
                        needs_leading_newline = True
        except (OSError, ValueError):
            pass

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker.upper(),
        "action": action.upper(),
        "shares": round(shares, 4),
        "price": round(price, 4),
        "total_value": round(shares * price, 2),
        "signal": signal,
        "tqqq_target_pct": tqqq_target_pct,
        "sqqq_target_pct": sqqq_target_pct,
        "cash_target_pct": cash_target_pct,
        "account_value": round(account_value, 2),
        "notes": notes,
    }

    with open(TRADES_FILE, "a", newline="", encoding="utf-8") as f:
        if needs_leading_newline:
            f.write("\n")
        writer = csv.DictWriter(f, fieldnames=TRADE_COLUMNS, lineterminator="\r\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())

def repair_trade_log() -> int:
    """Detect and split concatenated rows in the trade log.

    Reads raw lines, finds any line with more than one timestamp-like
    token, and splits it into multiple lines at each timestamp boundary.
    Returns the number of rows repaired.
    """
    import re

    if not TRADES_FILE.exists():
        print("[repair] No trade log to repair.")
        return 0

    with open(TRADES_FILE, "r", encoding="utf-8") as f:
        raw = f.read()

    lines = raw.splitlines()
    if not lines:
        return 0

    # A row starts with YYYY-MM-DD HH:MM:SS followed by a comma
    timestamp_re = re.compile(r"(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},)")

    repaired = 0
    out_lines = []
    for line in lines:
        # Count how many timestamp markers appear in the line
        positions = [m.start() for m in timestamp_re.finditer(line)]
        if len(positions) <= 1:
            out_lines.append(line)
            continue

        # Split at each timestamp position
        pieces = []
        for i, pos in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(line)
            piece = line[pos:end].rstrip(",")
            pieces.append(piece)
        out_lines.extend(pieces)
        repaired += len(positions) - 1

    if repaired:
        # Back up the original
        backup = TRADES_FILE.with_suffix(".csv.pre_repair")
        TRADES_FILE.rename(backup)
        print(f"[repair] Backed up original to {backup}")

        # Write the repaired file
        with open(TRADES_FILE, "w", encoding="utf-8", newline="") as f:
            for line in out_lines:
                f.write(line + "\r\n")
        print(f"[repair] Repaired {repaired} concatenated row(s).")

    return repaired

def load_trades() -> pd.DataFrame:
    """Load the trade log as a DataFrame (empty if no file yet).

    Tolerates malformed rows by skipping them rather than crashing the app.
    Prints a warning when a row is skipped so the corruption is visible.
    """
    if not TRADES_FILE.exists():
        return pd.DataFrame(columns=TRADE_COLUMNS)

    # Strict parse first. If it fails, fall back to a tolerant parse that
    # skips bad rows instead of raising.
    try:
        df = pd.read_csv(
            TRADES_FILE,
            engine="python",
            on_bad_lines="error",
        )
    except pd.errors.ParserError as e:
        print(f"[trades] Malformed rows detected in {TRADES_FILE}: {e}")
        print("[trades] Falling back to tolerant parse — bad rows will be skipped.")
        df = pd.read_csv(
            TRADES_FILE,
            engine="python",
            on_bad_lines="skip",
        )

    # Ensure every expected column exists (in case the header was damaged)
    for col in TRADE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df


def clear_trades() -> None:
    """Delete the trade log file."""
    if TRADES_FILE.exists():
        TRADES_FILE.unlink()