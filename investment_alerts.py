"""
Standalone alert service for MoatCheck.
Checks Fidelity positions for three signal types and sends WhatsApp alerts
via CallMeBot:

  1. 52-week high proximity (default: within 2% of the high)
  2. 52-week low proximity  (default: within 2% of the low)
  3. MOS buy price hit      (reads from data/investments_signals.json — the
                             cache the My Investments page builds)

The 52-week checks run on every cycle. The MOS check runs at most once per day
(guarded by a timestamp in .alert_state.json).

Run manually:   python investment_alerts.py
Install as Windows Service with NSSM (see README).
"""
from __future__ import annotations

import sys
import io

# Force UTF-8 encoding for Windows services so emojis don't crash the cycle
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r".*yfinance_cache.*",
)

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# ----------------------------------------------------------------
# Timestamped logging — patches print() so every line gets a prefix
# like "2026-10-07 14:32:15 [alerts] ...". Delete this block if you
# ever want raw output for debugging.
# ----------------------------------------------------------------
import builtins as _builtins
_real_print = _builtins.print

def _timestamped_print(*args, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _real_print(f"{ts}", *args, **kwargs)

_builtins.print = _timestamped_print

# ============================================================
# CONFIG — edit these values
# ============================================================

CONFIG = {
    "csv_path": "data/fidelity_positions.csv",
    "phone": "18477309134",          # international format, no +
    "api_key": "2633561",  # from CallMeBot

    # 52-week proximity thresholds (fraction of the high/low)
    "high_threshold": 0.98,          # within 2% of 52-week high
    "low_threshold": 0.02,           # within 2% of 52-week low (price/low ratio)

    # Cycle timing
    "check_interval_seconds": 3600,  # 52-week checks: once per hour
    "mos_interval_hours": 24,        # MOS check: once per day

    # Per-ticker cooldown (hours) before re-alerting the same signal
    "cooldown_hours": 24,
    
    "bnf_interval_hours": 24,        # BNF check: once per day
    "bnf_cooldown_hours": 168,       # 7 days per ticker (panic events cluster)
    
    # TQQQ daily signal alert
    "tqqq_signal_interval_hours": 24,       # once per day
    "tqqq_signal_target_hour_ct": 14,       # 2 PM Central Time
    "tqqq_signal_target_minute_ct": 45,     # at 2:45 PM CT
    "tqqq_signal_window_minutes": 15,       # fire if within ±15 minutes

    # Swing Trading signal alert
    "swing_interval_hours": 24,      # 0 = run every hour (change to 24 for daily)
    "swing_thresholds": [0.15, 0.20], # Alert on both 15% and 20% moves
}

# Path to the My Investments cache that holds intrinsic values
SIGNALS_CACHE = Path("data/investments_signals.json")

# Where we persist which alerts we've already sent
STATE_FILE = Path("data/.alert_state.json")

import socket, time

def wait_for_network(hosts=("query1.finance.yahoo.com", "api.callmebot.com"), timeout=180):
    """Block until DNS resolves at least one host, or timeout. Returns bool."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for h in hosts:
            try:
                socket.getaddrinfo(h, 443)
                return True
            except socket.gaierror:
                continue
        time.sleep(5)
    return False

def _in_signal_window() -> bool:
    """Return True if we're currently within the target daily window for
    the TQQQ signal alert. Uses America/Chicago explicitly so the schedule
    is stable across time zone changes and DST transitions."""
    try:
        from zoneinfo import ZoneInfo
        now_ct = datetime.now(ZoneInfo("America/Chicago"))
    except Exception:
        # If zoneinfo fails, fall back to the system's local time
        now_ct = datetime.now()

    target_hour = CONFIG.get("tqqq_signal_target_hour_ct", 14)
    target_min = CONFIG.get("tqqq_signal_target_minute_ct", 45)
    window = CONFIG.get("tqqq_signal_window_minutes", 15)

    # Minutes since midnight, local Chicago time
    now_minutes = now_ct.hour * 60 + now_ct.minute
    target_minutes = target_hour * 60 + target_min

    return abs(now_minutes - target_minutes) <= window

# ============================================================
# CSV PARSING — Fidelity positions export
# ============================================================

def load_positions(csv_path: str) -> pd.DataFrame:
    """Read Fidelity positions CSV, return only real equity rows.

    Standard Fidelity positions export layout (16 header cols, 17 data cols):
      col 2  = Symbol
      col 4  = Quantity
      col 5  = Last price
      col 7  = Current value
      col 14 = Average cost basis
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"[alerts] CSV not found: {path}")
        return pd.DataFrame()

    try:
        import csv

        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                if not row or len(row) < 15:
                    continue
                sym = (row[2] or "").strip()
                if not sym or "*" in sym:
                    continue
                if not sym.replace(".", "").replace("-", "").isalpha():
                    continue
                if len(sym) > 6:
                    continue

                def _num(v):
                    if v is None:
                        return None
                    try:
                        return float(str(v).replace("$", "").replace(",", "").replace("%", "").strip())
                    except Exception:
                        return None

                rows.append({
                    "Symbol": sym,
                    "Quantity": _num(row[4] if len(row) > 4 else None),
                    "LastPrice": _num(row[5] if len(row) > 5 else None),
                    "CurrentValue": _num(row[7] if len(row) > 7 else None),
                    "AvgCost": _num(row[14] if len(row) > 14 else None),
                })

        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[alerts] Error parsing CSV: {e}")
        return pd.DataFrame()

# ============================================================
# BNF Reversal Setup Takashi Kotegawa (BNF) style mean-reversion check. 
# ============================================================
def check_bnf_setups(df: pd.DataFrame) -> list[dict]:
    """Scan positions for the BNF reversal setup."""
    from technical_indicators import TechnicalIndicatorAnalyzer

    print(f"[alerts] Starting BNF reversal check for {len(df)} positions.")

    hits = []
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip().upper()
        if not sym:
            continue
        try:
            a = TechnicalIndicatorAnalyzer(sym, "2y")
            if not a.fetch_data():
                continue
            a.calculate_all_indicators()
            result = a.check_bnf_reversal_setup()
            if result.get("setup"):
                hits.append({"symbol": sym, "price": float(result["current_price"]), "ma_25": float(result["ma_25"]),
                    "dislocation": float(result["dislocation_pct"]),
                    "rsi": float(result["rsi"]),
                })
        except Exception:
            continue
    return hits

# ============================================================
# 52-WEEK HIGH / LOW CHECKS
# ============================================================

def check_52w_highs(df: pd.DataFrame, threshold: float) -> list[dict]:
    """Positions within `threshold` of their 52-week high."""
    from data_provider import get_ticker

    hits = []
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip()
        if not sym:
            continue
        try:
            info = get_ticker(sym).info or {}
            high = info.get("fiftyTwoWeekHigh")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if high is None or price is None:
                continue
            high = float(high)
            price = float(price)
             # Guard: never alert on missing/zero/nonsense prices.
            if high <= 0.1 or price <= 0.1:
                continue
            pct = price / high
            if pct >= threshold:
                hits.append({
                    "symbol": sym,
                    "price": price,
                    "high": high,
                    "pct": pct,
                })
        except Exception:
            continue
    return hits


def check_52w_lows(df: pd.DataFrame, threshold: float) -> list[dict]:
    """Positions within `threshold` of their 52-week low."""
    from data_provider import get_ticker

    hits = []
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip()
        if not sym:
            continue
        try:
            info = get_ticker(sym).info or {}
            low = info.get("fiftyTwoWeekLow")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if low is None or price is None:
                continue
            low = float(low)
            price = float(price)
            # Guard: never alert on missing/zero/nonsense prices.
            if low <= 0.1 or price <= 0.1:
                continue
            pct = (price - low) / low
            if pct <= threshold:
                hits.append({
                    "symbol": sym,
                    "price": price,
                    "low": low,
                    "pct_above": pct,
                })
        except Exception:
            continue
    return hits


# ============================================================
# MOS PRICE CHECK — reads from the My Investments cache
# ============================================================

def check_mos_hits(df: pd.DataFrame) -> list[dict]:
    """Positions trading at or below their MOS buy price."""
    from data_provider import get_ticker
    from moatcheck import fetch, compute_big5, value_price

    print(f"[alerts] Starting MOS price check for {len(df)} positions.")

    hits = []
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip().upper()
        if not sym:
            continue
        try:
            fin = fetch(sym)
        except Exception:
            continue

        try:
            if fin is None or fin.current_price is None or fin.current_price <= 0:
                continue

            big5 = compute_big5(fin)
            eps_windows = big5.eps.values
            stable = [v for w, v in eps_windows.items() if w in (5, 3, 10) and v is not None]
            if stable:
                stable.sort()
                n = len(stable)
                if n % 2:
                    growth = stable[n // 2]
                else:
                    growth = (stable[n // 2 - 1] + stable[n // 2]) / 2
            else:
                growth = eps_windows.get(1)

            if growth is None:
                continue

            if growth > 0.30:
                growth = 0.30
            if growth < 0.02:
                growth = 0.02

            eps_ttm = float(fin.eps.iloc[-1]) if not fin.eps.empty else None
            current_eps = eps_ttm
            if current_eps is None or current_eps <= 0:
                if not fin.eps.empty:
                    positive = fin.eps[fin.eps > 0]
                    if not positive.empty:
                        current_eps = float(positive.tail(3).mean())

            if current_eps is None or current_eps <= 0:
                continue

            val = value_price(
                current_eps=current_eps,
                big5_eps_growth=None,
                analyst_growth=None,
                historical_pe=fin.pe_ratio_ttm,
                custom_growth=0.15,
            )
            if val is None or val.value_price is None or val.value_price <= 0:
                continue

            mos_price = val.mos_price
            price = float(fin.current_price)

            if price <= mos_price:
                hits.append({
                    "symbol": sym,
                    "price": price,
                    "mos_price": mos_price,
                    "intrinsic": float(val.value_price),
                    "discount": (price - mos_price) / mos_price,
                })
        except Exception:
            continue

    return hits


# ============================================================
# STATE — track what we've already alerted
# ============================================================

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _cooldown_key(kind: str, symbol: str) -> str:
    return f"{kind}:{symbol}"


def should_alert(kind: str, symbol: str, state: dict, cooldown_hours: int) -> bool:
    last = state.get(_cooldown_key(kind, symbol))
    if last is None:
        return True
    return (time.time() - last) > cooldown_hours * 3600


def mark_alerted(kind: str, symbol: str, state: dict) -> None:
    state[_cooldown_key(kind, symbol)] = time.time()


# ============================================================
# WHATSAPP — CallMeBot
# ============================================================

def send_whatsapp(phone: str, api_key: str, message: str) -> bool:
    if api_key == "YOUR_API_KEY_HERE":
        print("[alerts] API key not set — skipping send.")
        return False
    print(f"[alerts] Sending WhatsApp message to:", phone)
    print(f"[alerts] Message content: {message}")
    url = "https://api.callmebot.com/whatsapp.php"
    params = {"phone": phone, "text": message, "apikey": api_key}
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            print(f"[alerts] WhatsApp sent to {phone}")
            return True
        print(f"[alerts] CallMeBot HTTP {r.status_code}: {r.text[:200]}")
        return False
    except Exception as e:
        print(f"[alerts] Send failed: {type(e).__name__}: {e}")
        return False


# ============================================================
# TQQQ SWING ALERTS
# ============================================================

def check_tqqq_swing_alerts() -> list[dict]:
    """Check if TQQQ has triggered a 15% or 20% BUY or SELL swing alert."""
    from data_provider import get_ticker
    from trade_manager import load_position
    
    # 1. Get current price
    try:
        hist = get_ticker("TQQQ").history(period="1y")
        if hist.empty:
            return []
        current_price = float(hist["Close"].iloc[-1])
    except Exception as e:
        print(f"[alerts] Failed to get TQQQ price: {e}")
        return []

    pos = load_position()
    tq_qty = float(pos.get("tqqq_shares", 0.0))
    tq_avg = pos.get("tqqq_avg_cost")
    
    # Fallback to $70 if no position is logged
    if tq_avg is None:
        tq_avg = 70.00

    alerts = []
    thresholds = CONFIG.get("swing_thresholds", [0.15, 0.20])
    
    # --- SELL LOGIC (If you hold shares) ---
    if tq_qty > 0:
        for pct in thresholds:
            target = tq_avg * (1 + pct)
            if current_price >= target:
                alerts.append({
                    "type": "SELL",
                    "pct": pct,
                    "current": current_price,
                    "target": target,
                    "entry": tq_avg
                })

    # --- BUY LOGIC (If you are looking for an entry) ---
    # Look for a 15% drop from the highest price in the last 20 days
    running_high = float(hist["High"].tail(20).max())
    for pct in thresholds:
        target = running_high * (1 - pct)
        if current_price <= target:
            alerts.append({
                "type": "BUY",
                "pct": pct,
                "current": current_price,
                "target": target,
                "high": running_high
            })

    return alerts


# ============================================================
# CYCLE
# ============================================================

def _run_52w_checks(df: pd.DataFrame, state: dict) -> int:
    """High + low checks. Returns number of alerts sent."""
    print("[alerts] Running 52-week high and low checks.. ")
    cfg = CONFIG
    sent = 0

    # --- 52-week high ---
    highs = check_52w_highs(df, cfg["high_threshold"])
    for hit in highs:
        sym = hit["symbol"]
        if not should_alert("HIGH", sym, state, cfg["cooldown_hours"]):
            continue
        msg = (
            f"🚨 52-Week High Alert\n"
            f"{sym}: {hit['price']:.2f} USD "
            f"({hit['pct']*100:.1f}% of 52w high {hit['high']:.2f} USD)"
        )
        if send_whatsapp(cfg["phone"], cfg["api_key"], msg):
            mark_alerted("HIGH", sym, state)
            sent += 1

    # --- 52-week low ---
    lows = check_52w_lows(df, cfg["low_threshold"])
    for hit in lows:
        sym = hit["symbol"]
        if not should_alert("LOW", sym, state, cfg["cooldown_hours"]):
            continue
        msg = (
            f"⚠️ 52-Week Low Alert\n"
            f"{sym}: {hit['price']:.2f} USD "
            f"({hit['pct_above']*100:+.1f}% above 52w low {hit['low']:.2f} USD)"
        )
        if send_whatsapp(cfg["phone"], cfg["api_key"], msg):
            mark_alerted("LOW", sym, state)
            sent += 1

    return sent


def _run_mos_check(df: pd.DataFrame, state: dict) -> int:
    """MOS hit check. Returns number of alerts sent."""
    cfg = CONFIG
    sent = 0
    hits = check_mos_hits(df)
    for hit in hits:
        sym = hit["symbol"]
        if not should_alert("MOS", sym, state, cfg["cooldown_hours"]):
            continue
        msg = (
            f"🎯 MOS Buy Price Hit\n"
            f"{sym}: {hit['price']:.2f} USD ≤ MOS {hit['mos_price']:.2f} USD "
            f"(intrinsic {hit['intrinsic']:.2f} USD, {hit['discount']*100:+.1f}%)"
        )
        if send_whatsapp(cfg["phone"], cfg["api_key"], msg):
            mark_alerted("MOS", sym, state)
            sent += 1
    return sent

def _run_bnf_check(df: pd.DataFrame, state: dict) -> int:
    """BNF reversal scan. Returns number of alerts sent."""
    cfg = CONFIG
    sent = 0
    hits = check_bnf_setups(df)
    for hit in hits:
        sym = hit["symbol"]
        if not should_alert("BNF", sym, state, cfg["bnf_cooldown_hours"]):
            continue
        msg = (
            f"🔄 BNF Reversal Setup\n"
            f"{sym}: {hit['price']:.2f} USD\n"
            f"Dislocation: {hit['dislocation']:.1f}% below MA25 ({hit['ma_25']:.2f} USD)\n"
            f"RSI: {hit['rsi']:.1f}"
        )
        if send_whatsapp(cfg["phone"], cfg["api_key"], msg):
            mark_alerted("BNF", sym, state)
            sent += 1
    return sent

def check_tqqq_signal() -> dict | None:
    """Run the live TQQQ/SQQQ strategy and compute the daily action."""
    print(f"[alerts] Starting TQQQ/SQQQ live strategy check.")
    try:
        from tqqq_sqqq_strategies import run_all_strategies
        from trade_manager import load_position
        from data_provider import get_ticker

        r = run_all_strategies()
        if not r or "error" in r:
            return None

        target_tq = float(r["target_tqqq_pct"])
        target_sq = float(r["target_sqqq_pct"])

        pos = load_position()
        cur_tq = float(pos.get("tqqq_shares", 0.0))
        cur_sq = float(pos.get("sqqq_shares", 0.0))

        try:
            tq_px = float(get_ticker("TQQQ").history(period="1d")["Close"].iloc[-1])
            sq_px = float(get_ticker("SQQQ").history(period="1d")["Close"].iloc[-1])
        except Exception:
            return None

        account_value = cur_tq * tq_px + cur_sq * sq_px
        if account_value <= 0:
            return None

        tq_target_shares = (account_value * target_tq / 100.0) / tq_px if tq_px > 0 else 0.0
        sq_target_shares = (account_value * target_sq / 100.0) / sq_px if sq_px > 0 else 0.0

        tq_delta = tq_target_shares - cur_tq
        sq_delta = sq_target_shares - cur_sq

        tq_delta_pct = abs(tq_delta) * tq_px / account_value * 100.0
        sq_delta_pct = abs(sq_delta) * sq_px / account_value * 100.0

        MIN_TRADE_PCT = 2.0

        if tq_delta_pct < MIN_TRADE_PCT:
            tq_action = f"No action needed (delta {tq_delta_pct:.2f}%)"
        elif tq_delta > 0:
            tq_action = f"BUY {tq_delta:.4f} shares TQQQ (~{abs(tq_delta) * tq_px:.2f} USD)"
        else:
            tq_action = f"SELL {abs(tq_delta):.4f} shares TQQQ (~{abs(tq_delta) * tq_px:.2f} USD)"

        if sq_delta_pct < MIN_TRADE_PCT:
            sq_action = f"No action needed (delta {sq_delta_pct:.2f}%)"
        elif sq_delta > 0:
            sq_action = f"BUY {sq_delta:.4f} shares SQQQ (~{abs(sq_delta) * sq_px:.2f} USD)"
        else:
            sq_action = f"SELL {abs(sq_delta):.4f} shares SQQQ (~{abs(sq_delta) * sq_px:.2f} USD)"

        return {
            "signal": r["signal"],
            "regime": r["regime"],
            "target_tqqq_pct": target_tq,
            "target_sqqq_pct": target_sq,
            "account_value": account_value,
            "tqqq_target_shares": tq_target_shares,
            "sqqq_target_shares": sq_target_shares,
            "tqqq_current_shares": cur_tq,
            "sqqq_current_shares": cur_sq,
            "tqqq_action": tq_action,
            "sqqq_action": sq_action,
            "needs_action": (tq_delta_pct >= MIN_TRADE_PCT) or (sq_delta_pct >= MIN_TRADE_PCT),
        }
    except Exception as e:
        print(f"[alerts] TQQQ signal check failed: {type(e).__name__}: {e}")
        return None

def _run_tqqq_signal_check(state: dict) -> int:
    """Send the daily TQQQ signal + action summary via WhatsApp."""
    cfg = CONFIG
    info = check_tqqq_signal()
    if info is None:
        print("[alerts] TQQQ signal check returned no data.")
        return 0

    key = "TQQQ_SIGNAL"
    if not should_alert(key, "_daily", state, 20):   # 20h cooldown
        return 0

    action_icon = "🎯" if info["needs_action"] else "😴"

    lines = [
        f"{action_icon} TQQQ Signal — {info['signal']}",
        f"Regime: {info['regime']}",
        f"Target: {info['target_tqqq_pct']:.2f}% TQQQ / {info['target_sqqq_pct']:.2f}% SQQQ",
        f"Sleeve value: {info['account_value']:.2f} USD",
        "",
        f"TQQQ: {info['tqqq_action']}",
        f"SQQQ: {info['sqqq_action']}",
    ]
    message = "\n".join(lines)

    if send_whatsapp(cfg["phone"], cfg["api_key"], message):
        state[f"{key}:_daily"] = time.time()
        return 1
    return 0

def run_once() -> int:
    """One full cycle. 52-week checks always run; MOS and BNF run once a day."""
    df = load_positions(CONFIG["csv_path"])
    if df.empty:
        print("[alerts] No positions to check.")
        return 0

    state = load_state()
    sent = 0

    # 52-week checks every cycle
    sent += _run_52w_checks(df, state)

    # MOS check — once per mos_interval_hours
    last_mos = state.get("_last_mos_run", 0)
    hours_since_mos = (time.time() - last_mos) / 3600
    if hours_since_mos >= CONFIG["mos_interval_hours"]:
        print(f"[alerts] Running MOS check (last run {hours_since_mos:.1f}h ago).")
        sent += _run_mos_check(df, state)
        state["_last_mos_run"] = time.time()
    else:
        print(f"[alerts] MOS check skipped ({hours_since_mos:.1f}h < {CONFIG['mos_interval_hours']}h).")

    # BNF check — once per bnf_interval_hours
    last_bnf = state.get("_last_bnf_run", 0)
    hours_since_bnf = (time.time() - last_bnf) / 3600
    if hours_since_bnf >= CONFIG["bnf_interval_hours"]:
        print(f"[alerts] Running BNF check (last run {hours_since_bnf:.1f}h ago).")
        sent += _run_bnf_check(df, state)
        state["_last_bnf_run"] = time.time()
    else:
        print(f"[alerts] BNF check skipped ({hours_since_bnf:.1f}h < {CONFIG['bnf_interval_hours']}h).")
    
    # TQQQ daily signal check — fires only inside the target time window
    last_sig = state.get("_last_tqqq_signal_run", 0)
    hours_since_sig = (time.time() - last_sig) / 3600
    in_window = _in_signal_window()

    if hours_since_sig >= CONFIG["tqqq_signal_interval_hours"] and in_window:
        print(f"[alerts] Running TQQQ signal check (window open, last run {hours_since_sig:.1f}h ago).")
        sent += _run_tqqq_signal_check(state)
        state["_last_tqqq_signal_run"] = time.time()
    elif not in_window:
        print("[alerts] TQQQ signal check skipped (outside 2:45 PM CT window).")
    else:
        print(f"[alerts] TQQQ signal check skipped (already sent {hours_since_sig:.1f}h ago).")

    # TQQQ Swing Trade Alerts
    last_swing = state.get("_last_swing_run", 0)
    hours_since_swing = (time.time() - last_swing) / 3600
    if hours_since_swing >= CONFIG.get("swing_interval_hours", 24):
        print(f"[alerts] Running TQQQ Swing check (last run {hours_since_swing:.1f}h ago).")
        swing_hits = check_tqqq_swing_alerts()
        for hit in swing_hits:
            cooldown_key = f"SWING:{hit['type']}:{hit['pct']}"
            if should_alert(cooldown_key, "TQQQ", state, cooldown_hours=24):
                msg = (
                    f"📈 TQQQ SWING {hit['type']} ALERT\n"
                    f"Price: {hit['current']:.2f} USD\n"
                )
                if hit['type'] == "SELL":
                    msg += f"Target: +{hit['pct']*100:.0f}% from entry ({hit['entry']:.2f} USD)\n"
                    msg += f"Hit: {hit['target']:.2f} USD"
                else:
                    msg += f"Target: -{hit['pct']*100:.0f}% from 20d high ({hit['high']:.2f} USD)\n"
                    msg += f"Hit: {hit['target']:.2f} USD"
                
                if send_whatsapp(CONFIG["phone"], CONFIG["api_key"], msg):
                    mark_alerted(cooldown_key, "TQQQ", state)
                    sent += 1
        state["_last_swing_run"] = time.time()
    else:
        print(f"[alerts] TQQQ Swing check skipped ({hours_since_swing:.1f}h < 24h).")

    save_state(state)
    return sent


def main() -> None:
    print("[alerts] Starting Alert Service... -------------------------------------")
    print(f"[alerts] CSV: {CONFIG['csv_path']}")
    print(f"[alerts] Interval: {CONFIG['check_interval_seconds']}s")
    print(f"[alerts] High threshold: {CONFIG['high_threshold']}")
    print(f"[alerts] Low threshold:  {CONFIG['low_threshold']}")

    while True:
        if not wait_for_network():
            print("[alerts] Network not ready after 180s, skipping this cycle.")
            time.sleep(CONFIG["check_interval_seconds"])
            continue

        try:
            run_once()
        except Exception as e:
            print(f"[alerts] Cycle failed: {type(e).__name__}: {e}")

        time.sleep(CONFIG["check_interval_seconds"])


if __name__ == "__main__":
    main()