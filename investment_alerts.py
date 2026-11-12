"""
Standalone alert service for MoatCheck. Runs as a Windows service (NSSM).

Alerts go to Telegram (primary) with CallMeBot WhatsApp as the fallback. During NYSE
market hours (taken from the exchange calendar, so holidays and early closes are
handled) it sweeps your Fidelity positions every 15 minutes, starting right after
the open:

  1. 52-week high proximity (within 2% of the high)
  2. 52-week low proximity  (within 2% of the low)
  3. MOS buy price hit
  4. BNF mean-reversion setup
  5. TQQQ swing 15% / 20% levels (reference only)

Once per trading day, 20 minutes before the close, it sends the Sleeve strategy
signal (the strategy actually traded). Per-ticker cooldowns stop repeat alerts.
Credentials: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from the environment or from
data/alert_secrets.json (created by telegram_setup.py).

Run manually:   python investment_alerts.py
Install as Windows Service with NSSM (see README).
"""
from __future__ import annotations

import io
import os
import re
import sys

# Force UTF-8 encoding for Windows services so emojis don't crash the cycle
# line_buffering/write_through: under NSSM stdout is a file, and a block-buffered wrapper
# holds log lines in memory until the buffer fills or the process exits cleanly.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True, write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True, write_through=True)
    
import warnings

# Third-party libraries (yfinance, yfinance_cache, pandas, numpy) re-enable their own
# DeprecationWarnings after startup, so a one-time filter does not hold. Silence warnings
# in three places: the filter list, the warn function, and the logging channel.
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.warn = lambda *a, **k: None
warnings.showwarning = lambda *a, **k: None
warnings.formatwarning = lambda *a, **k: ""
import logging
logging.captureWarnings(False)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # "possibly delisted" noise
logging.getLogger("yfinance_cache").setLevel(logging.CRITICAL)

import json
import threading
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

    # Market-hours schedule (America/Chicago; the NYSE calendar handles holidays and early closes)
    "open_delay_minutes": 1,                # first threshold sweep this long after the open
    "intraday_interval_seconds": 900,       # threshold sweep every 15 minutes while the market is open
    "post_close_grace_minutes": 5,          # one last sweep just after the close

    # Per-ticker cooldown (hours) before re-alerting the same signal
    "cooldown_hours": 24,                   # 52-week, MOS, swing
    "bnf_cooldown_hours": 168,              # 7 days per ticker (panic events cluster)

    # Daily Sleeve signal: sent once per trading day, this long before the close
    "tqqq_signal_minutes_before_close": 20,
    "tqqq_signal_retry_seconds": 300,       # retry a failed attempt every 5 minutes

    # Startup report: a full status of every strategy each time the service starts
    "startup_report_min_minutes": 10,       # a restart sooner than this sends a one-line notice (crash-loop guard)
    "startup_timeout_seconds": 900,

    # Service loop
    "tick_seconds": 60,
    "cycle_timeout_seconds": 1500,          # a hung sweep exits so the service manager restarts it
    "signal_timeout_seconds": 300,
    "heartbeat_seconds": 1800,

    "swing_thresholds": [0.15, 0.20],       # TQQQ swing levels (15% and 20%)
    "bargain_cooldown_hours": 168,  # 7 days per tier per ticker
}

# Path to the My Investments cache that holds intrinsic values
SIGNALS_CACHE = Path("data/investments_signals.json")

# Where we persist which alerts we've already sent
STATE_FILE = Path("data/.alert_state.json")

import socket, time

def wait_for_network(hosts=("query1.finance.yahoo.com", "api.telegram.org", "api.callmebot.com"), timeout=180):
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

_TZ_WARNED = False


def _now_ct() -> datetime:
    """Current time in America/Chicago. Falls back to machine time if tzdata is missing."""
    global _TZ_WARNED
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Chicago"))
    except Exception:
        if not _TZ_WARNED:
            print("[alerts] WARNING: America/Chicago unavailable (pip install tzdata); using machine local time.")
            _TZ_WARNED = True
        return datetime.now()


_CAL = None


def _calendar():
    """NYSE trading calendar (holidays, early closes). None if the package is unavailable."""
    global _CAL
    if _CAL is None:
        try:
            import exchange_calendars as xc
            _CAL = xc.get_calendar("XNYS")
        except Exception as e:
            print(f"[alerts] WARNING: exchange_calendars unavailable ({type(e).__name__}); using Mon-Fri 8:30-15:00 CT.")
            _CAL = False
    return _CAL or None


def market_session(now_ct: datetime | None = None):
    """(open, close) datetimes in Central Time if now_ct's date is a trading day, else None."""
    now_ct = now_ct or _now_ct()
    cal = _calendar()
    if cal is not None:
        try:
            day = pd.Timestamp(now_ct.date())
            if not cal.is_session(day):
                return None
            o = cal.session_open(day).tz_convert("America/Chicago").to_pydatetime()
            c = cal.session_close(day).tz_convert("America/Chicago").to_pydatetime()
            if now_ct.tzinfo is None:
                o, c = o.replace(tzinfo=None), c.replace(tzinfo=None)
            return o, c
        except Exception:
            pass   # date outside the calendar's range -> weekday fallback below
    if now_ct.weekday() >= 5:
        return None
    return (now_ct.replace(hour=8, minute=30, second=0, microsecond=0),
            now_ct.replace(hour=15, minute=0, second=0, microsecond=0))


def is_market_open(now_ct: datetime | None = None) -> bool:
    now_ct = now_ct or _now_ct()
    sess = market_session(now_ct)
    return bool(sess) and sess[0] <= now_ct < sess[1]


def _in_signal_window(now_ct: datetime | None = None) -> bool:
    """True from N minutes before the close until the close, on trading days (early closes included)."""
    now_ct = now_ct or _now_ct()
    sess = market_session(now_ct)
    if not sess:
        return False
    return sess[1] - timedelta(minutes=CONFIG["tqqq_signal_minutes_before_close"]) <= now_ct < sess[1]


def _intraday_window(now_ct: datetime | None = None) -> bool:
    """True from open + delay until close + grace on trading days."""
    now_ct = now_ct or _now_ct()
    sess = market_session(now_ct)
    if not sess:
        return False
    return (sess[0] + timedelta(minutes=CONFIG["open_delay_minutes"])
            <= now_ct < sess[1] + timedelta(minutes=CONFIG["post_close_grace_minutes"]))


_last_intraday = 0.0


def intraday_due(now_ct: datetime | None = None) -> bool:
    return _intraday_window(now_ct) and time.time() - _last_intraday >= CONFIG["intraday_interval_seconds"]


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

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        # A stock held in several accounts is one position for alerting: one row per ticker, so a
        # hit is scanned, listed and sent once instead of once per account.
        return out.drop_duplicates(subset="Symbol", keep="first").reset_index(drop=True)
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

_RANGE_CACHE: dict = {}


def _price_range(sym: str):
    """(last_price, 52w_high, 52w_low) from daily history, or None if missing or stale.

    Uses price history, not Ticker.info: info's currentPrice can lag by days (SPY showed the
    close from two sessions earlier), which produced false 52-week alerts. Tickers whose last
    bar is more than 5 days old (delisted) are skipped.
    """
    hit = _RANGE_CACHE.get(sym)
    if hit and time.time() - hit[0] < 120:
        return hit[1]
    from data_provider import get_ticker
    result = None
    h = get_ticker(sym).history(period="1y")
    if h is not None and not h.empty:
        h = h.dropna(subset=["Close"])
    if h is not None and not h.empty:
        last = pd.Timestamp(h.index[-1])
        if last.tzinfo is not None:
            last = last.tz_localize(None)
        if (pd.Timestamp.now() - last).days <= 5:
            result = (float(h["Close"].iloc[-1]), float(h["High"].max()), float(h["Low"].min()))
    _RANGE_CACHE[sym] = (time.time(), result)
    return result


def check_52w_highs(df: pd.DataFrame, threshold: float) -> list[dict]:
    """Positions within `threshold` of their 52-week high."""
    hits = []
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip()
        if not sym:
            continue
        try:
            r = _price_range(sym)
            if r is None:
                continue
            price, high, _low = r
            if high <= 0.1 or price <= 0.1:
                continue
            pct = price / high
            if pct >= threshold:
                hits.append({"symbol": sym, "price": price, "high": high, "pct": pct})
        except Exception:
            continue
    return hits


def check_52w_lows(df: pd.DataFrame, threshold: float) -> list[dict]:
    """Positions within `threshold` of their 52-week low."""
    hits = []
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip()
        if not sym:
            continue
        try:
            r = _price_range(sym)
            if r is None:
                continue
            price, _high, low = r
            if low <= 0.1 or price <= 0.1:
                continue
            pct = (price - low) / low
            if pct <= threshold:
                hits.append({"symbol": sym, "price": price, "low": low, "pct_above": pct})
        except Exception:
            continue
    return hits


# ============================================================
# BUYING OPPORTUNITIES (BARGAIN BUYS) CHECK
# ============================================================

def check_bargain_hits(df: pd.DataFrame) -> list[dict]:
    """Scan positions for stocks trading at least 15% below their Intrinsic Value.
    Uses the cached signals from My Investments — no re-fetch needed."""
    print(f"[alerts] Starting Bargain Buy check for {len(df)} positions.")

    if not SIGNALS_CACHE.exists():
        print(f"[alerts] No signals cache at {SIGNALS_CACHE}; skipping Bargain check.")
        return []

    try:
        signals = json.loads(SIGNALS_CACHE.read_text())
    except Exception as e:
        print(f"[alerts] Could not read signals cache: {e}")
        return []

    hits = []
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip().upper()
        if not sym or sym not in signals:
            continue

        sig = signals[sym] or {}
        intrinsic = sig.get("intrinsic_value")
        if not intrinsic or intrinsic <= 0:
            continue

        rng = _price_range(sym)
        if rng is None:
            continue
        price = rng[0]

        discount = (intrinsic - price) / intrinsic
        if discount < 0.15:
            continue

        if discount >= 0.50:
            tier = "BARGAIN BUY"
        elif discount >= 0.30:
            tier = "DEEP DISCOUNT"
        else:
            tier = "BUY ZONE"

        hits.append({
            "symbol": sym,
            "price": price,
            "intrinsic": intrinsic,
            "discount": discount,
            "tier": tier,
        })

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
# MESSAGING - Telegram (primary), CallMeBot WhatsApp (fallback)
# ============================================================

def _secret(name: str, default: str = "") -> str:
    """Read a credential from the environment, then from data/alert_secrets.json."""
    v = os.environ.get(name)
    if v and v.strip():
        return v.strip()
    try:
        f = Path("data/alert_secrets.json")
        if f.exists():
            return str(json.loads(f.read_text(encoding="utf-8")).get(name, default)).strip()
    except Exception:
        pass
    return default


def telegram_configured() -> bool:
    return bool(_secret("TELEGRAM_BOT_TOKEN") and _secret("TELEGRAM_CHAT_ID"))


def send_telegram(text: str) -> bool:
    token, chat = _secret("TELEGRAM_BOT_TOKEN"), _secret("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": text, "disable_web_page_preview": "true"},
            timeout=30,
        )
        try:
            body = r.json()
        except Exception:
            body = {}
        if r.status_code == 200 and body.get("ok"):
            print("[alerts] Telegram sent")
            return True
        print(f"[alerts] Telegram failed: HTTP {r.status_code} {str(body.get('description', r.text))[:200]}")
    except Exception as e:
        print(f"[alerts] Telegram send failed: {type(e).__name__}: {str(e).replace(token, '***')[:200]}")
    return False


def send_whatsapp(phone: str, api_key: str, message: str) -> bool:
    """CallMeBot WhatsApp. It answers HTTP 200 even when it refuses ('0 messages left'), so read the body."""
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        return False
    print(f"[alerts] Sending WhatsApp message to: {phone}")
    try:
        r = requests.get("https://api.callmebot.com/whatsapp.php",
                         params={"phone": phone, "text": message, "apikey": api_key}, timeout=30)
        reply = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text)).strip()
        low = reply.lower()
        if r.status_code == 200 and not any(m in low for m in ("message not sent", "0 messages left", "invalid", "error")):
            print(f"[alerts] WhatsApp accepted by CallMeBot: {reply[:80]}")
            return True
        print(f"[alerts] CallMeBot refused (HTTP {r.status_code}): {reply[:200]}")
        return False
    except Exception as e:
        print(f"[alerts] WhatsApp send failed: {type(e).__name__}: {str(e).replace(api_key, '***')[:200]}")
        return False


def _chunks(text: str, limit: int = 3500) -> list[str]:
    """Split on line boundaries to stay under Telegram's 4096-character limit."""
    out, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + len(line) + 1 > limit:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        out.append(cur)
    return out


def send_alert(message: str) -> bool:
    """Deliver an alert. Telegram first; CallMeBot only if Telegram is not set up or failed."""
    print(f"[alerts] Alert:\n{message}")
    all_ok = True
    for part in _chunks(message):
        delivered = telegram_configured() and send_telegram(part)
        if not delivered:
            delivered = send_whatsapp(CONFIG["phone"], CONFIG["api_key"], part)
        all_ok = all_ok and delivered
    return all_ok


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

def _send_group(title: str, entries: list, state: dict, cooldown_hours: float) -> int:
    """Send one message for all new hits. entries = [(kind, symbol, line)]. Returns hits delivered."""
    fresh = [(k, sym, line) for k, sym, line in entries if should_alert(k, sym, state, cooldown_hours)]
    if not fresh:
        return 0
    message = title + "\n" + "\n".join(line for _, _, line in fresh)
    if not send_alert(message):
        return 0
    for k, sym, _ in fresh:
        mark_alerted(k, sym, state)
    return len(fresh)


def _entries_52w_highs(df: pd.DataFrame) -> list:
    return [("HIGH", h["symbol"],
             f"{h['symbol']}: {h['price']:.2f} USD ({h['pct']*100:.1f}% of 52w high {h['high']:.2f})")
            for h in check_52w_highs(df, CONFIG["high_threshold"])]


def _entries_52w_lows(df: pd.DataFrame) -> list:
    return [("LOW", h["symbol"],
             f"{h['symbol']}: {h['price']:.2f} USD ({h['pct_above']*100:+.1f}% above 52w low {h['low']:.2f})")
            for h in check_52w_lows(df, CONFIG["low_threshold"])]


def _entries_bargain(df: pd.DataFrame) -> list:
    return [(f"BARGAIN:{h['tier']}",  # <-- tier in the key
             h["symbol"],
             f"{h['symbol']} [{h['tier']}]: {h['price']:.2f} USD is {h['discount']*100:.0f}% below Intrinsic Value "
             f"({h['intrinsic']:.2f} USD)")
            for h in check_bargain_hits(df)]


def _entries_bnf(df: pd.DataFrame) -> list:
    return [("BNF", h["symbol"],
             f"{h['symbol']}: {h['price']:.2f} USD, {h['dislocation']:+.1f}% vs MA25 ({h['ma_25']:.2f}), RSI {h['rsi']:.1f}")
            for h in check_bnf_setups(df)]


def _entries_swing() -> list:
    entries = []
    for h in check_tqqq_swing_alerts():
        kind = f"SWING:{h['type']}:{h['pct']}"
        if h["type"] == "SELL":
            line = (f"SELL level: TQQQ {h['current']:.2f} USD is +{h['pct']*100:.0f}% over entry "
                    f"{h['entry']:.2f} (target {h['target']:.2f})")
        else:
            line = (f"BUY level: TQQQ {h['current']:.2f} USD is -{h['pct']*100:.0f}% from the 20d high "
                    f"{h['high']:.2f} (target {h['target']:.2f})")
        entries.append((kind, "TQQQ", line))
    return entries


def _run_52w_checks(df: pd.DataFrame, state: dict) -> int:
    """52-week high and low checks. Returns number of hits delivered."""
    print("[alerts] Running 52-week high and low checks.")
    sent = _send_group("🚨 52-Week High Alert", _entries_52w_highs(df), state, CONFIG["cooldown_hours"])
    sent += _send_group("⚠️ 52-Week Low Alert", _entries_52w_lows(df), state, CONFIG["cooldown_hours"])
    return sent

def _run_bnf_check(df: pd.DataFrame, state: dict) -> int:
    """BNF reversal scan. Returns number of hits delivered."""
    return _send_group("🔄 BNF Reversal Setup", _entries_bnf(df), state, CONFIG["bnf_cooldown_hours"])


def _run_swing_check(state: dict) -> int:
    """TQQQ swing levels (reference only; the traded signal is the Sleeve). Returns hits delivered."""
    return _send_group("📈 TQQQ Swing Levels (reference only; the Sleeve is the traded signal)",
                       _entries_swing(), state, CONFIG["cooldown_hours"])


def check_tqqq_signal() -> dict | None:
    """Run the live TQQQ/SQQQ strategy and compute the daily action."""
    print(f"[alerts] Starting TQQQ/SQQQ live strategy check.")
    try:
        from sleeves_strategy import get_sleeve_target
        from trade_manager import load_position
        from data_provider import get_ticker

        # Sleeve strategy (the one actually traded). Never fall back to the dial.
        r = get_sleeve_target()
        if not r or r.get("error") or not r.get("as_of"):
            print(f"[alerts] Sleeve target unavailable: {(r or {}).get('error', 'no data')}")
            return None

        target_tq = float(r["tqqq_pct"])
        target_sq = float(r["sqqq_pct"])
        target_cash = float(r["cash_pct"])

        pos = load_position()
        cur_tq = float(pos.get("tqqq_shares", 0.0))
        cur_sq = float(pos.get("sqqq_shares", 0.0))
        cur_cash = float(pos.get("cash_balance", 0.0))  # NEW: Read cash balance

        try:
            tq_px = float(get_ticker("TQQQ").history(period="1d")["Close"].iloc[-1])
            sq_px = float(get_ticker("SQQQ").history(period="1d")["Close"].iloc[-1])
        except Exception:
            return None

        # FIX: Include cash in the total account value
        account_value = (cur_tq * tq_px) + (cur_sq * sq_px) + cur_cash
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
            "signal": "SLEEVE",
            "as_of": r["as_of"],
            "regime": f"{r['regime']} (NDX as of {r['as_of']})",
            "target_tqqq_pct": target_tq,
            "target_sqqq_pct": target_sq,
            "target_cash_pct": target_cash,
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

def _sleeve_lines(info: dict) -> list:
    """Message lines for the Sleeve signal (used by the daily signal and the startup report)."""
    action_icon = "🎯" if info["needs_action"] else "😴"
    return [
        f"{action_icon} TQQQ Sleeve Signal",
        f"Regime: {info['regime']}",
        f"Total Sleeve: {info['account_value']:.2f} USD",
        "",
        "🎯 TARGETS:",
        f"TQQQ: {info['target_tqqq_pct']:.1f}% | SQQQ: {info['target_sqqq_pct']:.1f}% | Cash: {info['target_cash_pct']:.1f}%",
        "",
        "⚡ ACTIONS:",
        f"TQQQ: {info['tqqq_action']}",
        f"SQQQ: {info['sqqq_action']}",
    ]


def _run_tqqq_signal_check(today) -> int:
    """Send the daily TQQQ sleeve signal via WhatsApp. Returns 1 only if it was sent."""
    cfg = CONFIG
    info = check_tqqq_signal()
    if info is None:
        print("[alerts] TQQQ signal check returned no data.")
        return 0
    if info["as_of"] != today:
        print(f"[alerts] Sleeve data is as of {info['as_of']}, not {today} (holiday or data lag); not sending.")
        return 0

    message = "\n".join(_sleeve_lines(info))

    if send_alert(message):
        return 1
    return 0


_last_signal_attempt = 0.0


def maybe_send_tqqq_signal() -> int:
    """Called every tick. Sends the sleeve signal once per weekday inside the window.

    The 'sent today' date is saved only after WhatsApp accepts the message, so a data
    or network failure is retried every tqqq_signal_retry_seconds until the window closes.
    """
    global _last_signal_attempt
    now_ct = _now_ct()
    if not _in_signal_window(now_ct):
        return 0
    today = now_ct.date()
    state = load_state()
    if state.get("_tqqq_signal_date") == today.isoformat():
        return 0
    if time.time() - _last_signal_attempt < CONFIG["tqqq_signal_retry_seconds"]:
        return 0
    _last_signal_attempt = time.time()
    print(f"[alerts] Running TQQQ sleeve signal (CT {now_ct:%H:%M}).")
    sent = _run_tqqq_signal_check(today)
    if sent:
        state = load_state()
        state["_tqqq_signal_date"] = today.isoformat()
        save_state(state)
    return sent

def _market_status_line(now_ct: datetime) -> str:
    sess = market_session(now_ct)
    if not sess:
        return f"Market is closed today ({now_ct:%A}); figures are from the last close."
    if is_market_open(now_ct):
        return f"Market is OPEN until {sess[1]:%H:%M} CT."
    if now_ct < sess[0]:
        return f"Market opens at {sess[0]:%H:%M} CT; figures are from the last close."
    return "Market is closed for the day; figures are from today's close."


def build_startup_report(df: pd.DataFrame, now_ct: datetime) -> tuple[str, list, bool]:
    """Full status of every strategy, ignoring cooldowns. Returns (message, [(kind, symbol)] hits, sleeve_is_today).

    Each check is isolated, so one failing check shows as 'check failed' instead of losing the report.
    """
    lines = [
        f"🟢 MoatCheck alert service started {now_ct:%Y-%m-%d %H:%M} CT",
        _market_status_line(now_ct),
        "Full status of every strategy now; regular alerts follow their timers.",
        "",
    ]
    hits: list = []
    sleeve_today = False

    # --- Sleeve (the traded strategy)
    lines.append("━━ TQQQ SLEEVE (traded signal) ━━")
    try:
        info = check_tqqq_signal()
        if info is None:
            lines.append("Sleeve target unavailable (data or position file problem); see the service log.")
        else:
            sl = _sleeve_lines(info)
            lines += sl[1:]                        # the section title above replaces the first line
            sleeve_today = info.get("as_of") == now_ct.date()
            if not sleeve_today:
                lines.append(f"(NDX data as of {info.get('as_of')})")
    except Exception as e:
        lines.append(f"Sleeve check failed: {type(e).__name__}")
    lines.append("")

    sections = [
        ("🚨 52-WEEK HIGHS (within 2%)", lambda: _entries_52w_highs(df)),
        ("⚠️ 52-WEEK LOWS (within 2%)", lambda: _entries_52w_lows(df)),
        ("🎯 BUYING OPPORTUNITIES (Bargain Buys)", lambda: _entries_bargain(df)),
        ("🔄 BNF REVERSAL SETUPS", lambda: _entries_bnf(df)),
        ("📈 TQQQ SWING LEVELS (reference only)", lambda: _entries_swing()),
    ]
    for title, fn in sections:
        try:
            if df.empty and title != sections[-1][0]:
                lines += [f"━━ {title} ━━", "No positions file loaded.", ""]
                continue
            entries = fn()
            lines.append(f"━━ {title}: {len(entries)} ━━")
            lines += [e[2] for e in entries] or ["None right now."]
            hits += [(e[0], e[1]) for e in entries]
        except Exception as e:
            lines.append(f"━━ {title} ━━")
            lines.append(f"Check failed: {type(e).__name__}")
        lines.append("")

    sess = market_session(now_ct)
    lines.append(f"Next: sweeps every {CONFIG['intraday_interval_seconds'] // 60} min while the market is open"
                 f"{' (' + format(sess[0], '%H:%M') + '-' + format(sess[1], '%H:%M') + ' CT today)' if sess else ''}; "
                 f"Sleeve signal {CONFIG['tqqq_signal_minutes_before_close']} min before the close.")
    return "\n".join(lines), hits, sleeve_today


def run_startup_report() -> int:
    """Send the full startup status. Returns 1 if the report was sent, 0 if not.

    Hits it lists are recorded as already alerted, so the first regular sweep does not repeat them.
    A restart within startup_report_min_minutes of the last report (a crash loop) sends one line instead.
    """
    now = _now_ct()
    state = load_state()
    last = float(state.get("_last_startup_report", 0) or 0)
    if time.time() - last < CONFIG["startup_report_min_minutes"] * 60:
        print("[alerts] Startup report skipped: one was sent less than "
              f"{CONFIG['startup_report_min_minutes']} minutes ago (restart loop?).")
        send_alert(f"🔄 MoatCheck alert service restarted {now:%Y-%m-%d %H:%M} CT "
                   f"(full report was sent {int((time.time() - last) / 60)} min ago).")
        return 0

    df = load_positions(CONFIG["csv_path"])
    print(f"[alerts] Building startup report ({len(df)} positions).")
    _RANGE_CACHE.clear()
    message, hits, sleeve_today = build_startup_report(df, now)
    if not send_alert(message):
        print("[alerts] Startup report could not be delivered.")
        return 0
    state = load_state()
    for kind, sym in hits:
        mark_alerted(kind, sym, state)
    state["_last_startup_report"] = time.time()
    if sleeve_today and _in_signal_window(now):
        state["_tqqq_signal_date"] = now.date().isoformat()     # the report carried today's Sleeve signal
    save_state(state)
    print(f"[alerts] Startup report sent ({len(hits)} threshold hit(s) recorded).")
    return 1


def run_intraday_cycle() -> int:
    """One threshold sweep: 52-week highs/lows, MOS, BNF, TQQQ swing. Returns alerts delivered."""
    df = load_positions(CONFIG["csv_path"])
    if df.empty:
        print("[alerts] No positions to check.")
        return 0
    _RANGE_CACHE.clear()
    state = load_state()
    sent = 0
    for label, fn in (("52-week", lambda: _run_52w_checks(df, state)),
                      ("Bargain", lambda: _send_group("🎯 Buying Opportunities", _entries_bargain(df), state, CONFIG["bargain_cooldown_hours"])),
                      ("BNF", lambda: _run_bnf_check(df, state)),
                      ("TQQQ swing", lambda: _run_swing_check(state))):
        try:
            sent += fn()
        except Exception as e:
            print(f"[alerts] {label} check failed: {type(e).__name__}: {e}")
        save_state(state)
    print(f"[alerts] Sweep done: {sent} hit(s) delivered.")
    return sent


run_once = run_intraday_cycle   # old name, kept for scripts that call it


def _run_with_watchdog(fn, timeout: int, label: str):
    """Run fn in a thread. If it hangs past `timeout`, exit so the service manager restarts us."""
    box = {}

    def target():
        try:
            box["result"] = fn()
        except Exception as e:
            print(f"[alerts] {label} failed: {type(e).__name__}: {e}")

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        print(f"[alerts] {label} hung for more than {timeout}s; exiting so the service restarts.")
        sys.stdout.flush()
        os._exit(1)
    return box.get("result")


def main() -> None:
    global _last_intraday
    print("[alerts] ================================================================")
    print("[alerts] Starting Alert Service ---------------------------------")
    print(f"[alerts] Started at {_now_ct():%Y-%m-%d %H:%M:%S} CT, pid {os.getpid()}")
    print(f"[alerts] CSV: {CONFIG['csv_path']}")
    print(f"[alerts] Channels: Telegram {'configured' if telegram_configured() else 'NOT configured'}; "
          f"CallMeBot WhatsApp fallback {'configured' if CONFIG.get('api_key') else 'off'}")
    print(f"[alerts] Threshold sweep every {CONFIG['intraday_interval_seconds']}s while the market is open "
          f"(first sweep {CONFIG['open_delay_minutes']} min after the open); tick {CONFIG['tick_seconds']}s")
    print(f"[alerts] Daily Sleeve signal: {CONFIG['tqqq_signal_minutes_before_close']} min before the close, once per trading day")
    print(f"[alerts] High threshold: {CONFIG['high_threshold']}  Low threshold: {CONFIG['low_threshold']}")
    # Full status of every strategy, then the normal timers take over.
    if wait_for_network(timeout=180):
        sent_full = _run_with_watchdog(run_startup_report, CONFIG["startup_timeout_seconds"], "Startup report")
        if sent_full:
            _last_intraday = time.time()        # the report covered a sweep; the next one follows the timer
        elif sent_full is None:                 # the report crashed: still say the service is up
            send_alert(f"🟢 MoatCheck alert service started {_now_ct():%Y-%m-%d %H:%M} CT "
                       "(the full startup report failed; see the service log).")
    else:
        print("[alerts] Network not ready after 180s; startup report skipped.")

    last_beat = 0.0
    while True:
        try:
            now = _now_ct()
            if intraday_due(now):
                if wait_for_network(timeout=60):
                    _run_with_watchdog(run_intraday_cycle, CONFIG["cycle_timeout_seconds"], "Sweep")
                else:
                    print("[alerts] Network not ready after 60s, skipping this sweep.")
                _last_intraday = time.time()

            _run_with_watchdog(maybe_send_tqqq_signal, CONFIG["signal_timeout_seconds"], "Daily signal")

            if time.time() - last_beat >= CONFIG["heartbeat_seconds"]:
                st = load_state()
                sess = market_session(now)
                mkt = (f"market {'OPEN' if is_market_open(now) else 'closed'} "
                       f"({sess[0]:%H:%M}-{sess[1]:%H:%M} CT)") if sess else "market closed today"
                print(f"[alerts] heartbeat: alive, CT {now:%Y-%m-%d %H:%M}, {mkt}, "
                      f"last daily signal {st.get('_tqqq_signal_date', 'never')}")
                last_beat = time.time()
        except Exception as e:
            print(f"[alerts] Tick failed: {type(e).__name__}: {e}")
        time.sleep(CONFIG["tick_seconds"])


if __name__ == "__main__":
    main()