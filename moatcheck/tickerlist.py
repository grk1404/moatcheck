from __future__ import annotations

import json
import time
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
import re

# Disk cache for the universe — survives Streamlit restarts
UNIVERSE_CACHE = Path("data/universe.json")
UNIVERSE_TTL = 7 * 24 * 3600  # 7 days
UNIVERSE_VERSION = 3   # bump whenever the filter logic changes


def _load_disk_cache() -> pd.DataFrame | None:
    if not UNIVERSE_CACHE.exists():
        return None
    if time.time() - UNIVERSE_CACHE.stat().st_mtime > UNIVERSE_TTL:
        return None
    try:
        payload = json.loads(UNIVERSE_CACHE.read_text())
        # New format is {"version": N, "rows": [...]}
        # Old format is just a list — treat as version 1 and reject if version differs
        if isinstance(payload, dict):
            if payload.get("version") != UNIVERSE_VERSION:
                return None
            return pd.DataFrame(payload.get("rows", []))
        return None   # old format — reject
    except (json.JSONDecodeError, OSError):
        return None


def _save_disk_cache(df: pd.DataFrame) -> None:
    UNIVERSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": UNIVERSE_VERSION, "rows": df.to_dict(orient="records")}
    UNIVERSE_CACHE.write_text(json.dumps(payload))

# ============================================================
# SEC CONFIGURATION
# ============================================================

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

# IMPORTANT:
# SEC asks automated users to identify themselves with a User-Agent.
# Replace this with your own application/contact information.
SEC_HEADERS = {
    "User-Agent": "MoatCheck Stock Screener contact@example.com"
}


# ============================================================
# GET US STOCK UNIVERSE
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_us_stock_universe(force_refresh: bool = False) -> pd.DataFrame:
    """
    Download the SEC company ticker/exchange list and return
    a filtered universe of US-listed securities.

    Cached in Streamlit memory for 24h and on disk for 7 days.
    Set force_refresh=True to bypass both.
    """
    if not force_refresh:
        cached = _load_disk_cache()
        if cached is not None:
            return cached

    response = requests.get(
        SEC_TICKERS_URL,
        headers=SEC_HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["data"], columns=data["fields"])

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["name"] = (
        df["name"]
        .astype(str)
        .str.strip()
    )

    df["exchange"] = (
        df["exchange"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Keep major US exchanges
    # --------------------------------------------------------

    allowed_exchanges = {
        "Nasdaq",
        "NYSE",
        "NYSE American",
        "NYSE Arca",
        "Cboe BZX",
        "Cboe BYX",
        "Cboe EDGA",
        "Cboe EDGX",
    }

    df = df[
        df["exchange"].isin(allowed_exchanges)
    ].copy()

    # --------------------------------------------------------
    # Remove obvious non-standard ticker formats
    #
    # Examples:
    #   BRK-B -> keep
    #   AAPL -> keep
    #
    # We allow letters, numbers and hyphen.
    # --------------------------------------------------------

    # Exclude warrants, units, rights, preferreds, and other non-common
    # securities. These don't file 10-Ks and always fail the EDGAR fetch.
    _NON_COMMON_SUFFIXES = {
        "WT", "WS", "W", "U", "UN", "R", "RT",
        "P", "PR", "PA", "PB", "PC", "PD",
        "CL", "CV", "CVT", "CONV",
    }

    def _is_common_stock(t: str) -> bool:
        if not isinstance(t, str) or not t:
            return False
        # Plain ticker: letters only, 1–5 chars
        if "-" not in t:
            return bool(re.match(r"^[A-Z]{1,5}$", t))
        base, _, suffix = t.partition("-")
        if suffix in _NON_COMMON_SUFFIXES:
            return False
        # Allow hyphenated share classes (BRK-A, BRK-B, BF-A, BF-B)
        return bool(re.match(r"^[A-Z]{1,5}$", base)) and bool(re.match(r"^[A-Z]{1,2}$", suffix))

    df = df[df["ticker"].apply(_is_common_stock)].copy()

    # --------------------------------------------------------
    # Remove duplicate tickers
    # --------------------------------------------------------

    df = (
        df.sort_values("ticker")
        .drop_duplicates(
            subset=["ticker"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    _save_disk_cache(df)
    return df
    


# ============================================================
# GET ONLY TICKER SYMBOLS
# ============================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_us_stock_tickers(force_refresh: bool = False) -> list[str]:
    df = get_us_stock_universe(force_refresh=force_refresh)
    return df["ticker"].tolist()

# ============================================================
# NSE (INDIA) TICKER UNIVERSE
# ============================================================

NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_MASTER_URL = ("https://huggingface.co/datasets/tickertruth/nse-india-security-master/" "resolve/main/data/nse_security_master.csv")

# NSE requires a browser-like User-Agent; they block python-requests default.
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MoatCheck/1.0; contact@moatcheck.com)",
    "Accept": "text/csv,application/csv,*/*",
}


@st.cache_data(ttl=UNIVERSE_TTL, show_spinner=False)
def get_indian_stock_tickers() -> list[str]:
    """
    Return NSE tickers with the .NS suffix (yfinance format).

    Uses the TickerTruth security master on Hugging Face — a clean,
    normalized CSV of NSE-listed equities. Falls back to an empty list
    on failure so the screener runs US-only instead of crashing.
    """
    import io
    try:
        r = requests.get(NSE_MASTER_URL, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print(f"NSE security master fetch failed: {e}")
        return []

    if "nse_symbol" not in df.columns:
        return []

    symbols = (
        df["nse_symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    symbols = symbols[symbols.str.match(r"^[A-Z0-9&\-]{1,20}$", na=False)]
    return [f"{s}.NS" for s in symbols.tolist()]


@st.cache_data(ttl=UNIVERSE_TTL, show_spinner=False)
def get_combined_tickers(include_us: bool = True, include_india: bool = False) -> list[str]:
    """Return a merged ticker list for the screener."""
    tickers: list[str] = []
    if include_us:
        tickers.extend(get_us_stock_tickers())
    if include_india:
        tickers.extend(get_indian_stock_tickers())
    return tickers