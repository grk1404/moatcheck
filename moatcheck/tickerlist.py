from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

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
def get_us_stock_universe() -> pd.DataFrame:
    """
    Download the SEC company ticker/exchange list and return
    a filtered universe of US-listed securities.

    Cached for 24 hours.
    """

    response = requests.get(
        SEC_TICKERS_URL,
        headers=SEC_HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    # SEC file structure:
    # {
    #     "fields": ["cik", "name", "ticker", "exchange"],
    #     "data": [...]
    # }

    df = pd.DataFrame(
        data["data"],
        columns=data["fields"],
    )

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

    df = df[
        df["ticker"].str.match(
            r"^[A-Z0-9]+(?:-[A-Z0-9]+)?$",
            na=False,
        )
    ].copy()

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

    return df


# ============================================================
# GET ONLY TICKER SYMBOLS
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_us_stock_tickers() -> list[str]:
    """
    Return a simple list of ticker symbols.
    """

    df = get_us_stock_universe()

    return df["ticker"].tolist()
