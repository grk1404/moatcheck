from __future__ import annotations
import streamlit as st

import shutil
import os
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
from data_provider import get_ticker
from investments import _metric_html, render_my_investments
from data_quality import check_data_quality

from trade_manager import (
    compute_target_shares,
    compute_rebalance,
    log_trade,
    load_trades,
    load_position,
    save_position,
    load_capital,
    save_capital,
    net_capital,
    compute_running_pnl,
    compute_realized_pnl,
)

warnings.filterwarnings('ignore')
# Import the technical indicator analyzer
from technical_indicators import TechnicalIndicatorAnalyzer

# Global constants
MIN_TRADE_PCT = 2.0   # 2% of account value  - TQQQ/SQQQ Strategy Configuration


def clear_yfinance_cache(ticker: str | None = None) -> bool:
    """Clear yfinance_cache data. 
    
    If `ticker` is provided, it only deletes the cache for that specific ticker.
    If `ticker` is None, it nukes the entire cache folder.
    Returns True if something was deleted, False otherwise.
    """
    # Common locations for yfinance_cache on Windows and Mac/Linux
    potential_dirs = [
        Path(os.path.expanduser("~")) / "AppData" / "Local" / "py-yfinance-cache",
        Path(os.path.expanduser("~")) / ".cache" / "py-yfinance-cache",
        Path("py-yfinance-cache"), # Local project folder
    ]
    
    cleared = False
    for base_dir in potential_dirs:
        if not base_dir.exists():
            continue
            
        if ticker:
            # Clean just one ticker (e.g. "AAPL")
            ticker_dir = base_dir / ticker.upper()
            if ticker_dir.exists():
                shutil.rmtree(ticker_dir)
                cleared = True
        else:
            # Clean the entire cache
            shutil.rmtree(base_dir)
            cleared = True
            
    return cleared

def _first_n_sentences(text: str, n: int = 2) -> str:
    """Return the first N sentences from text, or the whole text if fewer."""
    if not text:
        return ""
    import re
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(parts[:n])

def clear_analyzer_cache():
    """Clear all cached data and reset analyzer state."""
    st.cache_data.clear()
    if "last_ticker" in st.session_state:
        st.session_state["last_ticker"] = None
    if "valuation_ticker" in st.session_state:
        st.session_state["valuation_ticker"] = None
    if "valuation_growth_result" in st.session_state:
        st.session_state["valuation_growth_result"] = None
    if "analyzing" in st.session_state:
        st.session_state["analyzing"] = False

from moatcheck import (
    Big5Result,
    Financials,
    compute_big5,
    dcf_two_stage,
    debt_to_fcf,
    fetch,
    graham_formula,
    graham_number,
    payback_time,
    peg_fair_value,
    peter_lynch_fair,
    value_price,
)

from moatcheck.big5 import WINDOWS, PASS_THRESHOLD
from moatcheck.fetcher import FetchError
from moatcheck.intrinsic import MethodResult
from screener import render_stock_screener

st.set_page_config(page_title="MoatCheck", page_icon=None, layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fetch(symbol: str) -> Financials:
    return fetch(symbol)


_COLOR_HEX = {
    "bargain": "#00E676",
    "green": "#4CAF50",
    "orange": "#FFA726",
    "red": "#EF5350",
    "gray": "#9AA0A6",
}

# Common / likely ticker typos that should be corrected before fetching.
_COMMON_TICKER_FIXES = {
    "APPL": "AAPL",
    "GOOL": "GOOG",
    "MSFTT": "MSFT",
    "AMZNN": "AMZN",
}

def _check_data_quality(prices: pd.DataFrame, ticker: str) -> None:
    """Perform sanity checks on the backtest data before running strategies."""
    if prices.empty:
        return

    st.markdown("### 🕵️ Data Quality Check")
    
    # 1. Date Range & Length
    start_date = prices.index[0].strftime("%Y-%m-%d")
    end_date = prices.index[-1].strftime("%Y-%m-%d")
    total_rows = len(prices)
    expected_years = (prices.index[-1] - prices.index[0]).days / 365.25
    
    # 2. Check for missing trading days
    # Generate a business day range (Mon-Fri) and see how many are missing
    bdays = pd.bdate_range(start=prices.index[0], end=prices.index[-1])
    missing_days = len(bdays) - total_rows
    # Rough estimate: ~9 market holidays per year
    expected_missing = expected_years * 9
    missing_pct = (missing_days / len(bdays)) * 100 if len(bdays) > 0 else 0
    
    # 3. Check for bad prices (0 or negative)
    bad_price_rows = prices[(prices["TQQQ_close"] <= 0) | (prices["TQQQ_open"] <= 0)]
    bad_price_count = len(bad_price_rows)
    
    # 4. Check for massive single-day jumps (possible unadjusted splits)
    # A 50% jump in a single day for an ETF is extremely rare and usually a split
    daily_returns = prices["TQQQ_close"].pct_change()
    jumps = daily_returns[daily_returns.abs() > 0.50]
    jump_count = len(jumps)
    
    # --- Display the Results ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trading Days", f"{total_rows:,}", help=f"From {start_date} to {end_date}")
    c2.metric("Missing Business Days", f"{missing_days:,}", delta=f"{missing_pct:.1f}%", delta_color="inverse" if missing_days > expected_missing else "normal")
    
    if bad_price_count > 0:
        c3.metric("Zero/Negative Prices", f"{bad_price_count:,}", delta="FAIL", delta_color="inverse")
    else:
        c3.metric("Zero/Negative Prices", "0", delta="PASS", delta_color="normal")
        
    if jump_count > 0:
        c4.metric("Massive Jumps (>50%)", f"{jump_count:,}", delta="WARN", delta_color="inverse")
    else:
        c4.metric("Massive Jumps (>50%)", "0", delta="PASS", delta_color="normal")

    # Warnings and Details
    if bad_price_count > 0:
        st.error(f"⚠️ Found {bad_price_count} rows with zero or negative prices. This will break the backtest engine.")
        
    if jump_count > 0:
        st.warning(f"⚠️ Found {jump_count} single-day jumps >50%. If this is a long-term backtest, Yahoo Finance likely failed to adjust for a stock split. Returns may be inaccurate.")
        with st.expander("View Massive Jumps"):
            jump_df = prices.loc[jumps.index, ["TQQQ_close"]].copy()
            jump_df["Return"] = jumps * 100
            st.dataframe(jump_df.style.format({"TQQQ_close": "${:.2f}", "Return": "{:+.1f}%"}))
            
    if missing_days > expected_missing + 20:
        st.warning(f"⚠️ Found {missing_days} missing business days (expected ~{expected_missing:.0f} for holidays). Data may have gaps.")
        st.session_state["backtest_warning"] = f"⚠️ Data gap detected: {missing_days} missing trading days."
    else:
        st.session_state.pop("backtest_warning", None)
    
    if bad_price_count == 0 and jump_count == 0 and missing_days <= expected_missing + 5:
        st.success("✅ Data quality looks excellent. No gaps, bad ticks, or massive split jumps detected.")

    st.markdown("---")

def _suggest_ticker(symbol: str) -> str | None:
    symbol = symbol.strip().upper()
    return _COMMON_TICKER_FIXES.get(symbol)


def _fmt_pct(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "no data"
    return f"{v * 100:.1f}%"

def get_currency_symbol(exchange: str, ticker: str = "") -> str:
    """Determine currency symbol based on exchange and ticker."""
    # Indian exchanges
    if exchange and exchange.upper() in ("NSE", "BSE"):
        return "₹"
    # If ticker ends with .NS or .BO, it's Indian
    if ticker and (ticker.endswith(".NS") or ticker.endswith(".BO")):
        return "₹"
    # Everything else is USD by default
    return "$"

def _fmt_money(v: float | None, exchange: str = "", ticker: str = "") -> str:
    if v is None:
        return "n/a"

    currency = get_currency_symbol(exchange, ticker)

    if abs(v) >= 1e12:
        return f"{currency}{v / 1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"{currency}{v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{currency}{v / 1e6:.2f}M"
    return f"{currency}{v:,.2f}"

def _as_fraction(value, *, is_percent: bool) -> float | None:
    """Normalize a Yahoo Finance percentage-ish field to a fraction.

    Yahoo returns some fields as fractions (yield=0.0098 means 0.98%)
    and others as percents (netExpenseRatio=0.0945 means 0.0945%).
    Pass is_percent=True when the raw value is already a percent.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v / 100.0 if is_percent else v

def _color_pass(val: float | None) -> str:
    if val is None:
        return "color: #888;"
    if val >= PASS_THRESHOLD:
        return "background-color: #1e4620; color: #b6f0b6;"
    return "background-color: #4b1e1e; color: #f0b6b6;"



def _render_big5_table(big5: Big5Result) -> None:
    df = big5.as_dataframe()

    # Calculate average across all Big 5 metrics for each window column
    avg_row = {"metric": "Average (All Metrics)"}
    for w in WINDOWS:
        col = f"{w}yr"
        valid_vals = [v for v in df[col] if pd.notna(v)]
        avg_row[col] = sum(valid_vals) / len(valid_vals) if valid_vals else None

    # Determine if overall average passes the 10% threshold
    avg_pass_count = sum(
        1 for w in WINDOWS if avg_row[f"{w}yr"] is not None and avg_row[f"{w}yr"] >= PASS_THRESHOLD
    )
    avg_row["pass"] = avg_pass_count >= len(WINDOWS) // 2

    # Append average row to dataframe
    df_with_avg = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # Format values for display
    display = df_with_avg.copy()
    for w in WINDOWS:
        col = f"{w}yr"
        display[col] = display[col].map(_fmt_pct)
    display["pass"] = display["pass"].map(lambda p: "PASS" if p else "FAIL")
    display = display.rename(columns={"metric": "Metric", "pass": "Verdict"})

    def _style(row):
        original = df_with_avg.loc[row.name]
        is_avg_row = row["Metric"] == "Average (All Metrics)"
        styles = pd.Series("", index=row.index)

        # Style Metric column
        if is_avg_row:
            styles["Metric"] = "font-weight: bold; background-color: rgba(255,255,255,0.05);"

        # Style year window columns dynamically based on column names
        for w in WINDOWS:
            col_name = f"{w}yr"
            if col_name in styles.index:
                cell_style = _color_pass(original[col_name])
                if is_avg_row:
                    cell_style += " font-weight: bold;"
                styles[col_name] = cell_style

        # Style Verdict column
        verdict_style = (
            "background-color: #1e4620; color: #b6f0b6;"
            if original["pass"]
            else "background-color: #4b1e1e; color: #f0b6b6;"
        )
        if is_avg_row:
            verdict_style += " font-weight: bold;"
        styles["Verdict"] = verdict_style

        return styles

    styled = display.style.apply(_style, axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)


def _verdict_price(current: float | None, mos: float | None, value: float | None) -> tuple[str, str, str]:
    """Return (headline, detail, color) for the verdict cell.

    Tiers (Buffett + one extra):
      - BARGAIN BUY: current price is <= 50% of MOS Buy Price (already-half of MOS)
      - BUY:        current <= MOS Buy Price
      - WATCH:      current between MOS and value (Intrinsic)
      - AVOID:      current > value (Intrinsic)
    """
    if current is None or mos is None or value is None or value <= 0 or mos <= 0:
        return ("Unknown", "", "gray")
    if current <= mos * 0.5:
        pct = (mos - current) / mos * 100
        return ("BARGAIN BUY", f"{pct:.0f}% below MOS Buy Price — rare deep discount", "bargain")
    if current <= mos:
        pct = (mos - current) / mos * 100
        return ("BUY", f"{pct:.0f}% below MOS Buy Price", "green")
    if current <= value:
        pct = (value - current) / value * 100
        return ("WATCH", f"between MOS and Intrinsic Value ({pct:.0f}% below Intrinsic)", "orange")
    pct = (current - value) / value * 100
    return ("AVOID", f"above Intrinsic Value by {pct:.0f}%", "red")


def _big5_eps_growth(big5: Big5Result) -> float | None:
    """Pick a stable EPS growth rate from the Big 5 EPS CAGR windows for valuation.

    Rule: use the **median of the 5yr and 3yr windows** — these capture the
    business trend without letting a single weak/strong year dominate. The 1yr
    window is only used as a last resort when longer windows aren't computable
    (e.g. foreign tickers with 4yrs of yfinance data). The 10yr window is
    included when it exists to further stabilize.

    Rationale: the "lowest window" rule (Buffett's discipline) is too brittle
    for tickers with short history — a single flat year (e.g. KSPI's 1yr = 3%)
    can suppress the value price by 4× vs the 3yr trend of 22%.
    """
    windows = big5.eps.values  # {10: v, 5: v, 3: v, 1: v}
    stable = [v for w, v in windows.items() if w in (5, 3, 10) and v is not None]
    if stable:
        stable.sort()
        n = len(stable)
        # median (with even-count average)
        if n % 2:
            growth = stable[n // 2]
        else:
            growth = (stable[n // 2 - 1] + stable[n // 2]) / 2
        
        # --- FIX: Cap growth at 30% and ensure minimum 2% ---
        if growth > 0.30:
            growth = 0.30
        if growth < 0.02:
            growth = 0.02
        return growth
    
    # Fallback: 1yr only when nothing longer is computable
    one = windows.get(1)
    if one is not None:
        # --- FIX: Cap fallback growth at 30% and ensure minimum 2% ---
        if one > 0.30:
            one = 0.30
        if one < 0.02:
            one = 0.02
        return one
    return None


def _inject_tab_styles() -> None:
    """Global CSS: buttons, headers, links, sidebar nav styling."""
    st.markdown(
        """
        <style>
        /* ----- Buttons ----- */
        div.stButton > button,
        div[data-testid="stForm"] button,
        button {
            background-color: #B0B8C0 !important;
            color: #1E293B !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.2s ease !important;
        }
        div.stButton > button:hover,
        div[data-testid="stForm"] button:hover,
        button:hover {
            background-color: #C8D0D8 !important;
            color: #1E293B !important;
            border: none !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
            transform: translateY(-1px) !important;
        }
        div.stButton > button:active,
        div[data-testid="stForm"] button:active,
        button:active {
            background-color: #9AA2AA !important;
            transform: translateY(0px) !important;
        }

        /* ----- Sidebar navigation links (Yahoo-Mail style) ----- */
        section[data-testid="stSidebar"] {
            background-color: #0b0f14;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {
            padding: 0.7rem 0.9rem !important;
            border-radius: 8px !important;
            color: #c5c8cd !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            transition: background-color 0.15s ease !important;
            margin-bottom: 2px;
        }
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {
            background-color: rgba(255,255,255,0.06) !important;
            color: #e8eaed !important;
        }
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] {
            background-color: rgba(76, 175, 80, 0.16) !important;
            color: #b6f0b6 !important;
            border-left: 3px solid #4CAF50 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_tab_styles()

def render_etf_dashboard(symbol: str, yft) -> None:
    """Lightweight ETF view — no financial statements, just fund facts + price stats."""
    import yfinance as yf

    try:
        info = yft.info or {}
    except Exception:
        info = {}

    name = info.get("longName") or info.get("shortName") or symbol
    st.subheader(f"{name} ({symbol.upper()})")
    st.caption(
        f"🏦 Exchange-Traded Fund · {info.get('category', 'ETF')} · "
        f"Listed on {info.get('exchange', 'n/a')}"
    )
    st.info(
        "ETFs don't report financial statements, so Big 5 growth, DCF, and the "
        "value-investing formulas don't apply. Below are the fund-level metrics "
        "that matter for an ETF. For entry/exit signals, use **📈 Technical Analysis**."
    )

    # --- Top-line metrics strip ---
    _last_price = info.get("regularMarketPrice") or info.get("previousClose") or info.get("navPrice")
    _aum = info.get("totalAssets")
    _expense = _as_fraction(info.get("netExpenseRatio") or info.get("annualReportExpenseRatio"), is_percent=True,)
    _yield = _as_fraction(info.get("yield"), is_percent=False)
    _beta = info.get("beta3Year") or info.get("beta")
    _ytd = _as_fraction(info.get("ytdReturn"), is_percent=True)
    

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Price", f"${_last_price:,.2f}" if _last_price else "n/a")
    m2.metric("Fund Size (AUM)", _fmt_money(_aum, "USD") if _aum else "n/a")
    m3.metric("Expense Ratio", f"{_expense * 100:.2f}%" if _expense is not None else "n/a")
    m4.metric("Distribution Yield", f"{_yield * 100:.2f}%" if _yield is not None else "n/a")
    m5.metric("Beta (3yr)", f"{_beta:.2f}" if _beta else "n/a")
    m6.metric("YTD Return", f"{_ytd * 100:+.1f}%" if _ytd is not None else "n/a", delta=f"{_ytd * 100:+.1f}%" if _ytd is not None else None, delta_color="normal",)
    

    # --- 52-week range ---
    _low = info.get("fiftyTwoWeekLow")
    _high = info.get("fiftyTwoWeekHigh")
    if _low and _high and _last_price:
        _pos_pct = (_last_price - _low) / (_high - _low) * 100 if _high > _low else 0
        st.markdown("#### 52-Week Range")
        st.markdown(
            f"""
            <div style="margin: 0.25rem 0 1rem;">
                <div style="display:flex; justify-content:space-between; color:#9aa0a6; font-size:0.85rem;">
                    <span>${_low:,.2f}</span><span>${_high:,.2f}</span>
                </div>
                <div style="position:relative; height:8px; background:rgba(255,255,255,0.08); border-radius:4px; margin-top:0.5rem; overflow:hidden;">
                    <div style="position:absolute; left:0; top:0; width:{_pos_pct:.1f}%; height:100%; background:#4DABF7; border-radius:4px 0 0 4px;"></div>
                    <div style="position:absolute; left:{_pos_pct:.1f}%; top:-4px; width:3px; height:16px; background:#00E676; border-radius:2px; transform:translateX(-50%);"></div>
                </div>
                <div style="color:#9aa0a6; font-size:0.75rem; margin-top:0.4rem;">
                    Current ${_last_price:,.2f} sits at {_pos_pct:.0f}% of the 52-week range.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Top holdings ---
    try:
        fd = yft.funds_data
        holdings = fd.top_holdings
        if holdings is not None and not holdings.empty:
            st.markdown("#### Top Holdings")
            _holdings_display = holdings.copy()
            if "Holding Percent" in _holdings_display.columns:
                _holdings_display["Holding Percent"] = _holdings_display["Holding Percent"].map(
                    lambda v: f"{v*100:.2f}%" if v is not None and v == v else "n/a"
                )
            st.dataframe(_holdings_display, width="stretch")
    except Exception:
        pass

    # --- Sector weightings ---
    try:
        fd = yft.funds_data
        sectors = fd.sector_weightings
        if sectors:
            st.markdown("#### Sector Weightings")
            sec_df = (
                pd.DataFrame(
                    [{"Sector": k.replace("_", " ").title(), "Weight": v}
                     for k, v in sectors.items()]
                )
                .sort_values("Weight", ascending=False)
                .reset_index(drop=True)
            )
            sec_df["Weight"] = sec_df["Weight"].map(lambda v: f"{v*100:.2f}%")
            st.dataframe(sec_df, width="stretch", hide_index=True)
    except Exception:
        pass

    # --- Annualized Returns (5/10/20 year) ---
    st.markdown("#### Annualized Returns (Total Return, CAGR)")
    rets = compute_annualized_returns(yft, symbol)
    if rets:
        r1, r2, r3 = st.columns(3)

        def _cagr_metric(col, label: str, value: float | None) -> None:
            if value is None:
                col.metric(label, "n/a")
                return
            pct = value * 100
            col.metric(
                label,
                f"{pct:+.2f}%",
                delta="positive" if pct >= 0 else "negative",
                delta_color="normal" if pct >= 0 else "inverse",
            )

        _cagr_metric(r1, "5-Year", rets.get(5))
        _cagr_metric(r2, "10-Year", rets.get(10))
        _cagr_metric(r3, "20-Year", rets.get(20))

        st.caption("Adjusted for dividends and splits. n/a if the ETF is younger than the period.")
    else:
        st.caption("Annualized returns unavailable.")
    
    # --- Price History with selectable window ---
    st.markdown("#### Price History")

    _window_options = {"1Y": 1,"5Y": 5,"10Y": 10,"20Y": 20,}

    selected_window = st.radio(
        "Window:",
        options=list(_window_options.keys()),
        index=0,                       # 1Y default
        horizontal=True,
        key=f"etf_price_window_{symbol}",
        label_visibility="collapsed",
    )
    years = _window_options[selected_window]

    try:
        hist = yft.history(period=f"{years}y")
    except Exception:
        hist = None

    if hist is None or hist.empty or "Close" not in hist.columns:
        st.caption(f"n/a — {symbol} has less than {years} years of history.")
    else:
        series = hist["Close"].dropna()
        base = float(series.iloc[0]) if not series.empty else 0.0
        if series.empty or base <= 0:
            st.caption("n/a — no usable data in this window.")
        else:
            total_return = (float(series.iloc[-1]) / base - 1) * 100
            st.caption(
                f"{series.index[0].date()} → {series.index[-1].date()} · "
                f"+{total_return:.1f}% total return"
            )
            st.line_chart(series, height=260)

    st.caption("Data from Yahoo Finance. Expense ratios, AUM, and holdings update daily; " "verify against the fund's prospectus before investing.")


def render_analyzer() -> None:
    st.markdown(
        '<h2 style="margin-bottom: 0.25rem;">'
        '🔍 Stock Analyzer '
        '<span style="font-size: 0.95rem; font-weight: 500; color: #4CAF50; letter-spacing: 0.03em;">'
        '— Separating Wonderful Businesses from the Merely Good'
        '</span>'
        '</h2>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Enter a US-listed ticker (e.g. AAPL, MSFT, KO) or an Indian ticker "
        "(e.g. RELIANCE, TCS). Results cover 5 key growth rates, five valuation "
        "methods (DCF, Peter Lynch Fair Value, Graham Number, Graham Formula, "
        "and PEG), plus Buffett-style health checks."
    )

    
        
    with st.form("analyze"):
        col_a, col_b, col_c = st.columns([2, 1, 5])
        with col_a:
            ticker = st.text_input("Ticker symbol", value="AAPL", max_chars=10).strip().upper()
        with col_b:
            # Blank markdown pushes the button down so it lines up with the input
            # (whose label adds ~28px of height above it).
            st.markdown("<div style='height: 28px;color: #FFFFFF'></div>", unsafe_allow_html=True)
            # Check if we're currently analyzing
            if "analyzing" in st.session_state and st.session_state["analyzing"]:
                submitted = st.form_submit_button("⏳ Analyzing...", width="stretch", disabled=True)
            else:
                submitted = st.form_submit_button("Analyze", width="stretch")

    if submitted:
        st.session_state["analyzing"] = True
        # Check if this is a new ticker
        if st.session_state.get("last_ticker") != ticker:
            # Clear old cached data for previous ticker
            st.cache_data.clear()
            st.session_state["valuation_ticker"] = None
            st.session_state["valuation_growth_result"] = None

        st.session_state["last_ticker"] = ticker
        suggestion = _suggest_ticker(ticker)
        if suggestion and suggestion != ticker:
            st.warning(f"'{ticker}' looks like a typo. Using '{suggestion}' instead.")
            ticker = suggestion
            st.session_state["last_ticker"] = ticker

    if not submitted and "last_ticker" not in st.session_state:
        st.info("Enter a US-listed ticker (e.g. AAPL, MSFT, KO) OR Indian stock ticker (e.g. RELIANCE, TCS, HDFC) and click **Analyze**.")
        return

    if submitted:
        st.session_state["last_ticker"] = ticker

    symbol = st.session_state.get("last_ticker", ticker)

    if "valuation_ticker" not in st.session_state or st.session_state["valuation_ticker"] != symbol:
        st.session_state["valuation_ticker"] = symbol
        st.session_state["valuation_growth_mode"] = "Value conservative"
        st.session_state["valuation_growth_result"] = None

    # --- Detect instrument type before doing anything expensive ---
    from data_provider import get_ticker

    with st.spinner(f"Looking up {symbol}..."):
        try:
            _probe = get_ticker(symbol)
            _qtype = (_probe.info or {}).get("quoteType", "").upper()
        except Exception:
            _qtype = ""

    if _qtype in ("ETF", "MUTUALFUND", "INDEX", "CRYPTOCURRENCY", "CURRENCY"):
        st.session_state["analyzing"] = False
        render_etf_dashboard(symbol, _probe)
        return

    with st.spinner(f"Fetching 10-year financials for {symbol}..."):
        try:
            if st.session_state.get("valuation_ticker") != symbol:
                st.cache_data.clear()
                st.session_state["valuation_ticker"] = symbol
            fin = _cached_fetch(symbol)
            st.session_state["analyzing"] = False
        except FetchError as e:
            st.session_state["analyzing"] = False
            st.error(str(e))
            # Add a fix-it button right there
            st.warning("This usually means the local cache is corrupted.")
            if st.button(f"🧹 Clear cache for {symbol} and try again", key=f"fix_{symbol}"):
                if clear_yfinance_cache(symbol):
                    st.cache_data.clear()
                    st.rerun()
            return
        except Exception as e:
            st.session_state["analyzing"] = False
            st.error(f"Unexpected error fetching {symbol}: {e}")
            # Add a fix-it button right there
            st.warning("This usually means the local cache is corrupted.")
            if st.button(f"🧹 Clear cache for {symbol} and try again", key=f"fix_{symbol}"):
                if clear_yfinance_cache(symbol):
                    st.cache_data.clear()
                    st.rerun()
            return

    # Company Name Display 
    st.subheader(f"{fin.company_name} ({fin.ticker})")
    # Guardrail: Yahoo sometimes returns absurd analyst growth rates (e.g., 247%) 
    # for small caps or post-earnings anomalies. Cap it at 25% so it doesn't
    # blow up the DCF and intrinsic value calculations into the millions.
    if getattr(fin, "analyst_5yr_growth", None) is not None:
        if fin.analyst_5yr_growth > 0.25:
            fin.analyst_5yr_growth = 0.25
        elif fin.analyst_5yr_growth < 0.0:
            fin.analyst_5yr_growth = 0.0    

    # Company Summary
    if fin.company_summary or fin.company_sector or fin.company_industry or fin.company_country:
        with st.expander("Company Summary:", expanded=True):
            # Prefer the more specific "industry" over the broad "sector".
            # Fall back to sector if industry is missing. Always append the
            # country, normalizing US variants to a consistent "USA" label.
            _label_parts = []
            _label_parts.append(fin.company_industry or fin.company_sector)
            if fin.company_country:
                _country_display = {
                    "United States": "USA",
                    "USA": "USA",
                    "US": "USA",
                }.get(fin.company_country, fin.company_country)
                _label_parts.append(_country_display)
            _label = " · ".join(p for p in _label_parts if p)

            if _label:
                st.markdown(
                    f'<div style="color:#9aa0a6; font-size:0.85rem; '
                    f'text-transform:uppercase; letter-spacing:0.05em; '
                    f'margin-bottom:0.5rem;">'
                    f'{_label}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if fin.company_summary:
                st.markdown(_first_n_sentences(fin.company_summary, 2))
            else:
                st.caption(
                    "Detailed summary unavailable for this ticker "
                    "(common for foreign filers and small caps)."
                )

    # ADD THIS LINE:
    if fin.exchange in ("NSE", "BSE"):
        st.caption(f"🇮🇳 Listed on {fin.exchange} (₹ INR)")
    elif fin.exchange:
        st.caption(f"US Listed on {fin.exchange}  ($ USD)")
    _eps_ttm = float(fin.eps.iloc[-1]) if not fin.eps.empty else None

    # ============================================================
    # CORE CALCULATIONS
    # ============================================================
    big5 = compute_big5(fin)
    big5_eps_g = _big5_eps_growth(big5)

    # --- EPS selection: prefer latest year, but fall back to a 3yr positive-EPS
    # average when the latest year is negative (value Price and other formulas
    # can't handle negative earnings). This surfaces to the user as an amber note.
    _eps_ttm_raw = float(fin.eps.iloc[-1]) if not fin.eps.empty else None
    _normalized_eps = None
    _normalized_years: list[int] = []
    if _eps_ttm_raw is not None and _eps_ttm_raw <= 0 and not fin.eps.empty:
        positive_history = fin.eps[fin.eps > 0]
        if not positive_history.empty:
            _normalized_eps = float(positive_history.mean())
            _normalized_years = [int(y) for y in positive_history.index]

    # Keep current_eps for display purposes
    current_eps = _normalized_eps if _normalized_eps is not None else _eps_ttm_raw
    
    # ADD THIS: Use the historical positive average for math, but fall back to current_eps if no history exists
    eps_for_valuation = _normalized_eps if _normalized_eps is not None else _eps_ttm_raw

    if _normalized_eps is not None:
        yrs_txt = ", ".join(str(y) for y in _normalized_years)
        st.warning(
            f":warning: Latest reported EPS is negative (${_eps_ttm_raw:.2f}). "
            f"To calculate intrinsic value, the app has substituted a **historical average EPS of ${_normalized_eps:.2f}**, "
            f"computed from the last {len(_normalized_years)} profitable years ({yrs_txt}). "
            f"Treat these valuations as guidance about what this business *could* be worth "
            f"if it returns to prior profitability, not what it's worth today."
        )

    def _moat_conservative_growth() -> float | None:
        """Return a conservative 15% growth rate for Buffett-style valuation."""
        # Always return 15% - this is Buffett's standard conservative rate
        return 0.15

    def _valuation_growth_rate_label(mode: str) -> str:
        if mode == "Value conservative":
            rate = _moat_conservative_growth()
            return f"Value conservative: {_fmt_pct(rate) if rate is not None else 'n/a'}"
        if mode == "Big 5 EPS growth":
            return f"Big 5 EPS growth: {_fmt_pct(big5_eps_g) if big5_eps_g is not None else 'n/a'}"
        if mode == "Analyst 5Y growth":
            return f"Analyst 5Y growth: {_fmt_pct(fin.analyst_5yr_growth) if fin.analyst_5yr_growth is not None else 'n/a'}"
        if mode == "Custom":
            return "Custom"
            
        return mode

    def _valuation_growth_mode_options() -> list[str]:
        options = ["Value conservative"]
        if big5_eps_g is not None:
            options.append("Big 5 EPS growth")
        if fin.analyst_5yr_growth is not None:
            options.append("Analyst 5Y growth")
        options.append("Custom")
        return options

    def _compute_valuation_value(mode: str):
        # st.write("DEBUG COMPUTE:", mode, "analyst=", fin.analyst_5yr_growth, "custom=", st.session_state.get("custom_growth_rate"))
        if mode == "Custom":
            custom_growth_pct = float(st.session_state.get("custom_growth_rate", 0) or 0)
            if custom_growth_pct <= 0:
                return None
            custom_growth = custom_growth_pct / 100.0
            return value_price(
                current_eps=eps_for_valuation,
                big5_eps_growth=None,
                analyst_growth=None,
                historical_pe=fin.pe_ratio_ttm,
                custom_growth=custom_growth,
            )

        if mode == "Value conservative":
            # --- FIX: Use the conservative growth function (returns 15%) ---
            conservative_growth = _moat_conservative_growth()
            if conservative_growth is None:
                return None
            
            # If EPS is negative or missing, we force a tiny $0.01 EPS 
            # just so the math completes and gives us a theoretical floor value.
            eps_for_math = eps_for_valuation
            if eps_for_math is None or eps_for_math <= 0:
                eps_for_math = 0.01

            return value_price(
                current_eps=eps_for_math,
                big5_eps_growth=None,           
                analyst_growth=None,            
                historical_pe=fin.pe_ratio_ttm,
                custom_growth=conservative_growth,
            )

        if mode == "Big 5 EPS growth":
            if big5_eps_g is None:
                return None
            return value_price(
                current_eps=eps_for_valuation,
                big5_eps_growth=big5_eps_g,
                analyst_growth=None,
                historical_pe=fin.pe_ratio_ttm,
            )

        if mode == "Analyst 5Y growth":
            if fin.analyst_5yr_growth is None:
                return None
            return value_price(
                current_eps=eps_for_valuation,
                big5_eps_growth=None,
                analyst_growth=fin.analyst_5yr_growth,
                historical_pe=fin.pe_ratio_ttm,
            )

        return None

    # Initialize Valuation State early so top metrics can use it
    if "valuation_growth_result" not in st.session_state:
        st.session_state["valuation_growth_result"] = None
    if "custom_growth_rate" not in st.session_state:
        st.session_state["custom_growth_rate"] = 0
    if isinstance(st.session_state["custom_growth_rate"], float):
        st.session_state["custom_growth_rate"] = int(round(st.session_state["custom_growth_rate"] * 100))

    mode_options = _valuation_growth_mode_options()
    if "valuation_growth_mode" not in st.session_state or st.session_state["valuation_growth_mode"] not in mode_options:
        st.session_state["valuation_growth_mode"] = "Value conservative"
    selected_growth_mode = st.session_state["valuation_growth_mode"]

    val = st.session_state.get("valuation_growth_result")
    if val is None:
        val = _compute_valuation_value(selected_growth_mode)
        st.session_state["valuation_growth_result"] = val

    # --- MOVED UP: Define the rendering helpers before calling them ---
    _VERDICT_TOOLTIPS = {
        "BARGAIN BUY": "Current price is at or below 50% of the MOS Buy Price...",
        "BUY": "Current price is below the Margin of Safety (MOS) Buy Price...",
        "WATCH": "Current price is between the MOS Buy Price and the Intrinsic Value...",
        "AVOID": "AVOID here doesn't mean the stock is a bad buy...",
        "Unknown": "Not enough data to compute a verdict.",
    }

    _ANALYST_STYLE = {
        "strong_buy":  ("STRONG BUY",  "bargain"),
        "buy":         ("BUY",         "green"),
        "outperform":  ("BUY",         "green"),
        "hold":        ("HOLD",        "orange"),
        "underperform":("SELL",        "red"),
        "sell":        ("SELL",        "red"),
        "strong_sell": ("STRONG SELL", "red"),
    }

    def _render_analyst_cell(col, fin) -> None:
        key = fin.analyst_rec_key
        count = fin.analyst_count
        label, color = _ANALYST_STYLE.get(key or "", (None, "gray"))

        if label is None or not count:
            col.markdown(
                '<div style="color: rgba(250,250,250,0.6); font-size: 0.875rem;">Analyst Rating</div>'
                '<div style="font-size: 1.5rem; color: #9AA0A6;">n/a</div>'
                '<div style="font-size: 0.72rem; color: #9aa0a6;">no analyst coverage</div>',
                unsafe_allow_html=True,
            )
            return

        color_hex = _COLOR_HEX[color]
        detail_parts = [f"{count} analyst{'s' if count != 1 else ''}"]
        if fin.analyst_target_mean and fin.current_price and fin.current_price > 0:
            tgt = fin.analyst_target_mean
            upside = (tgt - fin.current_price) / fin.current_price * 100
            detail_parts.append(f"target ${tgt:.0f} ({upside:+.0f}%)")
        detail = " · ".join(detail_parts)

        col.markdown(
            f'<div style="color: rgba(250,250,250,0.6); font-size: 0.875rem;">Analyst Rating</div>'
            f'<div style="font-size: 2.0rem; font-weight: 400; line-height: 1.2; color: {color_hex};">{label}</div>'
            f'<div style="font-size: 0.72rem; color: #9aa0a6; line-height: 1.3; margin-top: 0.2rem;">{detail}</div>',
            unsafe_allow_html=True,
        )

    def _render_verdict_cell(col, verdict: str, detail: str, color: str) -> None:
        tip_raw = _VERDICT_TOOLTIPS.get(verdict, _VERDICT_TOOLTIPS["Unknown"])
        tip = tip_raw.replace('"', "&quot;")
        col.markdown(
            f'<div style="color: rgba(250,250,250,0.6); font-size: 0.875rem;">'
            f'Verdict <span title="{tip}" style="cursor: help; color: #4CAF50; margin-left: 4px; font-size: 0.9rem;">&#9432;</span>'
            f'</div>'
            f'<div style="font-size: 2.0rem; font-weight: 400; line-height: 1.2; color: {_COLOR_HEX[color]};">{verdict}</div>'
            f'<div style="font-size: 0.72rem; color: #9aa0a6; line-height: 1.3; margin-top: 0.2rem;">{detail}</div>',
            unsafe_allow_html=True,
        )
    # ---------------------------------------------------------------------

    if val is None:
        st.warning(
            "⚠️ Cannot compute standard Value Price — this company has negative earnings or missing growth data. "
            "Use the **Custom** growth source below to manually input a growth rate and calculate a theoretical value."
        )

    # Safe evaluation (checks if val is None before trying to read val.mos_price)
    v_verdict, v_detail, v_color = _verdict_price(
        fin.current_price, 
        val.mos_price if val else None, 
        val.value_price if val else None
    )

    # ============================================================
    # METRICS ROW
    # ============================================================
    _intrinsic_sub = ""
    _intrinsic_color = "#9aa0a6"
    if val and fin.current_price and val.value_price:
        _upside = (val.value_price - fin.current_price) / fin.current_price
        if _upside < 0:
            _intrinsic_sub = f"{abs(_upside)*100:.1f}% above fair"
            _intrinsic_color = "#FFA726" # Orange
        else:
            _intrinsic_sub = f"{_upside*100:.1f}% below fair"
            _intrinsic_color = "#00E676" # Green

    # Give Market Cap (col 2) a bit more room to breathe
    top1, top2, top3, top4, top5, top6, top7, top8 = st.columns([1, 1.4, 1, 1, 1, 1, 1, 1.2])
    # Replace top1.metric with the custom HTML version to support the colored sub-line
    top1.markdown(_metric_html(
        "Current Price", 
        _fmt_money(fin.current_price, fin.exchange, fin.ticker), 
        sub=_intrinsic_sub, 
        sub_color=_intrinsic_color
    ), unsafe_allow_html=True)
    top2.metric("Market Cap", _fmt_money(fin.market_cap, fin.exchange, fin.ticker))
    top3.metric("EPS (TTM)", _fmt_money(_eps_ttm, fin.exchange or "USD"))
    top4.metric("TTM P/E", f"{fin.pe_ratio_ttm:.1f}" if fin.pe_ratio_ttm else "n/a")

    # Long-term debt with health coloring
    _ltd_ttm = float(fin.long_term_debt.iloc[-1]) if not fin.long_term_debt.empty else None
    _fcf_ttm = float(fin.free_cash_flow.iloc[-1]) if not fin.free_cash_flow.empty else None

    if _ltd_ttm is None:
        _debt_value = "n/a"
        _debt_color = "#9AA0A6"
        _debt_sub = ""
    elif _ltd_ttm <= 0:
        _debt_value = "$0"
        _debt_color = "#9AA0A6"
        _debt_sub = "debt-free"
    elif _fcf_ttm is None:
        _debt_value = _fmt_money(_ltd_ttm, fin.exchange, fin.ticker)
        _debt_color = "#9AA0A6"
        _debt_sub = "no FCF data"
    elif _fcf_ttm <= 0:
        _debt_value = _fmt_money(_ltd_ttm, fin.exchange, fin.ticker)
        _debt_color = "#EF5350"
        _debt_sub = "⚠ SEVERE · negative FCF"
    else:
        _years_to_pay = _ltd_ttm / _fcf_ttm
        _debt_value = _fmt_money(_ltd_ttm, fin.exchange, fin.ticker)
        if _years_to_pay > 5:
            _debt_color = "#EF5350"
            _debt_sub = f"{_years_to_pay:.1f}× FCF · Heavy"
        elif _years_to_pay > 3:
            _debt_color = "#FFC107"   # amber
            _debt_sub = f"{_years_to_pay:.1f}× FCF · Elevated"
        else:
            _debt_color = "#00E676"
            _debt_sub = f"{_years_to_pay:.1f}× FCF · Healthy"

    top5.markdown(_metric_html("LT Debt", _debt_value, sub=_debt_sub, sub_color=_debt_color,
        help_text=(
            "Long-term debt ÷ TTM free cash flow, shown as 'years to pay off'. "
            "Green ≤ 3×, yellow 3–5×, red > 5×. Negative FCF is the worst case — "
            "no operating cash to service the debt."
        ),
    ), unsafe_allow_html=True)

    top6.metric("BVPS", _fmt_money(fin.book_value_per_share, fin.exchange, fin.ticker))
    top7.metric("Years of data", str(fin.years_available))

    if fin.week52_high and fin.week52_low and fin.week52_high > fin.week52_low:
        _w52_pct = (
            (fin.current_price - fin.week52_low)
            / (fin.week52_high - fin.week52_low) * 100
            if fin.current_price else None
        )
        _w52_sub = (
            f"Current ${fin.current_price:,.2f} ({_w52_pct:.0f}%)"
            if fin.current_price and _w52_pct is not None
            else ""
        )
        top8.markdown(_metric_html(
            "52-Week Range",
            f"${fin.week52_low:,.0f} – ${fin.week52_high:,.0f}",
            sub=_w52_sub,
            sub_color="#9aa0a6",
            help_text=(
                f"52-week low ${fin.week52_low:,.2f}, "
                f"high ${fin.week52_high:,.2f}."
                + (
                    f" Current price ${fin.current_price:,.2f} sits at "
                    f"{_w52_pct:.0f}% of the range."
                    if _w52_pct is not None else ""
                )
            ),
        ), unsafe_allow_html=True)
    elif fin.week52_high and fin.week52_low:
        top8.markdown(_metric_html(
            "52-Week Range",
            f"${fin.week52_low:,.0f} – ${fin.week52_high:,.0f}",
        ), unsafe_allow_html=True)
    else:
        top8.metric("52-Week Range", "n/a")

    # Data-source note is shown in the footer only — not up top. Split-artifact
    # flag is precomputed here and rendered alongside the footer note.
    _show_split_warning = False
    if fin.data_source.startswith("edgar") and not fin.eps.empty and len(fin.eps) >= 3:
        ratios = fin.eps.diff() / fin.eps.shift(1).abs()
        _show_split_warning = bool((ratios.abs() > 0.5).any())

    # ============================================================
    # BIG 5 TABLE
    # ============================================================
    st.markdown(
        '### Big 5 Growth Rates — <span style="color: #00E676;">Is this a Wonderful Company?</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "As per Buffett, a wonderful business has a durable competitive advantage — a "
        f'"moat". Ideally, all five metrics below should show historical growth rates of '
        f"10% or more per year over the 10, 5, 3, and 1-year windows "
        "(adjust thresholds in the sidebar). All these numbers are Compounded Annual "
        "Growth Rates (CAGRs)."
    )
    st.caption("Note: ROIC is shown as a period-average return, not a CAGR.")
    _render_big5_table(big5)

    # --- Wonderfulness hero banner ---
    _ws = big5.wonderfulness()
    _ws_color_hex = _COLOR_HEX[_ws.color]
    # Score is 0-10; render a horizontal fill proportional to it.
    _bar_pct = int(round(_ws.overall * 10))
    _strength_html = " · ".join(f"<span style='color:#b6f0b6'>{s}</span>" for s in _ws.strengths) if _ws.strengths else "<span style='color:#9aa0a6'>—</span>"
    _weakness_html = " · ".join(f"<span style='color:#f0b6b6'>{s}</span>" for s in _ws.weaknesses) if _ws.weaknesses else "<span style='color:#9aa0a6'>—</span>"

    st.markdown(
        f"""
        <div style="border-radius: 10px; padding: 1.25rem 1.5rem; margin: 0.75rem 0 1rem;
                    background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.06));
                    border: 1px solid rgba(255,255,255,0.08);">
            <div style="font-size: 0.85rem; color: #9aa0a6; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;">
                How wonderful is the company:
            </div>
            <div style="display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap;">
                <div style="font-size: 3.5rem; font-weight: 700; line-height: 1; color: {_ws_color_hex};">
                    {_ws.overall:.1f}<span style="font-size: 1.5rem; color: #9aa0a6; font-weight: 400;">/10</span>
                </div>
                <div style="flex: 1;">
                    <div style="font-size: 1.5rem; font-weight: 600; color: {_ws_color_hex};">{_ws.label}</div>
                    <div style="font-size: 0.85rem; color: #9aa0a6; margin-top: 0.15rem;">
                        {_ws.checks_passed}/{_ws.checks_total} checks pass the 10% bar
                    </div>
                </div>
            </div>
            <div style="height: 8px; background: rgba(255,255,255,0.06); border-radius: 4px; margin: 0.75rem 0 0.9rem; overflow: hidden;">
                <div style="height: 100%; width: {_bar_pct}%; background: {_ws_color_hex}; border-radius: 4px;"></div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; font-size: 0.85rem;">
                <div>
                    <div style="color:#9aa0a6; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;">Pass Rate</div>
                    <div style="font-size: 1.25rem; color: {_ws_color_hex};">{_ws.pass_rate:.1f}/10</div>
                    <div style="color:#9aa0a6; font-size: 0.7rem;">How many checks beat 10%</div>
                </div>
                <div>
                    <div style="color:#9aa0a6; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;">Magnitude</div>
                    <div style="font-size: 1.25rem; color: {_ws_color_hex};">{_ws.magnitude:.1f}/10</div>
                    <div style="color:#9aa0a6; font-size: 0.7rem;">How far above 10% they sit</div>
                </div>
                <div>
                    <div style="color:#9aa0a6; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;">Consistency</div>
                    <div style="font-size: 1.25rem; color: {_ws_color_hex};">{_ws.consistency:.1f}/10</div>
                    <div style="color:#9aa0a6; font-size: 0.7rem;">How stable growth is across four time windows</div>
                    <div style="color:#FFFFFF; font-size: 0.7rem;">{_ws.consistency_reason}</div>                    
                </div>
            </div>
            <div style="margin-top: 0.9rem; font-size: 0.8rem; color: #c5c8cd;">
                <div><strong style="color:#4CAF50;">Strengths:</strong> {_strength_html}</div>
                <div style="margin-top: 0.2rem;"><strong style="color:#EF5350;">Weaknesses:</strong> {_weakness_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # VALUATION (Intrinsic Value Section)
    # ============================================================
    st.markdown("### Valuation")
    
    v1, v2, v3, v4, v5 = st.columns(5)
    v1.metric("Value Price / Intrinsic Value", _fmt_money(val.value_price, fin.exchange, fin.ticker) if val else "n/a")
    v2.metric("Margin of Safety (MOS) Buy Price", _fmt_money(val.mos_price, fin.exchange, fin.ticker) if val else "n/a")
    v3.metric("Current Price", _fmt_money(fin.current_price, fin.exchange, fin.ticker))
    _render_verdict_cell(v4, v_verdict, v_detail, v_color)
    _render_analyst_cell(v5, fin)

    # --- FIX: Everything in one row with proper sizing ---
    growth_group_col, button_col = st.columns([4, 1])
    
    with growth_group_col:
        st.markdown(
            "<div style='font-size: 0.875rem; margin-bottom: 0.25rem;'>Growth source</div>",
            unsafe_allow_html=True,
        )

        col_radio, col_input, col_pct, col_btn, col_spacer = st.columns([2.1, 1.1, 0.2, 1.0, 1.8])

        with col_radio:
            selected_growth_mode = st.radio(
                "Growth source",
                options=mode_options,
                horizontal=True,
                format_func=_valuation_growth_rate_label,
                key="valuation_growth_mode",
                label_visibility="collapsed",
            )

        with col_input:
            st.markdown(
                """
                <style>
                [data-testid="stNumberInput"] {
                    width: 100% !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height: 36px;'></div>", unsafe_allow_html=True)
            st.number_input(
                "Custom %",
                key="custom_growth_rate",
                min_value=1,
                max_value=100,
                step=1,
                format="%d",
                disabled=(selected_growth_mode != "Custom"),
                label_visibility="collapsed",
            )

        with col_pct:
            st.markdown(
                "<div style='height: 44px;'></div><div style='font-size: 0.875rem; color: #9aa0a6;'>%</div>",
                unsafe_allow_html=True,
            )

        with col_btn:
            st.markdown("<div style='height: 36px;'></div>", unsafe_allow_html=True)
            revalue_button = st.button(
                "RE-VALUATE", 
                key="revaluate_btn",
                help="Recalculate with selected growth rate"
            )

    st.caption(f"DEBUG — mode: {selected_growth_mode} | custom box: {st.session_state.get('custom_growth_rate')}")

    if revalue_button:
        st.session_state["valuation_growth_result"] = _compute_valuation_value(selected_growth_mode)
        st.session_state["valuation_calculated_mode"] = selected_growth_mode
        st.rerun()

    # Only show the "How it was calculated" expander if val actually exists
    if val is not None:
        with st.expander("How the value Price was calculated"):
            _big5_g_txt = _fmt_pct(big5_eps_g)
            _analyst_g_txt = _fmt_pct(fin.analyst_5yr_growth)
            _hist_pe_txt = f"{fin.pe_ratio_ttm:.1f}" if fin.pe_ratio_ttm else "no data"
            _default_pe = val.growth_rate * 100 * 2
            _mos_pct_int = int(round(100 * (1 - val.mos_price / val.value_price))) if val.value_price else 50

            st.markdown(
                f"""
    **Step 1 — Pick the growth rate (the lower of two, capped at 15%)**
    - Big 5 EPS growth (median of 5yr/3yr): **{_big5_g_txt}**
    - Analyst 5yr growth: **{_analyst_g_txt}**
    - Chosen: **{_fmt_pct(val.growth_rate)}** *(source: {val.growth_source}, capped at 15%)*

    **Step 2 — Estimate future P/E (the lower of two)**
    - Historical TTM P/E: **{_hist_pe_txt}**
    - 2 × growth rate: **{_default_pe:.1f}**
    - Chosen future P/E: **{val.future_pe:.2f}**

    **Step 3 — Project EPS 10 years forward**
    - Current EPS × (1 + growth)^10
    - = {val.current_eps:.2f} × (1 + {val.growth_rate:.4f})^10
    - = **{val.future_eps:.2f}**

    **Step 4 — Future stock price**
    - Future EPS × Future P/E
    - = {val.future_eps:.2f} × {val.future_pe:.2f}
    - = **{_fmt_money(val.future_price, fin.exchange, fin.ticker)}**

    **Step 5 — Discount back to today at {_fmt_pct(val.discount_rate)} (Buffett's required return)**
    - Future price ÷ (1 + {val.discount_rate:.2f})^{val.horizon_years}
    - = **{_fmt_money(val.value_price, fin.exchange, fin.ticker)}**  ← *Value Price*

    **Step 6 — Apply {_mos_pct_int}% Margin of Safety**
    - value × {1 - _mos_pct_int/100:.2f}
    - = **{_fmt_money(val.mos_price, fin.exchange, fin.ticker)}**  ← *MOS Buy Price*
    """.strip()
            )

    st.markdown("---")

    # ============================================================
    # DCF METHOD VALUATION
    # ============================================================
    st.markdown("#### DCF Method Valuation")
    st.caption(
        "Discounted Cash Flow — projects free cash flow forward and discounts "
        "it back to today. Adjust the two assumptions below; only this section "
        "reacts to the sliders."
    )
    dcf_col1, dcf_col2 = st.columns(2)
    with dcf_col1:
        dcf_discount = st.slider(
            "DCF discount rate",
            min_value=6.0,
            max_value=15.0,
            value=10.0,
            step=0.5,
            format="%.1f%%",
            help="Required annual return. 10% = long-run S&P 500 average; 15% = Buffett's aggressive rate.",
            key="dcf_discount_slider",
        ) / 100
    with dcf_col2:
        dcf_terminal = st.slider(
            "DCF terminal growth",
            min_value=0.0,
            max_value=4.0,
            value=2.5,
            step=0.5,
            format="%.1f%%",
            help="Perpetual growth rate after the fade period. Should be <= long-run GDP growth (~2.5-3%).",
            key="dcf_terminal_slider",
        ) / 100

    st.caption(f"DEBUG: discount={dcf_discount:.4f}, terminal={dcf_terminal:.4f}")

    # Run the two-stage DCF with these inputs
    _fcf_ttm_dcf = float(fin.free_cash_flow.iloc[-1]) if not fin.free_cash_flow.empty else None
    _growth_dcf = big5_eps_g if big5_eps_g is not None else fin.analyst_5yr_growth
    _dcf_val = dcf_two_stage(
        fcf_ttm=_fcf_ttm_dcf,
        shares_out=fin.shares_outstanding,
        current_price=fin.current_price,
        growth_rate=_growth_dcf,
        discount_rate=dcf_discount,
        terminal_growth=dcf_terminal,
    )
    st.caption(f"DEBUG: dcf fair_value={_dcf_val.fair_value if _dcf_val else None}")
    if _dcf_val is None:
        st.info("DCF not computable — needs positive TTM free cash flow and shares outstanding.")
    else:
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("DCF Fair Value", _fmt_money(_dcf_val.fair_value, fin.exchange, fin.ticker))
        with d2:
            st.metric("DCF MOS Buy Price", _fmt_money(_dcf_val.mos_price, fin.exchange, fin.ticker))
        with d3:
            st.metric(
                "DCF Upside",
                _fmt_pct(_dcf_val.upside_pct) if _dcf_val.upside_pct is not None else "n/a",
            )
        with d4:
            st.metric("DCF Verdict", _dcf_val.verdict)

        with st.expander("How the DCF was calculated"):
            st.markdown(
                f"""
                **Inputs**
                - TTM Free Cash Flow: **{_fmt_money(_fcf_ttm_dcf, fin.exchange, fin.ticker)}**
                - Shares outstanding: **{fin.shares_outstanding:,.0f}** *(from yfinance)*
                - Projection growth rate: **{_fmt_pct(_growth_dcf)}**
                - Discount rate (WACC proxy): **{_fmt_pct(dcf_discount)}**
                - Terminal growth: **{_fmt_pct(dcf_terminal)}**

                **Method**
                Two-stage DCF: FCF grows at the projection rate for a fade period,
                then settles into perpetual terminal growth. All cash flows are
                discounted at the required return rate. Fair value is divided by
                shares outstanding to get a per-share price. MOS Buy Price applies
                a 25% margin of safety.
                """
            )

    st.markdown("---")

    # ============================================================
    # PRICE-TO-SALES (P/S) FALLBACK
    # ============================================================
    if current_eps is None or current_eps <= 0:
        st.markdown("#### Price-to-Sales (P/S) Valuation")
        st.caption(
            "Since this company has negative earnings or missing data, we use a Price-to-Sales multiple "
            "to estimate fair value. This is the standard Wall Street method for pre-profit high-growth companies."
        )
        
        _revenue_ttm = float(fin.revenue.iloc[-1]) if not fin.revenue.empty else None
        _shares = fin.shares_outstanding
        
        if _revenue_ttm and _revenue_ttm > 0 and _shares and _shares > 0:
            _sales_per_share = _revenue_ttm / _shares
            
            ps_col1, ps_col2 = st.columns(2)
            with ps_col1:
                _fair_ps_multiple = st.slider(
                    "Fair P/S Multiple", 
                    min_value=1.0, max_value=30.0, value=10.0, step=1.0, 
                    help="A 10x multiple means you are willing to pay $10 for every $1 of revenue."
                )
            
            _fair_value_ps = _sales_per_share * _fair_ps_multiple
            _mos_ps = _fair_value_ps * 0.75  # 25% margin of safety
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Revenue (TTM)", _fmt_money(_revenue_ttm, fin.exchange, fin.ticker))
            col2.metric("Sales Per Share", _fmt_money(_sales_per_share, fin.exchange, fin.ticker))
            col3.metric("Fair Value (P/S)", _fmt_money(_fair_value_ps, fin.exchange, fin.ticker))
            col4.metric("P/S MOS Buy", _fmt_money(_mos_ps, fin.exchange, fin.ticker))
            
            if fin.current_price:
                _upside_ps = (_fair_value_ps - fin.current_price) / fin.current_price
                if _upside_ps > 0:
                    st.success(f"Stock is **{_upside_ps*100:.1f}% below** its P/S fair value.")
                else:
                    st.warning(f"Stock is **{abs(_upside_ps)*100:.1f}% above** its P/S fair value.")
        else:
            st.info("Cannot calculate P/S — missing Revenue or Shares Outstanding.")
            
        st.markdown("---")

    # ============================================================
    # OTHER INTRINSIC VALUE METHODS
    # ============================================================
    # Assumptions for the other methods (Graham Formula and Lynch/Graham/PEG MOS)
    with st.expander("⚙️ Assumptions for the other methods", expanded=False):
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            aaa_yield = st.slider(
                "AAA corporate bond yield (Graham Formula)",
                min_value=2.0,
                max_value=10.0,
                value=4.5,
                step=0.5,
                format="%.1f%%",
                help="Used in Graham's revised 1974 formula.",
                key="aaa_yield_slider",
            ) / 100
        with a_col2:
            mos_pct = st.slider(
                "Margin of Safety (Lynch/Graham/PEG)",
                min_value=10.0,
                max_value=60.0,
                value=25.0,
                step=5.0,
                format="%.1f%%",
                help="Discount applied to fair value to get a buy price.",
                key="mos_pct_slider",
            ) / 100

    _tip_other = _VERDICT_TOOLTIPS.get("BUY", "").replace('"', "&quot;")
    st.markdown(
        f"""
        <h3 style="margin-bottom: 0.25rem;">
            Other Intrinsic Value Methods
            <span title="{_tip_other}" style="cursor: help; color: #4CAF50; font-size: 0.9rem; margin-left: 6px;">&#9432;</span>
        </h3>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Multiple valuation lenses on the same ticker. Each shows fair value, a margin-of-safety "
        "buy price, upside vs current, and a verdict. No single method is right — look for consensus. "
        "Hover the ⓘ icon for how to interpret the verdicts."
    )

    _fcf_ttm_for_dcf = float(fin.free_cash_flow.iloc[-1]) if not fin.free_cash_flow.empty else None
    _growth_for_intrinsic = big5_eps_g if big5_eps_g is not None else fin.analyst_5yr_growth

    methods: list[MethodResult] = []

    def _add(m: MethodResult | None):
        if m is not None:
            methods.append(m)

    _add(
        dcf_two_stage(
            fcf_ttm=_fcf_ttm_for_dcf,
            shares_out=fin.shares_outstanding,
            current_price=fin.current_price,
            growth_rate=_growth_for_intrinsic,
            discount_rate=dcf_discount,
            terminal_growth=dcf_terminal,
        )
    )
    _add(
        peter_lynch_fair(
            current_eps=eps_for_valuation,
            growth_rate=_growth_for_intrinsic,
            dividend_yield=fin.dividend_yield,
            current_price=fin.current_price,
            mos=mos_pct,
        )
    )
    _add(
        graham_number(
            current_eps=eps_for_valuation,
            book_value_per_share=fin.book_value_per_share,
            current_price=fin.current_price,
            mos=mos_pct,
        )
    )
    _add(
        graham_formula(
            current_eps=eps_for_valuation,
            growth_rate=_growth_for_intrinsic,
            current_price=fin.current_price,
            aaa_bond_yield=aaa_yield,
            mos=mos_pct,
        )
    )
    _add(
        peg_fair_value(
            current_eps=eps_for_valuation,
            growth_rate=_growth_for_intrinsic,
            current_price=fin.current_price,
            mos=mos_pct,
        )
    )

    if not methods:
        st.info(
            "No intrinsic-value methods computable — need positive current EPS and at least "
            "one growth estimate (Big 5 EPS or analyst)."
        )
    else:
        def _verdict_with_pct(m: MethodResult) -> str:
            """Append a small "(±NN%)" hint to the verdict cell.

            Sign convention: positive % = current price is BELOW fair (upside),
            negative = current price is ABOVE fair (overvalued).
            """
            if m.upside_pct is None:
                return m.verdict
            pct = m.upside_pct * 100
            if m.verdict == "BUY":
                return f"{m.verdict} ({pct:+.0f}% below fair)"
            if m.verdict == "WATCH":
                return f"{m.verdict} ({pct:+.0f}% vs fair)"
            if m.verdict == "AVOID":
                return f"{m.verdict} ({-pct:.0f}% above fair)"
            return m.verdict

        def _row(m: MethodResult) -> dict:
            return {
                "Method": m.name,
                "Fair Value": _fmt_money(m.fair_value, fin.exchange, fin.ticker),
                "MOS Buy": _fmt_money(m.mos_price, fin.exchange, fin.ticker),
                "Upside vs Current": _fmt_pct(m.upside_pct),
                "Verdict": _verdict_with_pct(m),
            }

        methods_df = pd.DataFrame([_row(m) for m in methods])

        def _style_row(row):
            v = row["Verdict"]
            if v.startswith("BUY"):
                return ["background-color: #1e4620; color: #b6f0b6;"] * len(row)
            if v.startswith("WATCH"):
                return ["background-color: #4a3a1e; color: #f0d8a6;"] * len(row)
            if v.startswith("AVOID"):
                return ["background-color: #4b1e1e; color: #f0b6b6;"] * len(row)
            return [""] * len(row)

        st.dataframe(
            methods_df.style.apply(_style_row, axis=1),
            width="stretch",
            hide_index=True,
        )

        # Consensus verdict.
        buys = sum(1 for m in methods if m.verdict == "BUY")
        watches = sum(1 for m in methods if m.verdict == "WATCH")
        avoids = sum(1 for m in methods if m.verdict == "AVOID")
        total = buys + watches + avoids
        if total:
            if buys >= total / 2:
                st.success(f"Consensus: **BUY** ({buys}/{total} methods say below MOS)")
            elif avoids >= total / 2:
                st.error(f"Consensus: **AVOID** ({avoids}/{total} methods say above fair value)")
            else:
                st.warning(f"Consensus: **WATCH** ({buys} buy / {watches} watch / {avoids} avoid)")

        with st.expander("Per-method assumptions"):
            for m in methods:
                st.markdown(f"**{m.name}**")
                st.write(m.assumptions)

    st.markdown("### Health checks")
    fcf_ttm = float(fin.free_cash_flow.iloc[-1]) if not fin.free_cash_flow.empty else None
    ltd_ttm = float(fin.long_term_debt.iloc[-1]) if not fin.long_term_debt.empty else None
    big5_growth_for_payback = big5_eps_g if big5_eps_g is not None else fin.analyst_5yr_growth

    pay = payback_time(fcf_ttm, big5_growth_for_payback, fin.market_cap)
    dtf = debt_to_fcf(ltd_ttm, fcf_ttm)

    h1, h2 = st.columns(2)
    with h1:
        if pay is None:
            st.metric("Payback Time", "n/a")
            st.caption("Needs positive TTM FCF and a valid market cap.")
        else:
            st.metric("Payback Time", f"{pay:.0f} years", delta=("PASS" if pay <= 8 else "FAIL"))
            st.caption("Buffett's rule: 8 years or fewer.")
    with h2:
        if dtf is None:
            st.metric("Debt / FCF", "n/a")
            st.caption("Needs positive TTM FCF.")
        else:
            st.metric("Debt / FCF", f"{dtf:.2f}", delta=("PASS" if dtf < 3 else "FAIL"))
            st.caption("Buffett's rule: less than 3 years to pay off all long-term debt.")

    st.markdown("### Underlying data (annual)")
    with st.expander("Show raw statement values used"):
        frames = {
            "Revenue": fin.revenue,
            "Net Income": fin.net_income,
            "EPS (Diluted)": fin.eps,
            "Equity": fin.equity,
            "Long-Term Debt": fin.long_term_debt,
            "Operating Cash Flow": fin.operating_cash_flow,
            "CapEx": fin.capex,
            "Free Cash Flow": fin.free_cash_flow,
            "ROIC (per year)": big5.roic_by_year,
        }
        combined = pd.DataFrame({k: v for k, v in frames.items() if not v.empty})
        combined.index.name = "Year"
        st.dataframe(combined.sort_index(ascending=False), width="stretch")

    st.divider()

    # Tiny disclaimer-style footer. Uses raw HTML so we can shrink below Streamlit's
    # default caption font (which is already small but not "fine print" small).
    _split_line = (
        "EDGAR reports EPS as-filed (not split-adjusted); large year-over-year "
        "EPS jumps may reflect a stock split rather than an earnings change. "
        if _show_split_warning else ""
    )
    st.markdown(
        f"""
        <div style="color: #6c6f75; font-size: 0.72rem; line-height: 1.5; margin-top: 0.5rem;">
            <em>
            {fin.data_source_note}
            {_split_line}
            Not investment advice. Cross-check with 10-K filings before any capital decision.
            </em>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_technical_analysis():
    """Technical Analysis tab - Trading signals based on technical indicators"""
    st.header("📊 Technical Analysis - Trading Signals")
    st.caption("Technical indicators for identifying potential entry and exit points")
    
    # Input section
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        tech_ticker = st.text_input("Enter Stock Ticker for Technical Analysis:", "AAPL").upper()
    with col2:
        tech_period = st.selectbox("Analysis Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    with col3:
        tech_analyze = st.button("📊 Analyze Technicals", type="primary", key="tech_analyze")
    
    if tech_analyze or tech_ticker:
        with st.spinner(f"Analyzing {tech_ticker} technical indicators..."):
            # Fetch
            analyzer = TechnicalIndicatorAnalyzer(tech_ticker, 'max')            

            if not analyzer.fetch_data():
                st.error(f"Could not fetch data for {tech_ticker}. Please check the ticker symbol.")
                return

            analyzer.calculate_all_indicators()
            analyzer.get_trading_signals()
            summary = analyzer.get_summary_dict()

            # Get verdict data
            verdict_data = analyzer.get_verdict()

            # Define volume_metrics early so it's available throughout the render
            volume_metrics = verdict_data.get('volume_metrics', {})
            
            # --- Display Key Indicators Table FIRST ---
            st.markdown("### Key Indicators")
            
            key_indicators = verdict_data['key_indicators']
            ind = analyzer.indicators
            price = verdict_data['current_price']
            
            # Display as a styled dataframe
            indicator_data = {
                'Indicator': ['RSI', 'MACD', 'Stochastic %K', 'Stochastic %D',
                              'SMA 50', 'SMA 200', 'Bollinger Bands', 'Volume'],
                'Value': [
                    f"{key_indicators['rsi']:.1f}",
                    f"{key_indicators['macd']:.4f}",
                    f"{key_indicators['stoch_k']:.1f}",
                    f"{key_indicators['stoch_d']:.1f}",
                    f"${key_indicators['sma_50']:.2f}" if key_indicators['sma_50'] else "N/A",
                    f"${key_indicators['sma_200']:.2f}" if key_indicators['sma_200'] else "N/A",
                    f"${ind['BB_Lower']:.2f} - ${ind['BB_Upper']:.2f}" if ind['BB_Lower'] and ind['BB_Upper'] else "N/A",
                    f"{volume_metrics.get('volume_ratio', 0):.2f}× avg" if volume_metrics else "N/A",
                ],
                'Status': []
            }
            
            # 1. RSI Status
            rsi = key_indicators['rsi']
            if rsi < 30:
                indicator_data['Status'].append('🟢 Oversold (Bullish)')
            elif rsi > 70:
                indicator_data['Status'].append('🔴 Overbought (Bearish)')
            elif 30 <= rsi <= 40:
                indicator_data['Status'].append('🟡 Approaching Oversold')
            elif 60 <= rsi <= 70:
                indicator_data['Status'].append('🟡 Approaching Overbought')
            else:
                indicator_data['Status'].append('🟡 Neutral')
            
            # 2. MACD Status
            macd = key_indicators['macd']
            macd_hist = ind['MACD_Histogram']
            if macd > 0 and macd_hist > 0:
                indicator_data['Status'].append('📈 Bullish (Increasing Momentum)')
            elif macd > 0 and macd_hist < 0:
                indicator_data['Status'].append('🟡 Bullish (Weakening)')
            elif macd < 0 and macd_hist < 0:
                indicator_data['Status'].append('📉 Bearish (Increasing Momentum)')
            elif macd < 0 and macd_hist > 0:
                indicator_data['Status'].append('🟡 Bearish (Weakening)')
            else:
                indicator_data['Status'].append('🟡 Neutral')
            
            # 3. Stochastic %K Status
            stoch_k = key_indicators['stoch_k']
            stoch_d = key_indicators['stoch_d']
            if stoch_k < 20 and stoch_d < 20:
                if stoch_k > stoch_d:
                    indicator_data['Status'].append('🟢 Oversold (Crossover Up)')
                else:
                    indicator_data['Status'].append('🟢 Oversold')
            elif stoch_k > 80 and stoch_d > 80:
                if stoch_k < stoch_d:
                    indicator_data['Status'].append('🔴 Overbought (Crossover Down)')
                else:
                    indicator_data['Status'].append('🔴 Overbought')
            else:
                indicator_data['Status'].append('🟡 Neutral')
            
            # 4. Stochastic %D Status
            if stoch_d < 20:
                indicator_data['Status'].append('🟢 Oversold')
            elif stoch_d > 80:
                indicator_data['Status'].append('🔴 Overbought')
            else:
                indicator_data['Status'].append('🟡 Neutral')
            
            # 5. SMA 50 Status
            sma_50 = key_indicators['sma_50']
            sma_200 = key_indicators['sma_200']
            if sma_50 and sma_200:
                if sma_50 > sma_200:
                    if price > sma_50:
                        indicator_data['Status'].append('📈 Above 200 (Golden Cross)')
                    else:
                        indicator_data['Status'].append('📈 Above 200 (Pullback)')
                else:
                    if price < sma_50:
                        indicator_data['Status'].append('📉 Below 200 (Death Cross)')
                    else:
                        indicator_data['Status'].append('📉 Below 200 (Bounce)')
            else:
                indicator_data['Status'].append('N/A')
            
            # 6. SMA 200 Status
            if sma_200:
                if price > sma_200:
                    percent_above = ((price - sma_200) / sma_200) * 100
                    if percent_above > 20:
                        indicator_data['Status'].append(f'📈 Above (+{percent_above:.0f}%)')
                    else:
                        indicator_data['Status'].append(f'📈 Above ({percent_above:.0f}%)')
                else:
                    percent_below = ((sma_200 - price) / sma_200) * 100
                    if percent_below > 20:
                        indicator_data['Status'].append(f'📉 Below (-{percent_below:.0f}%)')
                    else:
                        indicator_data['Status'].append(f'📉 Below ({percent_below:.0f}%)')
            else:
                indicator_data['Status'].append('N/A')
            
            # 7. Bollinger Bands Status
            if ind['BB_Lower'] and ind['BB_Upper'] and ind['current_price']:
                bb_pos = ((ind['current_price'] - ind['BB_Lower']) / (ind['BB_Upper'] - ind['BB_Lower'])) * 100
                bb_width = ind['BB_Upper'] - ind['BB_Lower']
                
                if bb_pos < 10:
                    indicator_data['Status'].append('🟢 Extreme Lower (Strong Oversold)')
                elif bb_pos < 20:
                    indicator_data['Status'].append('🟢 Near Lower (Oversold)')
                elif bb_pos > 90:
                    indicator_data['Status'].append('🔴 Extreme Upper (Strong Overbought)')
                elif bb_pos > 80:
                    indicator_data['Status'].append('🔴 Near Upper (Overbought)')
                elif 40 <= bb_pos <= 60:
                    indicator_data['Status'].append('🟡 Middle (Neutral)')
                else:
                    indicator_data['Status'].append('🟡 Mid-Range')
                
                # Add bandwidth info if available
                if bb_width and bb_width > 0:
                    avg_price = (ind['BB_Upper'] + ind['BB_Lower']) / 2
                    bandwidth_pct = (bb_width / avg_price) * 100
                    if bandwidth_pct > 20:
                        indicator_data['Status'][-1] += ' 🔸 Wide Volatility'
                    elif bandwidth_pct < 10:
                        indicator_data['Status'][-1] += ' 🔹 Narrow (Squeeze)'
            else:
                indicator_data['Status'].append('N/A')

            # Volume status
            if volume_metrics:
                sev = volume_metrics.get('severity', 'normal')
                direction = volume_metrics.get('price_direction', 'FLAT')
                if sev in ('spike', 'extreme'):
                    if direction == 'UP':
                        indicator_data['Status'].append('🔵 Spike + Price Up (Bullish)')
                    elif direction == 'DOWN':
                        indicator_data['Status'].append('🔴 Spike + Price Down (Bearish)')
                    else:
                        indicator_data['Status'].append('🟠 Spike, Flat Price (Indecision)')
                elif sev == 'elevated':
                    indicator_data['Status'].append('🟢 Elevated Volume')
                elif sev in ('low', 'very_low'):
                    indicator_data['Status'].append('⚪ Low Volume (Weak Conviction)')
                else:
                    indicator_data['Status'].append('🟡 Normal')
            else:
                indicator_data['Status'].append('N/A')
                        
            # Create DataFrame and style it
            df_indicators = pd.DataFrame(indicator_data)
            
            # Style the dataframe to match Stock Analyzer theme
            st.dataframe(
                df_indicators.style.set_properties(**{
                    'background-color': 'rgba(255,255,255,0.05)',
                    'border-color': 'rgba(255,255,255,0.1)',
                    'color': '#e8eaed'
                }).set_table_styles([
                    {'selector': 'thead th', 'props': [('background-color', '#1e1e1e'), ('color', '#b6f0b6')]}
                ]),
                width="stretch",
                hide_index=True
            )
            
            # --- Display Verdict (moved below Key Indicators table) ---
            st.markdown("### Verdict")
            
            # Create verdict display with color matching Stock Analyzer
            verdict_color = verdict_data['color']
            verdict_icon = verdict_data['icon']
            verdict_text = verdict_data['verdict']
            confidence = verdict_data['confidence']
            
            # Color mapping to match Stock Analyzer
            color_map = {
                'bargain': {'bg': '#1e4620', 'text': '#b6f0b6', 'border': '#00E676'},
                'green': {'bg': '#1e4620', 'text': '#b6f0b6', 'border': '#4CAF50'},
                'orange': {'bg': '#4a3a1e', 'text': '#f0d8a6', 'border': '#FFA726'},
                'red': {'bg': '#4b1e1e', 'text': '#f0b6b6', 'border': '#EF5350'},
                'gray': {'bg': '#2a2a2a', 'text': '#9aa0a6', 'border': '#9AA0A6'}
            }
            
            colors = color_map.get(verdict_color, color_map['gray'])
            
            # Get contributing signals
            contributing_signals = verdict_data.get('contributing_signals', [])
            buy_signals = verdict_data.get('buy_signals', [])
            sell_signals = verdict_data.get('sell_signals', [])
            
            # Create verdict card matching Stock Analyzer style
            st.markdown(f"""
            <div style="
                background-color: {colors['bg']};
                border: 2px solid {colors['border']};
                border-radius: 10px;
                padding: 1.5rem;
                margin: 1rem 0;
            ">
                <div style="display: flex; align-items: flex-start; gap: 1rem;">
                    <div style="font-size: 3rem;">{verdict_icon}</div>
                    <div style="flex: 1;">
                        <div style="font-size: 2rem; font-weight: 700; color: {colors['text']};">
                            {verdict_text}
                            <span style="font-size: 1rem; font-weight: 400; color: #9aa0a6;">
                                (Confidence: {confidence:.0f}%)
                            </span>
                        </div>
                        <div style="color: #9aa0a6; font-size: 0.9rem; margin-top: 0.25rem;">
                            {verdict_data['detail']}
                        </div>
            """, unsafe_allow_html=True)
            
            # Add contributing signals
            if contributing_signals:
                st.markdown(f"""
                        <div style="margin-top: 0.75rem;">
                            <div style="color: #9aa0a6; font-size: 0.8rem; margin-bottom: 0.25rem;">
                                <strong>Key contributing indicators:</strong>
                            </div>
                            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                """, unsafe_allow_html=True)
                
                # Display each contributing signal as a badge
                for signal in contributing_signals:
                    # Determine badge color based on signal type
                    if "bullish" in signal.lower() or "above" in signal.lower() or "oversold" in signal.lower():
                        badge_color = "#4CAF50"  # Green for bullish
                    elif "bearish" in signal.lower() or "below" in signal.lower() or "overbought" in signal.lower():
                        badge_color = "#EF5350"  # Red for bearish
                    else:
                        badge_color = "#FFA726"  # Orange for neutral/wait
                    
                    # Shorten long signals for display
                    display_signal = signal
                    if len(signal) > 60:
                        display_signal = signal[:57] + "..."
                    
                    st.markdown(f"""
                        <span style="
                            background-color: rgba(255,255,255,0.08);
                            border-left: 3px solid {badge_color};
                            padding: 0.3rem 0.6rem;
                            border-radius: 4px;
                            font-size: 0.75rem;
                            color: #e8eaed;
                            display: inline-block;
                            margin: 2px 0;
                        ">
                            {display_signal}
                        </span>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                            </div>
                        </div>
                """, unsafe_allow_html=True)
            
            # Close the div
            st.markdown("""
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- Add expandable section for all signals ---
            with st.expander("📋 See all signals that contributed to this verdict"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🟢 Buy Signals**")
                    if buy_signals:
                        for signal in buy_signals:
                            st.markdown(f"<span style='color: #b6f0b6;'>✅ {signal}</span>", unsafe_allow_html=True)
                    else:
                        st.info("No buy signals detected")
                
                with col2:
                    st.markdown("**🔴 Sell Signals**")
                    if sell_signals:
                        for signal in sell_signals:
                            st.markdown(f"<span style='color: #f0b6b6;'>❌ {signal}</span>", unsafe_allow_html=True)
                    else:
                        st.info("No sell signals detected")
                
                # Show neutral signals if any
                neutral_signals = verdict_data.get('neutral_signals', [])
                if neutral_signals:
                    st.markdown("**⏳ Neutral / Wait Signals**")
                    for signal in neutral_signals[:5]:  # Show first 5 to keep it clean
                        st.markdown(f"<span style='color: #f0d8a6;'>⏳ {signal}</span>", unsafe_allow_html=True)
                    if len(neutral_signals) > 5:
                        st.caption(f"... and {len(neutral_signals) - 5} more neutral signals")

            # ============================================================
            # VOLUME ANALYSIS (Levels 1-4)
            # ============================================================
            #volume_metrics = verdict_data.get('volume_metrics', {})

            if volume_metrics:
                st.markdown("### 📊 Volume Analysis")

                vm = volume_metrics
                severity = vm.get('severity', 'normal')
                interpretation = vm.get('interpretation', '')
                statistically_significant = vm.get('statistically_significant', False)

                # Color mapping
                if severity in ('spike', 'extreme'):
                    sev_color, sev_bg = '#FFA726', '#4a3a1e'
                elif severity == 'elevated':
                    sev_color, sev_bg = '#8BC34A', '#2a3d1e'
                elif severity in ('low', 'very_low'):
                    sev_color, sev_bg = '#9AA0A6', '#2a2a2a'
                else:
                    sev_color, sev_bg = '#4CAF50', '#1e4620'

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown(f"""
                    <div style="background-color:{sev_bg}; border:2px solid {sev_color};
                                border-radius:10px; padding:1rem; text-align:center;">
                        <div style="font-size:1.25rem; font-weight:700; color:{sev_color};">
                            {vm['classification']}
                        </div>
                        <div style="color:#9aa0a6; font-size:0.8rem; margin-top:0.25rem;">
                            {vm['volume_ratio']:.2f}× 20-day avg
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"**Interpretation**: {interpretation}")

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Today's Volume", f"{vm['latest_volume']:,.0f}")
                    mc2.metric("20-day Avg", f"{vm['avg_volume_20d']:,.0f}")
                    mc3.metric(
                        "Z-Score",
                        f"{vm['volume_zscore']:+.2f}",
                        delta="Significant" if statistically_significant else "Normal",
                        delta_color="normal" if statistically_significant else "off",
                    )

                    if statistically_significant:
                        st.caption(
                            f"🔬 Volume is {vm['volume_zscore']:.1f} standard deviations above the "
                            f"60-day mean — statistically unusual for this stock."
                        )

                # Show volume weight applied to the signal
                volume_weight = summary['signals'].get('volume_weight', 1.0)
                if abs(volume_weight - 1.0) > 0.1:
                    direction = "increased" if volume_weight > 1.0 else "reduced"
                    st.info(
                        f"⚖️ Signal confidence {direction} to "
                        f"**{volume_weight:.2f}×** due to volume conditions."
                    )
            
            # --- Key Levels (similar to Stock Analyzer) ---
            st.markdown("### Key Levels")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${verdict_data['current_price']:.2f}")
            with col2:
                support = verdict_data['support_level']
                if support:
                    st.metric("Support Level", f"${support:.2f}", 
                             delta=f"{((verdict_data['current_price'] - support) / verdict_data['current_price'] * 100):.1f}% above")
                else:
                    st.metric("Support Level", "N/A")
            with col3:
                resistance = verdict_data['resistance_level']
                if resistance:
                    st.metric("Resistance Level", f"${resistance:.2f}",
                             delta=f"{((resistance - verdict_data['current_price']) / verdict_data['current_price'] * 100):.1f}% above")
                else:
                    st.metric("Resistance Level", "N/A")

                        # --- Suggested entry price (from inverting the technical verdict) ---
            st.markdown("### 🎯 Technical BUY Price")

            entry_info = analyzer.find_entry_price()

            if entry_info is None:
                st.info("Not enough data to compute a BUY price.")
            elif entry_info["entry_price"] is None:
                st.warning(
                    f"No BUY price found even at 60% below the current price. "
                    f"The verdict remains {entry_info['entry_verdict']} at "
                    f"${entry_info['search_range'][0]:.2f}."
                )
            else:
                tgt = entry_info["entry_price"]
                cur = entry_info["current_price"]
                drop_pct = (tgt - cur) / cur * 100

                # Color depends on how close the current price is to the target
                if drop_pct >= -0.5:
                    # Already there — bright green
                    box_bg, box_color = "#1e4620", "#00E676"
                    status_text = "READY NOW"
                elif drop_pct >= -3.0:
                    # Close — light green
                    box_bg, box_color = "#2a3d1e", "#8BC34A"
                    status_text = "NEAR ENTRY"
                elif drop_pct >= -8.0:
                    # Moderate pullback needed — amber
                    box_bg, box_color = "#4a3a1e", "#FFA726"
                    status_text = "WAIT FOR PULLBACK"
                else:
                    # Deep pullback needed — red
                    box_bg, box_color = "#4b1e1e", "#EF5350"
                    status_text = "DEEP PULLBACK NEEDED"

                # Big highlighted card
                st.markdown(f"""
                <div style="background-color:{box_bg}; border:2px solid {box_color};
                            border-radius:10px; padding:1.5rem; text-align:center;
                            margin:0.5rem 0;">
                    <div style="color:#9aa0a6; font-size:0.75rem;
                                text-transform:uppercase; letter-spacing:0.05em;
                                margin-bottom:0.35rem;">
                        Technical BUY Price
                    </div>
                    <div style="font-size:3rem; font-weight:700; color:{box_color};
                                line-height:1.1;">
                        ${tgt:.2f}
                    </div>
                    <div style="color:{box_color}; font-size:1rem;
                                font-weight:600; margin-top:0.35rem;">
                        {status_text}
                    </div>
                    <div style="color:#9aa0a6; font-size:0.85rem; margin-top:0.25rem;">
                        {drop_pct:+.2f}% from current (${cur:.2f})
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if drop_pct >= -0.5:
                    st.success(
                        f"✅ Price is essentially at the BUY level. "
                        f"A marginal move down flips the verdict to BUY."
                    )
                else:
                    st.info(
                        f"⏳ Wait for a pullback to ${tgt:.2f} before the "
                        f"technical rules flip to BUY."
                    )

            # --- 52-week range ---
            st.markdown("### 52-Week Range")
            try:
                _df = analyzer.stock_data
                if _df is not None and len(_df) >= 200:
                    _high_52 = float(_df["High"].tail(252).max())
                    _low_52 = float(_df["Low"].tail(252).min())
                    _cur = float(verdict_data["current_price"])

                    if _high_52 > _low_52 and _cur == _cur:
                        _pos_pct = max(0.0, min(100.0,
                            (_cur - _low_52) / (_high_52 - _low_52) * 100))

                        st.markdown(
                            f"""
                            <div style="margin: 0.5rem 0 1rem;">
                                <div style="display:flex; justify-content:space-between;
                                            color:#9aa0a6; font-size:0.85rem;">
                                    <span>Low ${_low_52:,.2f}</span>
                                    <span>Current <strong style="color:#e8eaed;">${_cur:,.2f}</strong></span>
                                    <span>High ${_high_52:,.2f}</span>
                                </div>
                                <div style="position:relative; height:8px; background:rgba(255,255,255,0.08); border-radius:4px; margin-top:0.5rem; overflow:hidden;">
                                    <div style="position:absolute; left:0; top:0; width:{_pos_pct:.1f}%; height:100%; background:#4DABF7; border-radius:4px 0 0 4px;"></div>
                                    <div style="position:absolute; left:{_pos_pct:.1f}%; top:-4px; width:3px; height:16px; background:#00E676; border-radius:2px; transform:translateX(-50%);"></div>
                                </div>
                                <div style="color:#9aa0a6; font-size:0.75rem; margin-top:0.5rem;">
                                    Current price is at {_pos_pct:.0f}% of the 52-week range.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("52-week range unavailable.")
                else:
                    st.caption("Not enough history for 52-week range.")
            except Exception:
                st.caption("52-week range unavailable.")

            # --- Signal Summary (matching Stock Analyzer table style) ---
            st.markdown("### Signal Summary")
            
            signal_summary = verdict_data['signal_summary']
            
            # Create metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Buy Signals", signal_summary['buy_signals'], 
                         delta="Bullish" if signal_summary['buy_signals'] > 0 else None)
            with col2:
                st.metric("📉 Sell Signals", signal_summary['sell_signals'],
                         delta="Bearish" if signal_summary['sell_signals'] > 0 else None)
            with col3:
                st.metric("⏳ Neutral Signals", signal_summary['neutral_signals'])
            
            # --- Display Chart ---
            st.markdown("### Technical Charts")

            # --- Chart window selector ---
            st.markdown("**Chart window**")
            period_labels = {
                "1M": 30,
                "2M": 60,
                "3M": 90,
                "6M": 180,
                "1Y": 365,
                "2Y": 730,
                "3Y": 1095,
                "5Y": 1825,
                "10Y": 3650,
                "MAX": None,
            }
            selected_period = st.radio(
                "Chart period:",
                options=list(period_labels.keys()),
                index=3,
                horizontal=True,
                key="chart_period",
                label_visibility="collapsed",
            )
            lookback_days = period_labels[selected_period]

            # --- Price overlay selector ---
            st.markdown("**Overlays**")
            cb1, cb2, cb3, cb4 = st.columns(4)
            with cb1:
                show_sma20 = st.checkbox("SMA 20", value=False, key="cb_sma20")
            with cb2:
                show_sma50 = st.checkbox("SMA 50", value=False, key="cb_sma50")
            with cb3:
                show_sma200 = st.checkbox("SMA 200", value=False, key="cb_sma200")
            with cb4:
                show_bb = st.checkbox("Bollinger Bands", value=False, key="cb_bb")

            try:
                fig = analyzer.create_interactive_chart(
                    show_sma20=show_sma20,
                    show_sma50=show_sma50,
                    show_sma200=show_sma200,
                    show_bb=show_bb,
                    lookback_days=lookback_days
                )
                
                if fig:
                    st.plotly_chart(fig, width="stretch")

                    # --- Historical signal performance ---
                    st.markdown("#### 📊 Signal Performance (Historical)")

                    cycles = analyzer.compute_signal_cycles(lookback_days=lookback_days)

                    for label, key in [("MACD (8, 17, 9)", "macd"),
                                    ("Stochastic Oscillator", "stoch")]:
                        st.markdown(f"**{label}**")
                        c_list = cycles.get(key, [])

                        if not c_list:
                            st.caption("No completed BUY→SELL cycles in this window.")
                            continue

                        rows = []
                        for c in c_list:
                            rows.append({
                                "Buy Date": c["buy_date"].strftime("%Y-%m-%d"),
                                "Buy Price": f"${c['buy_price']:.2f}",
                                "Sell Date": c["sell_date"].strftime("%Y-%m-%d"),
                                "Sell Price": f"${c['sell_price']:.2f}",
                                "% Move": f"{c['pct_change']:+.2f}%",
                            })
                        df = pd.DataFrame(rows)

                        # Summary stats
                        pcts = [c["pct_change"] for c in c_list]
                        wins = sum(1 for p in pcts if p > 0)
                        losses = sum(1 for p in pcts if p < 0)
                        avg_win = sum(p for p in pcts if p > 0) / wins if wins else 0
                        avg_loss = sum(p for p in pcts if p < 0) / losses if losses else 0
                        hit_rate = wins / len(pcts) * 100 if pcts else 0

                        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                        sc1.metric("Trades", len(pcts))
                        sc2.metric("Hit Rate", f"{hit_rate:.0f}%")
                        sc3.metric("Avg Win", f"{avg_win:+.2f}%")
                        sc4.metric("Avg Loss", f"{avg_loss:+.2f}%")
                        sc5.metric("Cumulative", f"{sum(pcts):+.2f}%")

                        st.dataframe(df, width="stretch", hide_index=True)
                else:
                    st.warning("Chart method returned None")
            except Exception as e:
                    st.error(f"Chart error: {type(e).__name__}: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # --- Display detailed signals (expandable) ---
            with st.expander("📋 Detailed Signal Breakdown"):
                signals = summary['signals']
                
                if signals.get('buy_signals'):
                    st.markdown("**Buy Signals**")
                    for signal in signals['buy_signals']:
                        st.success(f"✅ {signal}")
                
                if signals.get('sell_signals'):
                    st.markdown("**Sell Signals**")
                    for signal in signals['sell_signals']:
                        st.error(f"❌ {signal}")
                
                if signals.get('neutral_signals'):
                    st.markdown("**Neutral / Wait Signals**")
                    for signal in signals['neutral_signals']:
                        st.warning(f"⏳ {signal}")
                # --- Volume Details (separate section) ---
                volume_details = signals.get('volume_details', [])
                if volume_details:
                    st.markdown("---")
                    st.markdown("**📊 Volume Analysis**")
                    for detail in volume_details:
                        dtype = detail.get('type', 'neutral')
                        text = detail.get('text', '')
                        if dtype == 'positive':
                            st.success(f"✅ {text}")
                        elif dtype == 'caution':
                            st.warning(f"⚠️ {text}")
                        else:
                            st.info(f"ℹ️ {text}")
    
            # ============================================================
            # MEAN-REVERSION STRATEGY FAMILY (7 sub-strategies)
            # Independent view — not part of the main verdict
            # ============================================================
            st.markdown("### 🔀 Mean-Reversion Strategy Family")
            st.caption(
                "Seven trend + mean-reversion rules, weighted independently. "
                "This is a second opinion, not a replacement for the Verdict above."
            )

            mr_signal = analyzer.get_mr_signal()

            if "error" in mr_signal:
                st.warning(mr_signal["error"])
            else:
                score = mr_signal["score"]
                label = mr_signal["label"]
                regime = mr_signal["regime"]
                multiplier = mr_signal["multiplier"]

                # Color by label
                if "STRONG BULLISH" in label:
                    color, bg = "#00E676", "#1e4620"
                elif "BULLISH" in label:
                    color, bg = "#4CAF50", "#1e4620"
                elif "STRONG BEARISH" in label:
                    color, bg = "#EF5350", "#4b1e1e"
                elif "BEARISH" in label:
                    color, bg = "#FF7043", "#3d241e"
                else:
                    color, bg = "#FFA726", "#4a3a1e"

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown(f"""
                    <div style="background-color:{bg}; border:2px solid {color};
                                border-radius:10px; padding:1.25rem; text-align:center;">
                        <div style="font-size:1.5rem; font-weight:700; color:{color};">
                            {label}
                        </div>
                        <div style="color:#9aa0a6; font-size:0.85rem; margin-top:0.4rem;">
                            Score: <strong>{score:.1f}</strong> / 100
                        </div>
                        <div style="color:#9aa0a6; font-size:0.75rem;">
                            Regime: {regime}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Raw Score", f"{mr_signal['raw_score']:.1f}")
                    mc2.metric("BB Multiplier", f"{multiplier:.1f}×")
                    mc3.metric("Adjusted Score", f"{score:.1f}")

                    st.caption(
                        f"BB width: {mr_signal['bb_width']*100:.1f}% · "
                        f"Distance from SMA 50: {mr_signal['dist_sma50']*100:+.1f}% · "
                        f"RSI: {mr_signal['rsi']:.1f}"
                    )

                # Per-strategy breakdown
                with st.expander("📋 Sub-Strategy Votes (7 rules)"):
                    vote_df = pd.DataFrame(
                        mr_signal["details"],
                        columns=["Strategy", "Vote (TQQQ-equivalent %)", "Reason"],
                    )
                    st.dataframe(vote_df, width="stretch", hide_index=True)

                    st.caption(
                        "Each rule votes a score from 0 to 100. The average is the raw score. "
                        "The Bollinger Band width multiplier adjusts for trend strength."
                    )

                # Disagreement note vs. main verdict
                main_verdict = verdict_data["verdict"]
                if ("BULLISH" in label and "SELL" in main_verdict) or \
                   ("BEARISH" in label and "BUY" in main_verdict):
                    st.warning(
                        f"⚠️ **Divergence detected**: the main Verdict says **{main_verdict}** "
                        f"while this strategy family says **{label}**. "
                        f"When they disagree, treat the setup as ambiguous — "
                        f"reduce position size or wait for alignment."
                    )

            # ============================================================
            # BNF REVERSAL SETUP SCANNER
            # Takashi Kotegawa style — mean-reversion on extreme dislocation
            # ============================================================
            st.markdown("---")
            st.markdown("### 🔄 BNF Reversal Setup")
            st.caption(
                "Takashi Kotegawa (BNF) style mean-reversion check. "
                "Looks for stocks dislocated 20%+ below their 25-day MA with RSI/BB confirmation. "
                "This is a bear-market / panic setup — not a trend-following signal."
            )

            bnf = analyzer.check_bnf_reversal_setup()

            # Color and icon by setup state
            if bnf.get("setup"):
                bnf_color, bnf_bg = "#00E676", "#1e4620"
                bnf_label = "⚡ SETUP DETECTED"
            else:
                bnf_color, bnf_bg = "#9AA0A6", "#2a2a2a"
                bnf_label = "No setup"

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                <div style="background-color:{bnf_bg}; border:2px solid {bnf_color};
                            border-radius:10px; padding:1.25rem; text-align:center;">
                    <div style="font-size:1.25rem; font-weight:700; color:{bnf_color};">
                        {bnf_label}
                    </div>
                    <div style="color:#9aa0a6; font-size:0.8rem; margin-top:0.4rem;">
                        {bnf.get('reason', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Show the dislocation as the headline metric
                dislocation = bnf.get("dislocation_pct")
                if dislocation is not None:
                    # Color delta by severity
                    if dislocation <= -35:
                        delta_color = "normal"   # extreme — highlighted
                    elif dislocation <= -20:
                        delta_color = "normal"   # setup zone
                    else:
                        delta_color = "off"      # not dislocated

                    st.metric(
                        "Dislocation from MA25",
                        f"{dislocation:+.2f}%",
                        delta="Extreme" if dislocation <= -35 else
                              "In setup zone" if dislocation <= -20 else
                              "Normal range",
                        delta_color=delta_color,
                    )

                # Confirmation metrics
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("RSI", f"{bnf.get('rsi', 0):.1f}",
                          delta="Oversold" if bnf.get("rsi", 50) < 30 else "Normal",
                          delta_color="normal" if bnf.get("rsi", 50) < 30 else "off")
                mc2.metric("MA 25", f"${bnf.get('ma_25', 0):.2f}")
                mc3.metric("Current Price", f"${bnf.get('current_price', 0):.2f}")

            # Thresholds explanation (collapsed by default)
            with st.expander("📖 How this setup works"):
                st.markdown("""
                **The BNF Reversal Setup** (Takashi Kotegawa style):

                - **Anchor**: 25-day moving average
                - **Dislocation threshold**: price at least 20% below MA25
                - **Extreme panic threshold**: price 35%+ below MA25
                - **Confirmation**: RSI < 30 **OR** price below lower Bollinger Band
                - **Holding period**: 2–6 days, exit as price reverts to the mean

                **When it works**: high-volatility bear markets and panic-driven selloffs.
                **When it fails**: strong trending bull markets — "oversold" stocks keep grinding lower.

                This is a **short-term swing setup**, not a buy-and-hold signal. Cut losses
                if the bounce doesn't materialize within 2–3 days.
                """)
            # --- Download Data ---
            with st.expander("📥 Download Data"):
                csv = analyzer.stock_data.to_csv()
                st.download_button(
                    label="Download Full Technical Data (CSV)",
                    data=csv,
                    file_name=f"{tech_ticker}_technical_data.csv",
                    mime="text/csv"
                )

def _read_sleeve_positions() -> dict:
    """Read TQQQ and SQQQ quantity + average cost basis from
    data/sleeve_positions.csv (Fidelity positions export).

    Reads with the csv module and addresses columns by index, because
    Fidelity's header row has one fewer field than the data rows — this
    causes pandas to misalign the columns. Returns:
        {"TQQQ": {"qty": float, "avg_cost": float}, "SQQQ": {...}}
    or {} if the file is missing or unreadable.
    """
    import csv
    from pathlib import Path

    path = Path("data/sleeve_positions.csv")
    if not path.exists():
        return {}

    out: dict[str, dict] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue  # skip header
                if not row or len(row) < 20:
                    continue
                sym = (row[2] or "").strip()
                if sym not in ("TQQQ", "SQQQ"):
                    continue

                def _num(v):
                    if v is None:
                        return None
                    try:
                        return float(str(v).replace("$", "").replace(",", "").strip())
                    except Exception:
                        return None

                qty = _num(row[18])
                avg = _num(row[19])
                if qty is not None and avg is not None:
                    out[sym] = {"qty": qty, "avg_cost": avg}
    except Exception:
        return {}

    return out

def _compute_unrealized_pnl() -> dict:
    """Compute unrealized P&L on the current TQQQ/SQQQ position.

    Prefers avg cost from data/current_position.json — which is kept up to date
    as you log trades. Falls back to the sleeve CSV only if the JSON fields
    are missing (e.g., first run before any trade is logged).

    Uses yfinance's last close for the current price.
    """
    from data_provider import get_ticker
    from trade_manager import load_position

    pos = load_position()
    tq_qty = float(pos.get("tqqq_shares", 0.0))
    sq_qty = float(pos.get("sqqq_shares", 0.0))
    tq_avg = pos.get("tqqq_avg_cost")
    sq_avg = pos.get("sqqq_avg_cost")

    source = "json"

    # Fall back to the sleeve CSV only if avg cost isn't in the JSON yet
    if tq_avg is None or sq_avg is None:
        sleeve = _read_sleeve_positions()
        if tq_avg is None and "TQQQ" in sleeve:
            tq_avg = sleeve["TQQQ"]["avg_cost"]
            tq_qty = sleeve["TQQQ"]["qty"]
        if sq_avg is None and "SQQQ" in sleeve:
            sq_avg = sleeve["SQQQ"]["avg_cost"]
            sq_qty = sleeve["SQQQ"]["qty"]
        if tq_avg is not None and sq_avg is not None:
            source = "sleeve_csv"

    # ---- Current prices (last close, consistent with sleeve value) ----
    try:
        tq_px = float(get_ticker("TQQQ").history(period="1d")["Close"].iloc[-1])
    except Exception:
        tq_px = None
    try:
        sq_px = float(get_ticker("SQQQ").history(period="1d")["Close"].iloc[-1])
    except Exception:
        sq_px = None

    tq_unreal = (
        (tq_px - float(tq_avg)) * tq_qty
        if tq_px is not None and tq_avg is not None else None
    )
    sq_unreal = (
        (sq_px - float(sq_avg)) * sq_qty
        if sq_px is not None and sq_avg is not None else None
    )

    total = None
    if tq_unreal is not None and sq_unreal is not None:
        total = tq_unreal + sq_unreal
    elif tq_unreal is not None:
        total = tq_unreal
    elif sq_unreal is not None:
        total = sq_unreal

    return {
        "total": total,
        "tqqq": tq_unreal,
        "sqqq": sq_unreal,
        "tqqq_avg": tq_avg,
        "sqqq_avg": sq_avg,
        "tqqq_qty": tq_qty,
        "sqqq_qty": sq_qty,
        "source": source,
    }

def render_tqqq_sqqq_signals():
    """TQQQ/SQQQ Multi-Strategy Signals tab — daily trading decisions"""
    st.header("🤖 TQQQ/SQQQ Multi-Strategy Signals")
    st.caption(
        "Seven independent sub-strategies vote daily on target allocation. "
        "Each strategy is a separate module — add, remove, or tune them independently."
    )

    # --- Apply pending widget-state updates BEFORE widgets are instantiated ---
    for _pending, _target in (
        ("_pending_cur_tqqq",      "cur_tqqq_input"),
        ("_pending_cur_sqqq",      "cur_sqqq_input"),
        ("_pending_actual_tqqq",   "actual_tqqq_fill"),
        ("_pending_actual_sqqq",   "actual_sqqq_fill"),
    ):
        if _pending in st.session_state:
            st.session_state[_target] = st.session_state.pop(_pending)

    # Manual refresh button — forces a fresh yfinance fetch
    # Manual refresh button — forces a fresh yfinance fetch
    refresh_col1, refresh_col2 = st.columns([1, 4])
    with refresh_col1:
        if st.button("🔄 Refresh Signals", key="refresh_tqqq", width="stretch"):
            # 1. Clear Streamlit's memory
            st.cache_data.clear()
            
            # 2. Nuke the local yfinance_cache for these specific tickers
            # This forces the app to reach out to Yahoo Finance and download fresh data
            clear_yfinance_cache("QQQ")
            clear_yfinance_cache("TQQQ")
            clear_yfinance_cache("SQQQ")            
            st.rerun()
            
    with refresh_col2:
        st.caption(
            "Click Refresh to force a fresh fetch at 3:50 PM ET. "
            "Otherwise cached data (1-hour TTL) is reused."
        )

    try:
        from tqqq_sqqq_strategies import run_all_strategies

        with st.spinner("Running TQQQ/SQQQ strategies..."):
            tqqq_signals = run_all_strategies()

        if "error" in tqqq_signals:
            st.warning(f"Strategy analysis unavailable: {tqqq_signals['error']}")
            return

        # --- Signal Header (Clean Text, No Box) ---
        signal = tqqq_signals["signal"]
        regime = tqqq_signals["regime"]
        confidence = tqqq_signals["confidence"]
        
        # QQQ SMAs
        sma_50 = tqqq_signals.get("sma_50", 0.0)
        sma_200 = tqqq_signals.get("sma_200", 0.0)
        sma_250 = tqqq_signals.get("sma_250", 0.0)
        
        # TQQQ SMAs
        tq_sma_50 = tqqq_signals.get("tqqq_sma_50", 0.0)
        tq_sma_200 = tqqq_signals.get("tqqq_sma_200", 0.0)
        tq_sma_250 = tqqq_signals.get("tqqq_sma_250", 0.0)

        if "TQQQ" in signal:
            color_tag = "green" if "HEAVY" in signal else "blue"
        elif "SQQQ" in signal:
            color_tag = "red" if "HEAVY" in signal else "orange"
        else:
            color_tag = "gray"

        # Split into two main columns: Signal on left, SMAs on right
        head_col1, head_col2 = st.columns([1, 2])
        
        with head_col1:
            st.markdown(f"### Signal: :{color_tag}[{signal}]")
            st.markdown(f"**Regime:** {regime} | **Confidence:** {confidence:.0f}%")
            
            # Extract prices
            qqq_price = tqqq_signals.get("current_price", 0.0)
            tqqq_price = tqqq_signals.get("tqqq_current_price", 0.0)
            
            # Add a little spacing before the prices
            st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
            
            # Display side-by-side prices
            p1, p2 = st.columns(2)            
            # Helper to generate the colored sub-text
            def _sma_status(price, sma_val, label):
                if not sma_val or sma_val == 0: return ""
                pct = ((price - sma_val) / sma_val) * 100
                color = "#00E676" if pct > 0 else "#EF5350"
                return f"<span style='color:{color};'>{pct:+.1f}%</span> vs {label}<br>"
            
            with p1:
                st.metric("QQQ Price", f"${qqq_price:.2f}")
                
                # QQQ SMA distances
                qqq_subtext = (
                    _sma_status(qqq_price, sma_50, "50") +
                    _sma_status(qqq_price, sma_200, "200") +
                    _sma_status(qqq_price, sma_250, "250")
                )
                st.markdown(f"<div style='font-size:0.75rem; line-height:1.5; margin-top:-0.5rem;'>{qqq_subtext}</div>", unsafe_allow_html=True)

            with p2:
                st.metric("TQQQ Price", f"${tqqq_price:.2f}")
                
                # TQQQ SMA distances
                tqqq_subtext = (
                    _sma_status(tqqq_price, tq_sma_50, "50") +
                    _sma_status(tqqq_price, tq_sma_200, "200") +
                    _sma_status(tqqq_price, tq_sma_250, "250")
                )
                st.markdown(f"<div style='font-size:0.75rem; line-height:1.5; margin-top:-0.5rem;'>{tqqq_subtext}</div>", unsafe_allow_html=True)
            
        with head_col2:
            # Row 1: QQQ SMAs
            st.markdown("**QQQ** *(Signal Source)*")
            q1, q2, q3 = st.columns(3)
            q1.metric("SMA 50", f"${sma_50:.2f}")
            q2.metric("SMA 200", f"${sma_200:.2f}")
            q3.metric("SMA 250", f"${sma_250:.2f}")
            
            # Row 2: TQQQ SMAs
            st.markdown("**TQQQ** *(Trading Vehicle)*")
            t1, t2, t3 = st.columns(3)
            t1.metric("SMA 50", f"${tq_sma_50:.2f}")
            t2.metric("SMA 200", f"${tq_sma_200:.2f}")
            t3.metric("SMA 250", f"${tq_sma_250:.2f}")
            
        st.divider()

        # --- Side-by-Side Strategy Targets ---
        st.markdown("#### 🎯 Strategy Targets")
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**MoatCheck (7-Strategy Ensemble)**")
            a1, a2, a3 = st.columns(3)
            a1.metric("TQQQ Target", f"{tqqq_signals['target_tqqq_pct']:.1f}%")
            a2.metric("SQQQ Target", f"{tqqq_signals['target_sqqq_pct']:.1f}%")
            a3.metric("Cash",        f"{tqqq_signals['target_cash_pct']:.1f}%")
            
        with c2:
            st.markdown("**Mallik Sim (Continuous Sizing)**")
            m1, m2, m3 = st.columns(3)
            m1.metric("TQQQ Target", f"{tqqq_signals['mallik_tqqq_pct']:.1f}%")
            m2.metric("SQQQ Target", f"{tqqq_signals['mallik_sqqq_pct']:.1f}%")
            m3.metric("Cash",        f"{tqqq_signals['mallik_cash_pct']:.1f}%")
            
        st.divider()

        # Per-strategy vote breakdown
        with st.expander("📋 Sub-Strategy Votes"):
            vote_df = pd.DataFrame(tqqq_signals["vote_details"])
            vote_df.columns = ["Strategy", "Target TQQQ %", "Reason"]

            # --- Derived insight columns ---------------------------------------
            # 1. Implied SQQQ %: if the sub-strategy is fully allocated, whatever
            #    isn't TQQQ is presumed to be the short-side vote.
            vote_df["Target SQQQ %"] = (100 - vote_df["Target TQQQ %"]).round(1)

            # 2. Equal-weight contribution to the final aggregated target.
            n = len(vote_df)
            vote_df["Weight"] = f"{100 / n:.1f}%"
            vote_df["Contribution"] = (vote_df["Target TQQQ %"] / n).round(1)

            # 3. Net directional bias per sub-strategy.
            vote_df["Bias"] = vote_df["Target TQQQ %"].apply(
                lambda v: "Bullish" if v >= 60 else ("Bearish" if v <= 40 else "Neutral")
            )

            # Order columns for readability
            vote_df = vote_df[
                ["Strategy", "Target TQQQ %", "Target SQQQ %",
                "Weight", "Contribution", "Bias", "Reason"]
            ]

            # Round the numeric columns
            vote_df["Target TQQQ %"] = vote_df["Target TQQQ %"].round(1)

            # --- Center the numeric columns, left-align text columns -----------
            numeric_cols = ["Target TQQQ %", "Target SQQQ %", "Weight", "Contribution", "Bias"]

            styled = (
                vote_df.style
                .set_properties(subset=numeric_cols, **{"text-align": "center"})
                .set_properties(subset=["Strategy", "Reason"], **{"text-align": "left"})
            )

            st.dataframe(styled, width="stretch", hide_index=True)

            st.markdown("---")
            st.markdown(
                f"**Aggregated target**: "
                f"{tqqq_signals['target_tqqq_pct']:.1f}% TQQQ / "
                f"{tqqq_signals['target_sqqq_pct']:.1f}% SQQQ"
            )
            st.caption(
                "Each strategy votes independently; the average becomes the daily target. "
                "Position size is adjusted by BB width (trend strength)."
            )

            st.markdown("---")
            st.markdown(
                f"**Aggregated target**: "
                f"{tqqq_signals['target_tqqq_pct']:.1f}% TQQQ / "
                f"{tqqq_signals['target_sqqq_pct']:.1f}% SQQQ"
            )
            st.caption(
                "Each strategy votes independently; the average becomes the daily target. "
                "Position size is adjusted by BB width (trend strength)."
            )

        # ============================================================
        # POSITION CALCULATOR
        # ============================================================
        st.markdown("---")
        st.markdown("#### 🎯 Position Calculator")
        st.caption(
            "Enter your current sleeve value and holdings. "
            "The calculator tells you exactly how many fractional shares to buy or sell."
        )

        # --- Read capital state ---
        _cap = load_capital()
        starting = _cap.get("starting_capital", 0.0)
        deposits = sum(x["amount"] for x in _cap.get("deposits", []))
        withdrawals = sum(x["amount"] for x in _cap.get("withdrawals", []))
        net_cap = starting + deposits - withdrawals

        if "tqqq_capital" not in st.session_state:
            st.session_state["tqqq_capital"] = net_cap

        # Fetch prices BEFORE rendering the row so everything lands on one line
        try:
            tqqq_price = float(get_ticker("TQQQ").history(period="1d")["Close"].iloc[-1])
            sqqq_price = float(get_ticker("SQQQ").history(period="1d")["Close"].iloc[-1])
        except Exception as e:
            st.error(f"Could not fetch prices: {e}")
            tqqq_price = 0.0
            sqqq_price = 0.0

        # --- Single row: capital stats + sleeve value + prices ---
        c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1.5, 1, 1])
        c1.metric("Starting capital", f"${starting:,.2f}")
        c2.metric("Net deposits", f"${deposits - withdrawals:+,.2f}")
        c3.metric("Total committed", f"${net_cap:,.2f}")
        with c4:
            account_value = st.number_input(
                "Sleeve value ($)",
                min_value=100.0,
                max_value=1_000_000.0,
                value=float(st.session_state["tqqq_capital"]),
                step=10.0,
                key="tqqq_capital_input",
                help=(
                    "Open Fidelity → Accounts & Trade → Portfolio → Positions. "
                    "Add ONLY the Market Value of your TQQQ and SQQQ rows."
                ),
            )
            st.session_state["tqqq_capital"] = account_value
        c5.metric("TQQQ price", f"${tqqq_price:.2f}")
        c6.metric("SQQQ price", f"${sqqq_price:.2f}")

        st.caption(
            "📋 Enter only TQQQ + SQQQ market value. "
            "Cash and other positions are **not** part of this sleeve."
        )

        # --- Cumulative P&L banner (two cards: committed vs unrealized) ---
        _running = compute_running_pnl(account_value=account_value)
        _unreal = _compute_unrealized_pnl()

        if _running["net_capital"] > 0 and _running["sleeve_value"] > 0:
            _pnl = _running["pnl_dollars"]
            _pnl_pct = _running["pnl_pct"]
            if _pnl > 0:
                _pnl_color, _pnl_bg, _pnl_icon = "#00E676", "#1e4620", "📈"
            elif _pnl < 0:
                _pnl_color, _pnl_bg, _pnl_icon = "#EF5350", "#4b1e1e", "📉"
            else:
                _pnl_color, _pnl_bg, _pnl_icon = "#9AA0A6", "#2a2a2a", "➖"

            _colA, _colB = st.columns(2)

            with _colA:
                st.markdown(f"""
                <div style="background-color:{_pnl_bg}; border:2px solid {_pnl_color};
                            border-radius:10px; padding:1.25rem; margin:0.75rem 0;">
                    <div style="color:#9aa0a6; font-size:0.7rem; text-transform:uppercase;
                                letter-spacing:0.05em;">
                        P&amp;L vs committed capital
                    </div>
                    <div style="color:{_pnl_color}; font-size:2rem; font-weight:700;
                                line-height:1.1; margin-top:0.35rem;">
                        {_pnl_icon} ${_pnl:+,.2f}
                    </div>
                    <div style="color:{_pnl_color}; font-size:1rem; font-weight:600;
                                margin-top:0.15rem;">
                        {_pnl_pct:+.2f}%
                    </div>
                    <div style="color:#9aa0a6; font-size:0.72rem; margin-top:0.6rem;">
                        Sleeve ${_running['sleeve_value']:,.2f} · Committed ${_running['net_capital']:,.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with _colB:
                _u = _unreal.get("total")
                if _u is None:
                    _u_color, _u_bg, _u_icon = "#9AA0A6", "#2a2a2a", "➖"
                    _u_text = "n/a"
                    _u_pct_text = ""
                elif _u > 0:
                    _u_color, _u_bg, _u_icon = "#00E676", "#1e4620", "📈"
                    _u_text = f"${_u:+,.2f}"
                elif _u < 0:
                    _u_color, _u_bg, _u_icon = "#EF5350", "#4b1e1e", "📉"
                    _u_text = f"${_u:+,.2f}"
                else:
                    _u_color, _u_bg, _u_icon = "#9AA0A6", "#2a2a2a", "➖"
                    _u_text = "$0.00"

                # Compute unrealized % relative to the cost basis of current positions
                _u_pct_text = ""
                try:
                    _tq_qty = _unreal.get("tqqq_qty")
                    _tq_avg = _unreal.get("tqqq_avg")
                    _sq_qty = _unreal.get("sqqq_qty")
                    _sq_avg = _unreal.get("sqqq_avg")
                    if _u is not None and all(v is not None for v in (_tq_qty, _tq_avg, _sq_qty, _sq_avg)):
                        _cost_basis = float(_tq_qty) * float(_tq_avg) + float(_sq_qty) * float(_sq_avg)
                        if _cost_basis > 0:
                            _u_pct = _u / _cost_basis * 100.0
                            _u_pct_text = f"{_u_pct:+.2f}%"
                except Exception:
                    pass

                _breakdown = ""
                if _unreal.get("tqqq") is not None and _unreal.get("sqqq") is not None:
                    # Add 0.0 to clean up negative zeros (e.g., -0.00 becomes 0.00)
                    _tq_val = _unreal['tqqq'] + 0.0
                    _sq_val = _unreal['sqqq'] + 0.0
                    _breakdown = f"TQQQ ${_tq_val:+,.2f} · SQQQ ${_sq_val:+,.2f}"

                # Render as a single continuous string to prevent Markdown code-block formatting
                st.markdown(
                    f'<div style="background-color:{_u_bg}; border:2px solid {_u_color}; border-radius:10px; padding:1.25rem; margin:0.75rem 0;">'
                    f'<div style="color:#9aa0a6; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.05em;">Unrealized P&amp;L (open positions)</div>'
                    f'<div style="color:{_u_color}; font-size:2rem; font-weight:700; line-height:1.1; margin-top:0.35rem;">{_u_icon} {_u_text}</div>'
                    f'<div style="color:{_u_color}; font-size:1rem; font-weight:600; margin-top:0.15rem;">{_u_pct_text}</div>'
                    f'<div style="color:#9aa0a6; font-size:0.72rem; margin-top:0.6rem;">{_breakdown}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info(
                "Cumulative P&L will appear once capital and a current position are recorded."
            )

            # P&L vs committed capital
            pnl = account_value - net_cap
            pnl_pct = (pnl / net_cap * 100) if net_cap > 0 else 0.0
            st.metric(
                "P&L vs committed capital",
                f"${pnl:+,.2f}",
                f"{pnl_pct:+.2f}%",
                delta_color="normal" if pnl >= 0 else "inverse",
            )

        # --- Cross-check: entered sleeve value vs saved position ---
        if tqqq_price > 0 and sqqq_price > 0:
            _pos_check = load_position()
            _implied = (
                _pos_check.get("tqqq_shares", 0.0) * tqqq_price
                + _pos_check.get("sqqq_shares", 0.0) * sqqq_price
                + _pos_check.get("cash_balance", 0.0)
            )
            if _implied > 0:
                diff_pct = abs(account_value - _implied) / _implied * 100
                if diff_pct > 5:
                    st.warning(
                        f"⚠️ Entered ${account_value:,.2f}, but the saved position "
                        f"implies ${_implied:,.2f} ({diff_pct:.1f}% off). "
                        f"Double-check the number from Fidelity."
                    )

            if tqqq_price > 0 and sqqq_price > 0:
                targets = compute_target_shares(
                    account_value=account_value,
                    tqqq_target_pct=tqqq_signals["target_tqqq_pct"],
                    sqqq_target_pct=tqqq_signals["target_sqqq_pct"], # <-- ADD THIS
                    tqqq_price=tqqq_price,
                    sqqq_price=sqqq_price,
                    #cash_target_pct=tqqq_signals["target_cash_pct"],
                )

            st.markdown("**Target Positions**")
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.metric(
                    "TQQQ target",
                    f"{targets['tqqq_target_shares']:.4f} shares",
                    delta=f"${targets['tqqq_target_value']:.2f}",
                )
            with tc2:
                st.metric(
                    "SQQQ target",
                    f"{targets['sqqq_target_shares']:.4f} shares",
                    delta=f"${targets['sqqq_target_value']:.2f}",
                )
            with tc3:
                st.metric(
                    "Cash target",
                    "—",
                    delta=f"${targets['cash_value']:.2f}",
                )

            _pos = load_position()

            st.markdown("**Your Current Holdings**")
            hc1, hc2 = st.columns(2)
            with hc1:
                current_tqqq = st.number_input(
                    "Current TQQQ shares",
                    min_value=0.0,
                    value=float(_pos.get("tqqq_shares", 0.0)),
                    step=0.001,
                    format="%.4f",
                    key="cur_tqqq_input",
                )
            with hc2:
                current_sqqq = st.number_input(
                    "Current SQQQ shares",
                    min_value=0.0,
                    value=float(_pos.get("sqqq_shares", 0.0)),
                    step=0.001,
                    format="%.4f",
                    key="cur_sqqq_input",
                )

            if st.button("💾 Save current holdings", key="save_position"):
                _tq_val = current_tqqq * tqqq_price
                _sq_val = current_sqqq * sqqq_price
                _cash_val = max(0.0, account_value - _tq_val - _sq_val)
                save_position(
                    current_tqqq, current_sqqq, 
                    cash_balance=_cash_val
                )
                st.success(
                    f"Saved: TQQQ {current_tqqq:.4f}, SQQQ {current_sqqq:.4f}, Cash ${_cash_val:.2f}"
                )

            rebalance = compute_rebalance(
                current_tqqq_shares=current_tqqq,
                current_sqqq_shares=current_sqqq,
                tqqq_target_shares=targets["tqqq_target_shares"],
                sqqq_target_shares=targets["sqqq_target_shares"],
                tqqq_price=tqqq_price,
                sqqq_price=sqqq_price,
            )
            
            # Calculate Mallik Sim target shares and actions for reference
            mallik_targets = compute_target_shares(
                account_value=account_value,
                tqqq_target_pct=tqqq_signals["mallik_tqqq_pct"],
                sqqq_target_pct=tqqq_signals["mallik_sqqq_pct"], # <-- ADD THIS
                tqqq_price=tqqq_price,
                sqqq_price=sqqq_price,
                #cash_target_pct=tqqq_signals["mallik_cash_pct"],
            )
            mallik_rebalance = compute_rebalance(
                current_tqqq_shares=current_tqqq,
                current_sqqq_shares=current_sqqq,
                tqqq_target_shares=mallik_targets["tqqq_target_shares"],
                sqqq_target_shares=mallik_targets["sqqq_target_shares"],
                tqqq_price=tqqq_price,
                sqqq_price=sqqq_price,
            )

            #Action Display Block
            st.markdown("**Actions to Take Now**")
            
            def render_action_block(rebalance_dict, title_prefix=""):
                ac1, ac2 = st.columns(2)
                with ac1:
                    delta = rebalance_dict["tqqq_delta_shares"]
                    delta_value = abs(rebalance_dict["tqqq_delta_value"])
                    delta_pct = (delta_value / account_value) * 100 if account_value > 0 else 0
                    if delta_pct < MIN_TRADE_PCT:
                        st.info(f"{title_prefix}TQQQ: no action needed (delta {delta_pct:.2f}%)")
                    elif delta > 0:
                        st.success(f"{title_prefix}TQQQ: **BUY {delta:.4f} shares** (~${delta_value:.2f})")
                    else:
                        st.warning(f"{title_prefix}TQQQ: **SELL {abs(delta):.4f} shares** (~${delta_value:.2f})")
                with ac2:
                    delta = rebalance_dict["sqqq_delta_shares"]
                    delta_value = abs(rebalance_dict["sqqq_delta_value"])
                    delta_pct = (delta_value / account_value) * 100 if account_value > 0 else 0
                    if delta_pct < MIN_TRADE_PCT:
                        st.info(f"{title_prefix}SQQQ: no action needed (delta {delta_pct:.2f}%)")
                    elif delta > 0:
                        st.success(f"{title_prefix}SQQQ: **BUY {delta:.4f} shares** (~${delta_value:.2f})")
                    else:
                        st.warning(f"{title_prefix}SQQQ: **SELL {abs(delta):.4f} shares** (~${delta_value:.2f})")

            # Show MoatCheck Actions
            st.markdown("**MoatCheck Strategy**")
            render_action_block(rebalance)
            
            st.markdown("---")
            
            # Show Mallik Sim Actions
            st.markdown("**Mallik Sim Strategy** *(For Reference)*")
            render_action_block(mallik_rebalance)

            # Log trade buttons
            # Snapshot signal + target so a refresh between display and click
            # can't desync what gets logged
            signal_snapshot = tqqq_signals["signal"]
            target_snapshot = tqqq_signals["target_tqqq_pct"]

            st.markdown("**Log These Trades**")
            st.caption("Enter the actual share count from your Fidelity fill confirmation.")
            lc1, lc2 = st.columns(2)

            with lc1:
                actual_tqqq_fill = st.number_input(
                    "Actual TQQQ shares filled",
                    min_value=0.0, value=0.0, step=0.001, format="%.4f",
                    key="actual_tqqq_fill",
                )
                if st.button("✅ Log TQQQ trade", key="log_tqqq"):
                    if actual_tqqq_fill > 0:
                        delta = rebalance["tqqq_delta_shares"]
                        action = "BUY" if delta > 0 else "SELL"

                        log_trade(
                            ticker="TQQQ", action=action,
                            shares=abs(actual_tqqq_fill),
                            price=tqqq_price,
                            signal=signal_snapshot,
                            tqqq_target_pct=target_snapshot,
                            sqqq_target_pct=tqqq_signals["target_sqqq_pct"],
                            cash_target_pct=tqqq_signals["target_cash_pct"],
                            account_value=account_value,
                        )

                        _pos = load_position()
                        _cur_tqqq = float(_pos.get("tqqq_shares", 0.0))
                        _cur_sqqq = float(_pos.get("sqqq_shares", 0.0))
                        _cur_tq_avg = float(_pos.get("tqqq_avg_cost", 0.0))
                        _cur_sq_avg = float(_pos.get("sqqq_avg_cost", 0.0))
                        _cur_cash = float(_pos.get("cash_balance", 0.0))

                        if action == "BUY":
                            _new_tqqq = _cur_tqqq + abs(actual_tqqq_fill)
                            _new_cash = _cur_cash - (abs(actual_tqqq_fill) * tqqq_price)
                            if _new_tqqq > 0:
                                _new_tq_avg = ((_cur_tqqq * _cur_tq_avg) + (abs(actual_tqqq_fill) * tqqq_price)) / _new_tqqq
                            else:
                                _new_tq_avg = 0.0
                        else:  # SELL
                            _new_tqqq = max(0.0, _cur_tqqq - abs(actual_tqqq_fill))
                            _new_cash = _cur_cash + (abs(actual_tqqq_fill) * tqqq_price)
                            _new_tq_avg = _cur_tq_avg

                        save_position(
                            _new_tqqq, _cur_sqqq,
                            tqqq_avg_cost=_new_tq_avg,
                            sqqq_avg_cost=_cur_sq_avg,
                            cash_balance=_new_cash,
                        )
                        

                        # 3. Queue widget updates for the NEXT run
                        st.session_state["_pending_cur_tqqq"] = _new_tqqq
                        st.session_state["_pending_cur_sqqq"] = _cur_sqqq
                        st.session_state["_pending_actual_tqqq"] = 0.0
                        st.session_state["_last_log_msg"] = (
                            f"Logged {action} {abs(actual_tqqq_fill):.4f} TQQQ @ ${tqqq_price:.2f}. "
                            f"Holdings now {_new_tqqq:.4f} TQQQ."
                        )
                        st.rerun()
                    else:
                        st.info("Enter the filled share count above first.")

            with lc2:
                actual_sqqq_fill = st.number_input(
                    "Actual SQQQ shares filled",
                    min_value=0.0, value=0.0, step=0.001, format="%.4f",
                    key="actual_sqqq_fill",
                )
                if st.button("✅ Log SQQQ trade", key="log_sqqq"):
                    if actual_sqqq_fill > 0:
                        delta = rebalance["sqqq_delta_shares"]
                        action = "BUY" if delta > 0 else "SELL"

                        log_trade(
                            ticker="SQQQ", action=action,
                            shares=abs(actual_sqqq_fill),
                            price=sqqq_price,
                            signal=signal_snapshot,
                            tqqq_target_pct=target_snapshot,
                            sqqq_target_pct=tqqq_signals["target_sqqq_pct"],
                            cash_target_pct=tqqq_signals["target_cash_pct"],
                            account_value=account_value,
                        )

                        _pos = load_position()
                        _cur_tqqq = float(_pos.get("tqqq_shares", 0.0))
                        _cur_sqqq = float(_pos.get("sqqq_shares", 0.0))
                        _cur_tq_avg = float(_pos.get("tqqq_avg_cost", 0.0))
                        _cur_sq_avg = float(_pos.get("sqqq_avg_cost", 0.0))
                        _cur_cash = float(_pos.get("cash_balance", 0.0))

                        if action == "BUY":
                            _new_sqqq = _cur_sqqq + abs(actual_sqqq_fill)
                            _new_cash = _cur_cash - (abs(actual_sqqq_fill) * sqqq_price)
                            if _new_sqqq > 0:
                                _new_sq_avg = ((_cur_sqqq * _cur_sq_avg) + (abs(actual_sqqq_fill) * sqqq_price)) / _new_sqqq
                            else:
                                _new_sq_avg = 0.0
                        else:  # SELL
                            _new_sqqq = max(0.0, _cur_sqqq - abs(actual_sqqq_fill))
                            _new_cash = _cur_cash + (abs(actual_sqqq_fill) * sqqq_price)
                            _new_sq_avg = _cur_sq_avg

                        save_position(
                            _cur_tqqq, _new_sqqq,
                            tqqq_avg_cost=_cur_tq_avg,
                            sqqq_avg_cost=_new_sq_avg,
                            cash_balance=_new_cash,
                        )

                        # 3. Queue widget updates for the NEXT run
                        st.session_state["_pending_cur_tqqq"] = _cur_tqqq
                        st.session_state["_pending_cur_sqqq"] = _new_sqqq
                        st.session_state["_pending_actual_sqqq"] = 0.0
                        st.session_state["_last_log_msg"] = (
                            f"Logged {action} {abs(actual_sqqq_fill):.4f} SQQQ @ ${sqqq_price:.2f}. "
                            f"Holdings now {_new_sqqq:.4f} SQQQ."
                        )
                        st.rerun()
                    else:
                        st.info("Enter the filled share count above first.")

            # Show the success message from a rerun (if any)
            _msg = st.session_state.pop("_last_log_msg", None)
            if _msg:
                st.success(_msg)
        # ============================================================
        # TRADE HISTORY
        # ============================================================
        st.markdown("---")
        st.markdown("#### 📒 Trade History")

        trades_df = compute_realized_pnl()

        if trades_df.empty:
            st.info("No trades logged yet. Use the Log buttons above to record your fills.")
        else:
            total_trades = len(trades_df)
            buy_count = (trades_df["action"] == "BUY").sum()
            sell_count = (trades_df["action"] == "SELL").sum()
            total_volume = trades_df["total_value"].sum()
            total_realized = trades_df["cumulative_realized_pnl"].iloc[-1]

            _unreal_hist = _compute_unrealized_pnl()
            _unreal_val = _unreal_hist.get("total")
            _net_pnl = (
                total_realized + _unreal_val
                if _unreal_val is not None else None
            )

            sc1, sc2, sc3, sc4, sc5, sc6, sc7 = st.columns(7)
            sc1.metric("Total trades", total_trades)
            sc2.metric("Buys", int(buy_count))
            sc3.metric("Sells", int(sell_count))
            sc4.metric("Total volume", f"${total_volume:,.2f}")
            sc5.metric(
                "Realized P&L",
                f"${total_realized:+,.2f}",
                delta_color="normal" if total_realized >= 0 else "inverse",
            )
            sc6.metric(
                "Unrealized P&L",
                f"${_unreal_val:+,.2f}" if _unreal_val is not None else "n/a",
                delta_color=(
                    "normal" if _unreal_val is not None and _unreal_val >= 0
                    else "inverse" if _unreal_val is not None and _unreal_val < 0
                    else "off"
                ),
            )
            sc7.metric(
                "Net P&L",
                f"${_net_pnl:+,.2f}" if _net_pnl is not None else "n/a",
                delta_color=(
                    "normal" if _net_pnl is not None and _net_pnl >= 0
                    else "inverse" if _net_pnl is not None and _net_pnl < 0
                    else "off"
                ),
            )

            # Display sorted by time descending so the newest trade is at the top
            display_df = trades_df.sort_values("timestamp", ascending=False).copy()

            # Format the new columns for readability
            display_df["realized_pnl_this_trade"] = display_df["realized_pnl_this_trade"].apply(
                lambda v: f"${v:+,.2f}" if v else "—"
            )
            display_df["cumulative_realized_pnl"] = display_df["cumulative_realized_pnl"].apply(
                lambda v: f"${v:+,.2f}"
            )

            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
            )

            csv_bytes = trades_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download trade log",
                data=csv_bytes,
                file_name=f"trades_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

    except ImportError:
        st.info("TQQQ/SQQQ strategies module not installed. "
                "Create `tqqq_sqqq_strategies.py` to enable this feature.")
    except Exception as e:
        st.warning(f"Could not run TQQQ/SQQQ strategies: {e}")


# ---------------------------------------------------------------------- #
# Page wrappers — each defines its own header, then calls the render fn
# ---------------------------------------------------------------------- #

def page_my_investments() -> None:
    render_my_investments()

def page_backtest() -> None:
    render_backtest()

def render_backtest() -> None:
    """Back Test page — run historical simulations of trading strategies."""
    st.header("🧪 Back Test")
    
    st.markdown(
        "Compare multiple long-or-cash strategies on historical data. "
        "All strategies execute at next-day open."
    )
    st.markdown(
        "- **Buy and Hold** — fully invested in the target ticker the entire window. The passive baseline.\n"
        "- **7-Step Strategy** — trades the live regime-aware framework (uses QQQ signals). Goes to cash in bear/neutral regimes.\n"
        "- **One Percent A Week** — weekly swing strategy. Enters on a 1% dip from Monday's open, exits on a 1% gain or at Friday close.\n"
        "- **Swing - since last exit** — buys when price drops *Swing %* below the running high, and sells when it recovers *Swing %* from entry.\n"
        "- **Swing - 20-day window** — buys when price drops *Swing %* below the 20-day high.\n"
        "- **Swing - 60-day window** — buys when price drops *Swing %* below the 60-day high.\n"
        "- **Regime-Aware Swing** — combines the Swing strategy with a macro filter. Only buys the *Swing %* dip if QQQ is above its 200-day moving average (bull market). Stays in cash if QQQ is below."
    )

    # ---------- Import lazily so the page only loads its deps when opened ----------
    try:
        from backtest import load_backtest_data, run_backtest
        import backtest_strategies as bs
    except Exception as e:
        st.error(f"Could not load backtest modules: {e}")
        return

    # ---------- Input row ----------
    c0, c1, c2, c3, c4 = st.columns([1.5, 1.5, 1.5, 1.2, 1.0])
    
    with c0:
        ticker_input = st.text_input("Ticker", value="TQQQ", key="backtest_ticker")
        backtest_ticker = ticker_input.strip().upper()
        
    with c1:
        period_label = st.selectbox("Period", options=["1m", "3m", "6m", "1y", "2y", "3y", "5y", "8y", "10y", "20y", "Max"], index=5)
        
    with c2:
        starting_cash = st.number_input("Starting cash ($)", min_value=1000.0, max_value=10_000_000.0, value=10_000.0, step=1000.0)
        
    with c3:
        swing_pct = st.number_input(
            "Swing %",
            min_value=1.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
            format="%.1f",
            help="For the Swing strategy: buy when price drops this % below the running high; "
                 "sell when up this % from entry.",
            key="backtest_swing_pct",
        )
        
    with c4:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        _btn_slot = st.empty()
        if not st.session_state.get("backtest_running"):
            run_clicked = _btn_slot.button("▶ Run All", type="primary", use_container_width=True, key="backtest_btn")
            if run_clicked:
                st.session_state["backtest_running"] = True
                st.rerun()
        else:
            run_clicked = False
            _btn_slot.markdown(
                '<div style="display:flex; align-items:center; justify-content:center; '
                'height:38px; background:#2a2a2a; border-radius:8px; color:#9aa0a6; '
                'font-size:0.9rem;">'
                '<span style="display:inline-block; width:16px; height:16px; '
                'border:2px solid #4DABF7; border-top-color:transparent; '
                'border-radius:50%; animation:spin 0.8s linear infinite; '
                'margin-right:0.5rem;"></span>'
                'Running…</div>'
                '<style>@keyframes spin { to { transform: rotate(360deg); } }</style>',
                unsafe_allow_html=True,
            )
    # ---------- Convert period label to a start date ----------
    from datetime import datetime, timedelta
    today = datetime.now().date()
    period_days = {
        "1m": 30, "3m": 90, "6m": 180, "1y": 365, 
        "2y": 730, "3y": 1095, "5y": 1825, 
        "8y": 2920, "10y": 3650, "20y": 7300, "Max": 9999,
    }
    start_date = (today - timedelta(days=period_days[period_label])).isoformat()
    end_date = today.isoformat()

    # ---------- Do the work when the spinner is showing ----------
    if st.session_state.get("backtest_running"):
        try:
            with st.spinner(f"Loading historical prices for {backtest_ticker}…"):
                try:
                    prices = load_backtest_data(backtest_ticker, start_date, end_date)
                except Exception as e:
                    st.error(f"Could not load price data: {e}")
                    st.session_state["backtest_running"] = False
                    return

            if prices.empty or len(prices) < 60:
                st.warning(f"Not enough data for a {period_label} backtest — got {len(prices)} bars.")
                st.session_state["backtest_running"] = False
                return

            # --- RUN THE DATA QUALITY CHECK HERE ---
            _check_data_quality(prices, backtest_ticker)

            qqq_close = prices["QQQ_close"]
            tqqq_close = prices["TQQQ_close"]
            
            start_ts = datetime.fromisoformat(start_date)
            
            results = {}
            import inspect

            for name, fn in bs.STRATEGIES.items():
                _params = inspect.signature(fn).parameters

                extra_kwargs = {}
                if "Swing" in name:
                    extra_kwargs["threshold"] = swing_pct / 100.0
                    if "tqqq_open" in _params:
                        extra_kwargs["tqqq_open"] = prices["TQQQ_open"]

                # 1. Generate signals on the FULL dataset (with warmup)
                if "progress_callback" in _params:
                    signal = fn(qqq_close, tqqq_close, progress_callback=None, **extra_kwargs)
                else:
                    signal = fn(qqq_close, tqqq_close, **extra_kwargs)
                
                # 2. Slice prices and signal to just the user's chosen window
                prices_sliced = prices[prices.index >= start_ts].copy()
                signal_sliced = signal.reindex(prices_sliced.index).fillna(0).astype(int)
                
                # 3. Run a fresh backtest on ONLY the sliced data (no extra warmup needed here)
                results[name] = run_backtest(
                    prices_sliced, 
                    signal_sliced, 
                    starting_cash=starting_cash, 
                    warmup_days=0, 
                    progress_callback=None
                )

            st.session_state["backtest_results"] = results
            st.session_state["backtest_prices"] = prices

        except Exception as e:
            st.error(f"Backtest error: {e}")
        finally:
            st.session_state["backtest_running"] = False

        st.rerun()

    # ---------- Display results ----------
    results = st.session_state.get("backtest_results")
    prices = st.session_state.get("backtest_prices")
    if results is None or prices is None:
        st.info("Choose a period and starting cash, then click **Run All**.")
        return

    first = next(iter(results.values()))
    st.caption(
        f"Window: **{first.start_date.date()} → {first.end_date.date()}** "
        f"({(first.end_date - first.start_date).days} days) · "
        f"Starting cash: **${first.starting_cash:,.0f}**"
    )

    if "backtest_warning" in st.session_state:
        st.warning(st.session_state["backtest_warning"])

    # ---------- Strategy comparison table ----------
    st.markdown("#### Strategy Comparison")

    rows = []
    for name, r in results.items():
        calmar = r.cagr / r.max_drawdown if r.max_drawdown > 0 else 0.0
        end_cash = r.starting_cash * (1 + r.total_return)
        display_name = name
        
        # Registry key looks like "Swing - 20-day window". Extract the descriptive suffix after the dash and re-attach it after the current threshold so the row reads "Swing 20% - 20-day window".
        if "Swing" in name and "Regime-Aware" not in name:
            parts = name.split(" - ", 1)
            suffix = parts[1] if len(parts) == 2 else ""
            display_name = f"Swing {swing_pct:.0f}% - {suffix}".rstrip(" -")

        rows.append({
            "Strategy": display_name,
            "Total Return": f"{r.total_return*100:+.2f}%",
            "CAGR": f"{r.cagr*100:+.2f}%",
            "Max Drawdown": f"{r.max_drawdown*100:.2f}%",
            "Total Capital": f"${end_cash:,.0f}",
            "Sharpe *": f"{r.sharpe:.2f}",
            "Calmar **": f"{calmar:.2f}",
            "Trades": f"{r.num_trades}",
            "Win Rate": f"{r.win_rate*100:.0f}%",
        })

    comparison_df = pd.DataFrame(rows)

    # Column-level color coding for Total Return and Max Drawdown
    def _style_comparison(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for i, (_, r) in enumerate(results.items()):
            ret = r.total_return
            dd = r.max_drawdown
            styles.loc[i, "Total Return"] = "color: #00E676;" if ret >= 0 else "color: #EF5350;"
            styles.loc[i, "Max Drawdown"] = "color: #EF5350;" if dd >= 0.30 else "color: #FFA726;" if dd >= 0.15 else "color: #00E676;"
            styles.loc[i, "Total Capital"] = ("color: #00E676; font-weight: 600;" if ret >= 0 else "color: #EF5350; font-weight: 600;")
            styles.loc[i, "Strategy"] = "font-weight: 600;"
        return styles

    st.dataframe(
        comparison_df.style.apply(_style_comparison, axis=None),
        use_container_width=True,
        hide_index=True,
    )
    
    st.caption(
        "\\* **Sharpe** = CAGR ÷ annualized volatility. Measures return per unit of "
        "day-to-day risk. Higher is better. Realistic range is 0.5 - 2.0 where 2 is excellent. Anything greater than 2 is Very Rare and should be treated with caution.\n\n"
        "\\*\\* **Calmar** = CAGR ÷ Max Drawdown. Measures return per unit of "
        "worst-case loss. Higher is better. Realistic range is 0.5 - 1.0 (Good). Between 1-2 is Very good but very rare.\n\n"
        "\\^ **TASC** = Technical Analysis of Stock Charts. a monthly magazine that has been published since 1982. The 1% per week strategy is inspired by its teachings."
    )    

    # ---------- Benchmarks ----------
    st.markdown("#### Buy & Hold Benchmarks")
    bc1, bc2 = st.columns(2)
    bc1.metric("TQQQ Buy & Hold", f"{first.tqqq_buy_hold_return*100:+.2f}%", delta=f"${first.starting_cash * (1 + first.tqqq_buy_hold_return):,.0f}", delta_color="normal" if first.tqqq_buy_hold_return >= 0 else "inverse")
    bc2.metric("QQQ Buy & Hold", f"{first.qqq_buy_hold_return*100:+.2f}%", delta=f"${first.starting_cash * (1 + first.qqq_buy_hold_return):,.0f}", delta_color="normal" if first.qqq_buy_hold_return >= 0 else "inverse")

    # ---------- Equity curve — all strategies on one chart ----------
    st.markdown("#### Equity Curves")
    import plotly.graph_objects as go
    fig = go.Figure()

    strategy_colors = ["#4DABF7", "#FF6B6B", "#FFD43B", "#B388FF", "#00E676", "#FFA726"]
    for i, (name, r) in enumerate(results.items()):
        color = strategy_colors[i % len(strategy_colors)]
        fig.add_trace(go.Scatter(x=r.equity_curve.index, y=r.equity_curve, name=name, line=dict(color=color, width=2.5)))

    # Add TQQQ buy-and-hold as a dashed reference line
    fig.add_trace(go.Scatter(x=first.benchmark_tqqq.index, y=first.benchmark_tqqq, name="TQQQ Buy & Hold", line=dict(color="#9AA0A6", width=1.5, dash="dot")))

    fig.update_layout(
        height=440,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Portfolio Value ($)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------- 15% Swing Analysis ----------
    st.markdown("#### 15% Swing Analysis")
    st.caption(
        "Peak-to-trough and trough-to-peak moves of at least the threshold "
        "below. Shows how often large swings occur and how much time "
        "separates them."
    )

    from swing_analysis import find_swings
    tqqq_series = prices["TQQQ_close"]

    _thr_col1, _thr_col2 = st.columns([1, 4])
    with _thr_col1:
        threshold_pct = st.number_input(
            "Swing threshold (%)",
            min_value=1.0,
            max_value=100.0,
            value=15.0,
            step=1.0,
            format="%.1f",
            key="swing_threshold_pct",
        )
    swings = find_swings(tqqq_series, threshold=threshold_pct / 100.0)

    if swings.empty:
        st.info("No 15% moves found in this window.")
    else:
        ups = swings[swings["direction"] == "UP"]
        downs = swings[swings["direction"] == "DOWN"]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Up moves ≥15%", f"{len(ups)}")
        s2.metric("Down moves ≥15%", f"{len(downs)}")

        if len(ups) > 1:
            _avg_up = ups["days_since_prior"].dropna().mean()
            s3.metric("Avg gap between up moves", f"{_avg_up:.0f} days")
        else:
            s3.metric("Avg gap between up moves", "—")

        if len(downs) > 1:
            _avg_dn = downs["days_since_prior"].dropna().mean()
            s4.metric("Avg gap between down moves", f"{_avg_dn:.0f} days")
        else:
            s4.metric("Avg gap between down moves", "—")

        # Table of moves
        display = swings.copy()
        display["date"] = display["date"].dt.strftime("%Y-%m-%d")
        display["from_date"] = display["from_date"].dt.strftime("%Y-%m-%d")
        display["direction"] = display["direction"].map(lambda d: "🟢 UP" if d == "UP" else "🔴 DOWN")
        display["from_price"] = display["from_price"].map(lambda v: f"${v:,.2f}")
        display["to_price"] = display["to_price"].map(lambda v: f"${v:,.2f}")
        display["move_pct"] = display["move_pct"].map(lambda v: f"{v*100:+.2f}%")
        display["days_since_prior"] = display["days_since_prior"].map(lambda v: "—" if pd.isna(v) else f"{int(v)}")
        display = display.rename(columns={"date": "Date", "direction": "Direction", "from_date": "From Date", "from_price": "From $", "to_price": "To $", "move_pct": "Move", "days_since_prior": "Days since prior"})
        st.dataframe(display, use_container_width=True, hide_index=True)

        # Chart with markers
        # Chart with markers
        import plotly.graph_objects as go
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=tqqq_series.index, y=tqqq_series, name="TQQQ Price", line=dict(color="#4DABF7", width=2)))

        # 1. Historical Swing Events (Peak/Troughs)
        ups_dates = [pd.Timestamp(d) for d in swings.loc[swings["direction"] == "UP", "date"]]
        ups_y = [float(tqqq_series.loc[d]) for d in ups_dates if d in tqqq_series.index]
        downs_dates = [pd.Timestamp(d) for d in swings.loc[swings["direction"] == "DOWN", "date"]]
        downs_y = [float(tqqq_series.loc[d]) for d in downs_dates if d in tqqq_series.index]

        fig2.add_trace(go.Scatter(x=ups_dates, y=ups_y, mode="markers", name=f"Up ≥{threshold_pct}%", marker=dict(color="rgba(0, 230, 118, 0.4)", size=8, symbol="circle")))
        fig2.add_trace(go.Scatter(x=downs_dates, y=downs_y, mode="markers", name=f"Down ≥{threshold_pct}%", marker=dict(color="rgba(239, 83, 80, 0.4)", size=8, symbol="circle")))

        # 2. Actual Strategy Trades (BUY / SELL)
        # Find the specific swing result (we'll use the 'since last exit' version as the canonical one)
        swing_res = None
        for name, r in results.items():
            if "Swing" in name and "since last exit" in name:
                swing_res = r
                break
        
        if swing_res and not swing_res.trades.empty:
            # Get the exact prices at the entry and exit dates
            buy_dates = swing_res.trades["entry_date"]
            buy_prices = [float(tqqq_series.loc[d]) for d in buy_dates if d in tqqq_series.index]
            
            sell_dates = swing_res.trades["exit_date"]
            sell_prices = [float(tqqq_series.loc[d]) for d in sell_dates if d in tqqq_series.index]

            fig2.add_trace(go.Scatter(
                x=buy_dates, y=buy_prices, 
                mode="markers", name="BUY Trade", 
                marker=dict(color="#00E676", size=14, symbol="triangle-up", line=dict(width=2, color="white"))
            ))
            fig2.add_trace(go.Scatter(
                x=sell_dates, y=sell_prices, 
                mode="markers", name="SELL Trade", 
                marker=dict(color="#EF5350", size=14, symbol="triangle-down", line=dict(width=2, color="white"))
            ))

        fig2.update_layout(
            height=440, 
            template="plotly_dark", 
            hovermode="x unified", 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
            yaxis_title="TQQQ Price ($)", 
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---------- Trade logs — one expander per strategy ----------
    for name, r in results.items():
        _display = name
        if "Swing" in name and "Regime-Aware" not in name:
            _parts = name.split(" - ", 1)
            _suffix = _parts[1] if len(_parts) == 2 else ""
            _display = f"Swing {swing_pct:.0f}% - {_suffix}".rstrip(" -")

        with st.expander(f"📋 {_display} — {r.num_trades} trades"):
            if r.trades.empty:
                st.caption("No trades in this window.")
                continue
            display = r.trades.copy()
            display["entry_date"] = pd.to_datetime(display["entry_date"]).dt.strftime("%Y-%m-%d")
            display["exit_date"] = pd.to_datetime(display["exit_date"]).dt.strftime("%Y-%m-%d")
            display["entry_price"] = display["entry_price"].map(lambda v: f"${v:,.2f}")
            display["exit_price"] = display["exit_price"].map(lambda v: f"${v:,.2f}")
            display["shares"] = display["shares"].map(lambda v: f"{v:,.2f}".rstrip("0").rstrip("."))
            display["pnl"] = display["pnl"].map(lambda v: f"${v:+,.2f}")
            display["return_pct"] = display["return_pct"].map(lambda v: f"{v*100:+.2f}%")
            display = display.rename(columns={"entry_date": "Entry", "exit_date": "Exit", "entry_price": "Entry $", "exit_price": "Exit $", "shares": "Shares", "pnl": "P&L", "return_pct": "Return", "days_held": "Days"})
            st.dataframe(display, use_container_width=True, hide_index=True)

    st.caption(
        "Backtest is hypothetical. Past performance does not guarantee future "
        "results. No transaction costs, slippage, or taxes are modeled."
    )

def page_home() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 0.5rem;">
            <div style="font-size: 1.1rem; font-weight: 500; color: #4CAF50; letter-spacing: 0.04em; margin-bottom: 0.35rem;">
                Find Great Businesses at Attractive Prices
            </div>
            <div style="font-size: 2.75rem; font-weight: 700; line-height: 1.1;">
                MoatCheck<span style="font-weight: 400; color: #9aa0a6; font-size: 1.6rem; margin-left: 0.5rem;">— Modern Value Investing Stock Scanner and Advanced Screeners</span>
            </div>
            <div style="font-size: 0.85rem; color: #9aa0a6; margin-top: 0.15rem;">
                Modern Value Investing with base principles rooted in the  Benjamin Graham, David Dodd, Warren Buffett and Charlie Munger.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        ### What this tool does

        MoatCheck brings together four independent lenses on a stock:

        - **🔍 Stock Analyzer** — Big 5 growth screening plus DCF, Peter Lynch Fair Value,
          Graham Number, Graham Formula, and PEG, side-by-side.
        - **📊 Stock Screener** — filter a universe of tickers by valuation and quality rules.
        - **📈 Technical Analysis** — RSI, MACD, Stochastic, SMA, Bollinger Bands,
          volume analysis, mean-reversion strategies, and a BNF-style reversal scanner.
        - **🤖 TQQQ/SQQQ Signals** — seven independent sub-strategies voting on daily
          target allocation, with a position calculator and trade log.

        Use the navigation on the left to switch between tools.
        """
    )
    st.caption("Not investment advice. Cross-check with 10-K filings before any capital decision.")

def compute_annualized_returns(yft, symbol: str) -> dict:
    """Calculate 5, 10, and 20-year annualized total returns.

    yfinance_cache returns a split- and dividend-adjusted 'Close' column and
    does not provide a separate 'Adj Close', so we use 'Close' directly.
    """
    

    try:
        hist = yft.history(period="max")
    except Exception as e:
        st.write("DEBUG history error:", repr(e))
        return {}

    if hist is None or hist.empty:
        return {}

    if "Adj Close" in hist.columns:
        adj = hist["Adj Close"].dropna()
    elif "Close" in hist.columns:
        adj = hist["Close"].dropna()
    else:
        return {}

    if len(adj) < 2:
        return {}

    end_date = adj.index[-1]
    end_price = float(adj.iloc[-1])
    results: dict[int, float | None] = {}

    for years in (5, 10, 20):
        target = end_date - pd.DateOffset(years=years)
        past = adj[adj.index <= target]
        if past.empty:
            results[years] = None
            continue
        start_price = float(past.iloc[-1])
        if start_price <= 0:
            results[years] = None
            continue
        results[years] = (end_price / start_price) ** (1 / years) - 1

    return results

def page_analyzer() -> None:
    render_analyzer()


def page_screener() -> None:
    render_stock_screener()


def page_technical() -> None:
    render_technical_analysis()


def page_tqqq() -> None:
    render_tqqq_sqqq_signals()


# ---------------------------------------------------------------------- #
# Navigation
# ---------------------------------------------------------------------- #

_pages = [
    st.Page(page_home,      title="Home",              icon="🏠", default=True),
    st.Page(page_analyzer,  title="Stock Analyzer",    icon="🔍"),
    st.Page(page_screener,  title="Stock Screener",    icon="📊"),
    st.Page(page_technical, title="Technical Analysis",icon="📈"),
    st.Page(page_tqqq,      title="TQQQ/SQQQ Signals", icon="🤖"),
    st.Page(page_my_investments, title="My Investments", icon="📁"),
    st.Page(page_backtest,        title="Back Test",           icon="🧪"),
]

_nav = st.navigation(_pages, position="sidebar")

# Small brand mark above the nav links
# Small brand mark above the nav links
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.25rem 0 0.5rem; margin-bottom: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.06);">
            <div style="font-size: 1.1rem; font-weight: 700; color: #e8eaed;">MoatCheck</div>
            <div style="font-size: 0.7rem; color: #9aa0a6; letter-spacing: 0.04em;">Value Investing Toolkit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("---")
    st.markdown("**Data Utilities**")
    
    # Form to clear specific ticker
    with st.form("clear_cache_form"):
        st.caption("Clear cached data for a specific stock (e.g., AAPL).")
        ticker_to_clear = st.text_input("Ticker to clear", placeholder="AAPL", label_visibility="collapsed").strip().upper()
        clear_single = st.form_submit_button("Clear this ticker's cache", use_container_width=True)
        
        if clear_single and ticker_to_clear:
            if clear_yfinance_cache(ticker_to_clear):
                st.cache_data.clear()
                st.success(f"Cleared cache for {ticker_to_clear}!")
            else:
                st.info(f"No cache found for {ticker_to_clear}.")

    # Button to nuke everything
    if st.button("🚨 Clear ENTIRE cache", type="primary", use_container_width=True):
        if clear_yfinance_cache(ticker=None):
            st.cache_data.clear()
            st.success("Entire cache cleared!")
            st.rerun()
        else:
            st.info("No cache folders found.")

_nav.run()