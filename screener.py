"""Stock Screener — Uses built-in universe + MoatCheck deep analysis (No API Key Needed)."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd
import streamlit as st
from data_provider import get_ticker

from moatcheck import compute_big5, fetch, value_price
from moatcheck.fetcher import FetchError
from moatcheck.tickerlist import (
    get_combined_tickers,
    get_us_stock_tickers,
    get_us_stock_universe,
    get_indian_stock_tickers,
)

import json
from pathlib import Path

SCREENER_CACHE = Path("data/screener_results.json")
SCREENER_TTL = 7 * 24 * 3600  # 7 days


def _load_screener_cache() -> dict[str, dict]:
    if not SCREENER_CACHE.exists():
        return {}
    try:
        return json.loads(SCREENER_CACHE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_screener_cache(cache: dict[str, dict]) -> None:
    SCREENER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SCREENER_CACHE.write_text(json.dumps(cache))

def _row_to_dict(r: ScreenerRow) -> dict:
    return {
        "company": r.company, "ticker": r.ticker, "score": r.score,
        "stock_price": r.stock_price, "value_price": r.value_price,
        "mos_price": r.mos_price, "verdict": r.verdict,
        "pe_ratio": r.pe_ratio, "exchange": r.exchange,
        "sector": r.sector, "industry": r.industry,
        "market_cap": r.market_cap, "error": r.error,
    }


def _row_from_dict(d: dict) -> ScreenerRow:
    """Reconstruct a ScreenerRow, coercing numeric fields to float."""
    def _f(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return ScreenerRow(
        company=d.get("company", ""),
        ticker=d.get("ticker", ""),
        score=_f(d.get("score")),
        stock_price=_f(d.get("stock_price")),
        value_price=_f(d.get("value_price")),
        mos_price=_f(d.get("mos_price")),
        verdict=d.get("verdict", "Unknown"),
        pe_ratio=_f(d.get("pe_ratio")),
        exchange=d.get("exchange"),
        sector=d.get("sector"),
        industry=d.get("industry"),
        market_cap=_f(d.get("market_cap")),
        error=d.get("error"),
    )

# ==================== CONFIGURATION ====================
DEFAULT_MAX_WORKERS = 20
DEFAULT_MAX_CANDIDATES = 100

# Filter Options
MARKET_CAP_RANGES = {
    "All Sizes": (0, float("inf")),
    "Micro Cap": (0, 300_000_000),
    "Small Cap": (300_000_000, 2_000_000_000),
    "Mid Cap": (2_000_000_000, 10_000_000_000),
    "Large Cap": (10_000_000_000, 100_000_000_000),
    "Mega Cap": (100_000_000_000, float("inf")),
}

EXCHANGES = ["All Exchanges", "NYSE", "NASDAQ", "AMEX", "BATS", "OTC", "NSE", "BSE"]

SECTORS = [
    "All Sectors",
    "Technology",
    "Financial Services",
    "Healthcare",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Industrials",
    "Energy",
    "Real Estate",
    "Utilities",
    "Communication Services",
    "Basic Materials",
]


# ==================== DATA CLASSES ====================
@dataclass
class ScreenerRow:
    company: str
    ticker: str
    score: float | None
    stock_price: float | None
    value_price: float | None
    mos_price: float | None
    verdict: str
    pe_ratio: float | None = None
    exchange: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    error: str | None = None


_VERDICT_COLORS = {
    "BARGAIN BUY": "background-color: #0d3d1f; color: #00E676;",
    "BUY": "background-color: #1e4620; color: #b6f0b6;",
    "WATCH": "background-color: #4a3a1e; color: #f0d8a6;",
    "AVOID": "background-color: #4b1e1e; color: #f0b6b6;",
    "Unknown": "background-color: #2a2a2a; color: #9aa0a6;",
    "Error": "background-color: #4b1e1e; color: #f0b6b6;",
}


# ==================== HELPERS ====================
def _currency_symbol(ticker: str) -> str:
    """Return the currency symbol for a ticker based on its suffix."""
    t = (ticker or "").upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return "₹"
    return "$"


def fmt_price(v: float | None, ticker: str = "") -> str:
    """Format a per-share price with the correct currency symbol."""
    if v is None:
        return "n/a"
    symbol = _currency_symbol(ticker)
    return f"{symbol}{v:,.2f}"


def fmt_money(v: float | None, ticker: str = "") -> str:
    if v is None:
        return "n/a"
    symbol = _currency_symbol(ticker)
    if abs(v) >= 1e9:
        return f"{symbol}{v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{symbol}{v / 1e6:.2f}M"
    return f"{symbol}{v:,.2f}"

# ==================== LOAD UNIVERSE (NO API KEY) ====================
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_universe() -> list[str]:
    """Load stock universe from moatcheck (no API key needed)."""
    with st.spinner("📊 Loading stock universe..."):
        df = get_us_stock_universe()
        tickers = df["ticker"].tolist()
        st.success(f"✅ Loaded **{len(tickers):,}** stocks")
        return tickers

# ==================== DEEP ANALYSIS ====================
def big5_eps_growth(big5) -> float | None:
    windows = big5.eps.values
    stable = [v for w, v in windows.items() if w in (5, 3, 10) and v is not None]
    if stable:
        stable.sort()
        n = len(stable)
        if n % 2:
            return stable[n // 2]
        return (stable[n // 2 - 1] + stable[n // 2]) / 2
    one = windows.get(1)
    return one if one is not None else None


def positive_eps(fin) -> float | None:
    """Return a plausible per-share EPS, or None if the data looks corrupted."""
    if fin.eps.empty:
        return None

    eps_ttm = float(fin.eps.iloc[-1])
    if eps_ttm > 0:
        if fin.current_price and fin.current_price > 0:
            # Tightened: a real per-share EPS is never more than 50% of the
            # stock price (that would be a P/E below 2, essentially impossible).
            if abs(eps_ttm) / fin.current_price > 0.5:
                return None
        return eps_ttm

    positive_history = fin.eps[fin.eps > 0]
    if positive_history.empty:
        return None

    candidate = float(positive_history.tail(3).mean())
    if fin.current_price and fin.current_price > 0:
        if abs(candidate) / fin.current_price > 0.5:
            return None
    return candidate


def verdict_price(current: float | None, mos: float | None, value: float | None) -> str:
    if current is None or mos is None or value is None or value <= 0 or mos <= 0:
        return "Unknown"
    if current <= mos * 0.5:
        return "BARGAIN BUY"
    if current <= mos:
        return "BUY"
    if current <= value:
        return "WATCH"
    return "AVOID"


@st.cache_data(ttl=3600, show_spinner=False)
def deep_analyze_ticker(ticker: str, thresholds_tuple: tuple) -> ScreenerRow:
    """Full MoatCheck analysis on a single ticker."""
    # Reconstruct thresholds dict from the tuple (Streamlit cache needs hashable args)
    thresholds = dict(thresholds_tuple) if thresholds_tuple else None
    try:
        # Get the financial data
        fin = fetch(ticker)
        # Normalize exchange for display and exchange-based filtering
        _EXCHANGE_DISPLAY = {
            "NSI": "NSE", "BSE": "BSE",
            "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ",
            "NYQ": "NYSE", "PCX": "NYSE Arca", "ASE": "AMEX",
        }
        big5 = compute_big5(fin, thresholds=thresholds)
        score = big5.wonderfulness().overall

        # Get valuation
        current_eps = positive_eps(fin)
         # Sanity guard: EPS must be plausibly per-share.
        # A per-share EPS cannot exceed the stock price by more than ~2×
        # (even the richest company has a P/E above 0.5). If it does,
        # the data source returned a total, not a per-share value.
        if current_eps is not None and fin.current_price and fin.current_price > 0:
            eps_to_price = current_eps / fin.current_price
            if eps_to_price > 0.5 or eps_to_price < -0.5:
                current_eps = None

         # --- Sanity guard: P/E must be plausible ---
        historical_pe = fin.pe_ratio_ttm
        if historical_pe is not None and (historical_pe <= 0 or historical_pe > 200):
            historical_pe = None
            
        big5_eps_g = big5_eps_growth(big5)

        val = value_price(
            current_eps=current_eps,
            big5_eps_growth=big5_eps_g,
            analyst_growth=fin.analyst_5yr_growth,
            historical_pe=historical_pe,
        )

        value_price_val = val.value_price if val else None
        mos_price_val = val.mos_price if val else None
        verdict = verdict_price(fin.current_price, mos_price_val, value_price_val)

        # Sanity guard: an intrinsic value more than 5× the market price is
        # almost always bad input data, not a real opportunity.
        if value_price_val is not None and fin.current_price and fin.current_price > 0:
            if value_price_val / fin.current_price > 5.0:
                value_price_val = None
                mos_price_val = None

        verdict = verdict_price(fin.current_price, mos_price_val, value_price_val)

        # Always fetch additional info from yfinance as fallback
        sector = None
        industry = None
        market_cap = fin.market_cap
        exchange = fin.exchange
        stock_price = fin.current_price
        pe_ratio = fin.pe_ratio_ttm
        company_name = fin.company_name or ticker

        # Get fresh info from yfinance for sector/industry
        try:
            stock = get_ticker(ticker)
            info = stock.info
            sector = info.get("sector")
            industry = info.get("industry")
            
            if ticker.upper().endswith(".NS"):
                exchange = "NSE"
            elif ticker.upper().endswith(".BO"):
                exchange = "BSE"
            elif exchange in _EXCHANGE_DISPLAY:
                exchange = _EXCHANGE_DISPLAY[exchange]
            
            if market_cap is None:
                market_cap = info.get("marketCap")
            
            if stock_price is None:
                stock_price = info.get("currentPrice") or info.get("regularMarketPrice")
            
            if company_name == ticker:
                company_name = info.get("longName") or info.get("shortName") or ticker
                
        except Exception:
            pass

        #print(f"DEBUG {ticker}: current_eps={current_eps}, big5_eps_g={big5_eps_g}, value_price={value_price_val}, mos_price={mos_price_val}, market_cap={market_cap}")

        return ScreenerRow(
            company=company_name,
            ticker=fin.ticker,
            score=score,
            stock_price=stock_price,
            value_price=value_price_val,
            mos_price=mos_price_val,
            verdict=verdict,
            pe_ratio=pe_ratio,
            exchange=exchange,
            sector=sector,
            industry=industry,
            market_cap=market_cap,
            error=None,
        )
    except (FetchError, Exception) as e:
        # If analysis fails, try to get at least basic info from yfinance
        company_name = ticker
        stock_price = None
        market_cap = None
        exchange = None
        sector = None
        industry = None
        pe_ratio = None
        
        try:
            stock = get_ticker(ticker)
            info = stock.info
            company_name = info.get("longName") or info.get("shortName") or ticker
            stock_price = info.get("currentPrice") or info.get("regularMarketPrice")
            market_cap = info.get("marketCap")
            exchange = info.get("exchange")
            sector = info.get("sector")
            industry = info.get("industry")
            pe_ratio = info.get("trailingPE")
        except Exception:
            pass

        return ScreenerRow(
            company=company_name,
            ticker=ticker,
            score=None,
            stock_price=stock_price,
            value_price=None,
            mos_price=None,
            verdict="Error",
            pe_ratio=pe_ratio,
            exchange=exchange,
            sector=sector,
            industry=industry,
            market_cap=market_cap,
            error=str(e),
        )


def deep_analyze_batch(
    tickers: list[str],
    max_workers: int = DEFAULT_MAX_WORKERS,
    show_live: bool = True,
    thresholds: dict | None = None,
) -> list[ScreenerRow]:
    """Run deep analysis in parallel with LIVE results display."""
    if not tickers:
        return []

    cache = _load_screener_cache()
    now = time.time()

    to_run: list[str] = []
    from_cache: list[ScreenerRow] = []
    for t in tickers:
        entry = cache.get(t)
        if entry and (now - entry.get("cached_at", 0) < SCREENER_TTL):
            from_cache.append(_row_from_dict(entry["row"]))
        else:
            to_run.append(t)

    if not to_run:
        st.info("✅ All results loaded from cache")
        all_results = list(from_cache)
        all_results.sort(key=lambda r: (r.score is None, -(r.score or 0)))
        return all_results

    st.write(f"🔄 Deep analysis of **{len(tickers)}** candidates (parallel processing)...")

    # --- Live results container ---
    live_container = st.empty() if show_live else None

    results = []
    completed = 0
    total = len(to_run)

    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        thresholds_tuple = tuple(sorted((thresholds or {}).items()))
        future_to_ticker = {
            executor.submit(deep_analyze_ticker, ticker, thresholds_tuple): ticker
            for ticker in to_run
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(
                    ScreenerRow(
                        company=ticker,
                        ticker=ticker,
                        score=None,
                        stock_price=None,
                        value_price=None,
                        mos_price=None,
                        verdict="Error",
                        error=str(e),
                    )
                )

            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"Analyzing: {completed}/{total} - {ticker}")

            # --- LIVE UPDATE: Show results as they come in ---
            if show_live and live_container is not None:
                # Sort current results by score
                sorted_results = sorted(
                    [r for r in results if r.error is None],
                    key=lambda x: -(x.score or 0)
                )
                
                # Create DataFrame from current results
                if sorted_results:
                    df_live = pd.DataFrame([
                        {
                            "Company": r.company[:30] + "..." if len(r.company) > 30 else r.company,
                            "Ticker": r.ticker,
                            "Score": f"{_safe_float(r.score):.1f}/10" if _safe_float(r.score) is not None else "n/a",
                            "Price": fmt_price(_safe_float(r.stock_price), r.ticker),
                            "Value": fmt_price(_safe_float(r.value_price), r.ticker),
                            "MOS Price": fmt_price(_safe_float(r.mos_price), r.ticker),
                            "Verdict": r.verdict,
                        }
                        for r in sorted_results[:20]
                    ])
                    
                    def style_live_row(row):
                        style = _VERDICT_COLORS.get(row["Verdict"], "")
                        return [style] * len(row)
                    
                    live_container.dataframe(
                        df_live.style.apply(style_live_row, axis=1),
                        use_container_width=True,
                        hide_index=True,
                        height=min(400, 25 * len(df_live) + 40),
                    )
                else:
                    live_container.info(f"⏳ Waiting for first results... ({completed}/{total})")

    progress_bar.empty()
    status_text.empty()

    # After analysis, merge and save cache
    all_results = from_cache + results
    new_cache = {**cache}
    for r in results:
        new_cache[r.ticker] = {
            "cached_at": now,
            "row": _row_to_dict(r),
        }
    _save_screener_cache(new_cache)

    all_results.sort(key=lambda r: (r.score is None, -(r.score or 0)))
    return all_results


# ==================== DISPLAY ====================
def _safe_float(v) -> float | None:
    """Coerce a value to float, returning None if it can't be converted."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        f = float(v)
        return f if f == f else None   # reject NaN
    except (TypeError, ValueError):
        return None


def rows_to_dataframe(rows: list[ScreenerRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Company": r.company,
                "Ticker": r.ticker,
                "Sector": r.sector or "n/a",
                "Industry": r.industry or "n/a",
                "Exchange": r.exchange or "n/a",
                "Market Cap": fmt_money(_safe_float(r.market_cap), r.ticker),
                "Wonderfulness": f"{_safe_float(r.score):.1f}/10" if _safe_float(r.score) is not None else "n/a",
                "Stock Price": fmt_price(_safe_float(r.stock_price), r.ticker),
                "Value Price": fmt_price(_safe_float(r.value_price), r.ticker),
                "MOS Price": fmt_price(_safe_float(r.mos_price), r.ticker),
                "P/E": f"{_safe_float(r.pe_ratio):.1f}" if _safe_float(r.pe_ratio) is not None else "n/a",
                "Verdict": r.verdict,
            }
            for r in rows
        ]
    )


def style_screener_table(df: pd.DataFrame):
    def style_row(row):
        style = _VERDICT_COLORS.get(row["Verdict"], "")
        return [style] * len(row)

    return df.style.apply(style_row, axis=1)


def display_screener_stats(rows: list[ScreenerRow]):
    col1, col2, col3, col4 = st.columns(4)

    total = len(rows)
    successful = len([r for r in rows if r.error is None])
    bargain_buys = len([r for r in rows if r.verdict == "BARGAIN BUY"])
    buys = len([r for r in rows if r.verdict in ["BARGAIN BUY", "BUY"]])

    with col1:
        st.metric("Total Analyzed", total)
    with col2:
        st.metric("Successful", successful)
    with col3:
        st.metric("Buy Candidates", buys)
    with col4:
        st.metric("Bargain Buys", bargain_buys)


def apply_filters(
    rows: list[ScreenerRow], 
    min_score: float, 
    verdict_filter: list, 
    sector_filter: list,
    exchange_filter: list,
    ticker_search: str, 
    show_full_table: bool,
    mcap_min: float = 0,
    mcap_max: float = float("inf"),
) -> list[ScreenerRow]:
    """Apply filters to the results."""
    filtered = rows.copy()
    
    # Filter by score
    if min_score > 0:
        filtered = [r for r in filtered if r.score is not None and r.score >= min_score]
    
    # Filter by verdict
    if verdict_filter:
        filtered = [r for r in filtered if r.verdict in verdict_filter]
    
    # Filter by sector (multi-select)
    if sector_filter:
        filtered = [r for r in filtered if r.sector in sector_filter]
    
    # Filter by exchange (multi-select)
    if exchange_filter:
        filtered = [r for r in filtered if r.exchange in exchange_filter]

    # Market cap filter (applied at display time)
    if mcap_min > 0:
        filtered = [r for r in filtered if r.market_cap is not None and r.market_cap >= mcap_min]
    if mcap_max < float("inf"):
        filtered = [r for r in filtered if r.market_cap is not None and r.market_cap <= mcap_max]
        
    # Filter by search term
    if ticker_search:
        search = ticker_search.upper()
        filtered = [r for r in filtered if search in r.ticker.upper() or search in r.company.upper()]
    
    # Filter out errors
    if not show_full_table:
        filtered = [r for r in filtered if r.error is None]
    
    return filtered


# ==================== MAIN SCREENER UI ====================
def render_stock_screener() -> None:
    """Main screener UI — shows results LIVE as companies are analyzed."""

    st.markdown("### 🏦 Stock Screener - Find Wonderful Businesses")
    st.caption(
      "**Full-universe screening:** Scans every US-listed stock via SEC EDGAR + "
        "yFinance. Results are cached for 7 days — reruns are incremental."
    )

    with st.expander("⚙️ Screener Settings", expanded=True):
        # Elegant single-row filter layout
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Company Size**")
            mcap_preset = st.selectbox(
                "Market Cap Range",
                options=list(MARKET_CAP_RANGES.keys()),
                index=0,
                label_visibility="collapsed",
            )
            min_mcap, max_mcap = MARKET_CAP_RANGES[mcap_preset]

        with col2:
            st.markdown("**Exchange**")
            exchange = st.selectbox(
                "Exchange",
                options=EXCHANGES,
                index=0,
                label_visibility="collapsed",
            )

        with col3:
            st.markdown("**Sector**")
            # Multi-select for sectors
            sector_filter = st.multiselect(
                "Select Sectors",
                options=[s for s in SECTORS if s != "All Sectors"],
                default=[],
                label_visibility="collapsed",
                help="Select one or more sectors. Leave empty for all sectors.",
                placeholder="All Sectors",
            )        
        # Performance settings in a secondary row
        col4, col5 = st.columns([1, 3])
        with col4:
            max_workers = st.slider(
                "Workers",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="Parallel processing speed"
            )        
        with col5:
            show_live_results = st.checkbox(
                "📡 Show LIVE results as they come in",
                value=True,
                help="Display each company immediately after analysis (recommended)"
            )
        include_india = st.checkbox(
            "🇮🇳 Include Indian stocks (NSE)",
            value=False,
            help=(
                "Adds ~2,000 NSE-listed tickers. Note: EDGAR doesn't cover Indian "
                "companies, so the Big 5 will show n/a for long-history metrics. "
                "Technical signals and intrinsic value still work."
            ),
        )

        force_refresh = st.checkbox(
            "🔄 Force fresh scan (ignore cache)",
            value=False,
            help="Ignore all cached results and re-analyze every ticker from scratch."
        )

        # --- Big 5 growth thresholds ---
        with st.expander("📊 Big 5 growth thresholds", expanded=False):
            st.caption(
                "Set a minimum and maximum acceptable growth rate for each metric. "
                "A company passes a metric only if its CAGR falls within [Min, Max] "
                "for **every** computed window. Changes take effect on the next run."
            )

            tcol1, tcol2, tcol3, tcol4, tcol5 = st.columns(5)

            with tcol1:
                st.markdown("**Revenue**")
                rev_min = st.number_input(
                    "Rev min", min_value=0.0, max_value=100.0,
                    value=10.0, step=1.0, format="%.0f",
                    key="scr_rev_min", label_visibility="collapsed",
                ) / 100
                rev_max = st.number_input(
                    "Rev max", min_value=0.0, max_value=100.0,
                    value=50.0, step=1.0, format="%.0f",
                    key="scr_rev_max", label_visibility="collapsed",
                ) / 100

            with tcol2:
                st.markdown("**EPS**")
                eps_min = st.number_input(
                    "EPS min", min_value=0.0, max_value=100.0,
                    value=10.0, step=1.0, format="%.0f",
                    key="scr_eps_min", label_visibility="collapsed",
                ) / 100
                eps_max = st.number_input(
                    "EPS max", min_value=0.0, max_value=100.0,
                    value=50.0, step=1.0, format="%.0f",
                    key="scr_eps_max", label_visibility="collapsed",
                ) / 100

            with tcol3:
                st.markdown("**Equity**")
                eq_min = st.number_input(
                    "Eq min", min_value=0.0, max_value=100.0,
                    value=10.0, step=1.0, format="%.0f",
                    key="scr_eq_min", label_visibility="collapsed",
                ) / 100
                eq_max = st.number_input(
                    "Eq max", min_value=0.0, max_value=100.0,
                    value=50.0, step=1.0, format="%.0f",
                    key="scr_eq_max", label_visibility="collapsed",
                ) / 100

            with tcol4:
                st.markdown("**OCF**")
                ocf_min = st.number_input(
                    "OCF min", min_value=0.0, max_value=100.0,
                    value=10.0, step=1.0, format="%.0f",
                    key="scr_ocf_min", label_visibility="collapsed",
                ) / 100
                ocf_max = st.number_input(
                    "OCF max", min_value=0.0, max_value=100.0,
                    value=50.0, step=1.0, format="%.0f",
                    key="scr_ocf_max", label_visibility="collapsed",
                ) / 100

            with tcol5:
                st.markdown("**ROIC**")
                roic_min = st.number_input(
                    "ROIC min", min_value=0.0, max_value=100.0,
                    value=10.0, step=1.0, format="%.0f",
                    key="scr_roic_min", label_visibility="collapsed",
                ) / 100
                roic_max = st.number_input(
                    "ROIC max", min_value=0.0, max_value=100.0,
                    value=50.0, step=1.0, format="%.0f",
                    key="scr_roic_max", label_visibility="collapsed",
                ) / 100

        # --- Additional quality filters ---
        with st.expander("📊 Additional quality filters", expanded=False):
            st.caption(
                "One-sided floors. A company passes if its metric is **at or above** "
                "the value entered here. Changes take effect on the next run."
            )

            acol1, acol2, acol3 = st.columns(3)

            with acol1:
                st.markdown("**Cash Conversion ≥**")
                cash_conv_min = st.number_input(
                    "Cash Conversion", min_value=0.0, max_value=500.0,
                    value=80.0, step=5.0, format="%.0f",
                    key="scr_cash_conv", label_visibility="collapsed",
                ) / 100

            with acol2:
                st.markdown("**Discount to FV (FCF) ≥**")
                disc_fcf_min = st.number_input(
                    "Discount FCF", min_value=-100.0, max_value=100.0,
                    value=30.0, step=5.0, format="%.0f",
                    key="scr_disc_fcf", label_visibility="collapsed",
                ) / 100

            with acol3:
                st.markdown("**Discount to FV (NP) ≥**")
                disc_np_min = st.number_input(
                    "Discount NP", min_value=-100.0, max_value=100.0,
                    value=30.0, step=5.0, format="%.0f",
                    key="scr_disc_np", label_visibility="collapsed",
                ) / 100

        # --- Combine into the thresholds dict (after both expanders) ---
        screener_thresholds = {
            "revenue":         {"min": rev_min,       "max": rev_max},
            "eps":             {"min": eps_min,       "max": eps_max},
            "equity":          {"min": eq_min,        "max": eq_max},
            "ocf":             {"min": ocf_min,       "max": ocf_max},
            "roic":            {"min": roic_min,      "max": roic_max},
            "cash_conversion": {"min": cash_conv_min, "max": 999.0},
            "discount_fcf":    {"min": disc_fcf_min,  "max": 999.0},
            "discount_np":     {"min": disc_np_min,   "max": 999.0},
        }

    st.caption(
        "**Active filters:** "
        f"Revenue {screener_thresholds['revenue']['min']*100:.0f}–{screener_thresholds['revenue']['max']*100:.0f}% · "
        f"EPS {screener_thresholds['eps']['min']*100:.0f}–{screener_thresholds['eps']['max']*100:.0f}% · "
        f"Equity {screener_thresholds['equity']['min']*100:.0f}–{screener_thresholds['equity']['max']*100:.0f}% · "
        f"OCF {screener_thresholds['ocf']['min']*100:.0f}–{screener_thresholds['ocf']['max']*100:.0f}% · "
        f"ROIC {screener_thresholds['roic']['min']*100:.0f}–{screener_thresholds['roic']['max']*100:.0f}% · "
        f"Cash Conversion ≥ {screener_thresholds['cash_conversion']['min']*100:.0f}% · "
        f"Discount FCF ≥ {screener_thresholds['discount_fcf']['min']*100:.0f}% · "
        f"Discount NP ≥ {screener_thresholds['discount_np']['min']*100:.0f}%"
    )

    if st.button("🔍 Start Screening", type="primary"):
        start_time = time.time()

        if force_refresh:
            # Nuke both caches
            st.cache_data.clear()                # Streamlit memory cache
            if SCREENER_CACHE.exists():
                SCREENER_CACHE.unlink()          # Disk cache
        
        all_tickers = get_combined_tickers(include_us=True, include_india=include_india)
        region = "US + India" if include_india else "US"
        st.info(f"📊 **Universe:** {len(all_tickers):,} {region}-listed stocks — running full scan")

        results = deep_analyze_batch(
            all_tickers,
            max_workers,
            show_live_results,
            thresholds=screener_thresholds,
        )

        elapsed = time.time() - start_time
        st.session_state["screener_results"] = results
        st.session_state["screener_time"] = elapsed
        st.success(f"✅ Scanned **{len(all_tickers):,}** stocks in **{elapsed:.1f}s**")

    # Display final results
    if "screener_results" not in st.session_state:
        st.info("👆 Click **Start Screening** to find wonderful businesses!")
        return

    rows: list[ScreenerRow] = st.session_state["screener_results"]

    if not rows:
        st.warning("No results to display.")
        return

    # Stats
    display_screener_stats(rows)

    # ========== FILTER SECTION (AUTO-APPLY, NO BUTTON) ==========
    st.markdown("### 🔍 Filter Results")
    st.caption("Adjust filters below - results update automatically")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        min_score = st.slider(
            "Minimum Wonderfulness Score",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            help="Only show stocks with score >= this value"
        )
        
        # Multi-select for verdicts
        verdict_filter = st.multiselect(
            "Verdict",
            options=["BARGAIN BUY", "BUY", "WATCH", "AVOID", "Unknown", "Error"],
            default=["BARGAIN BUY", "BUY", "WATCH", "AVOID", "Unknown", "Error"],
            help="Select which verdicts to show"
        )
    
    with filter_col2:
        # Multi-select for sectors in results
        available_sectors = sorted(set([r.sector for r in rows if r.sector]))
        
        sector_result_filter = st.multiselect(
            "Sector",
            options=available_sectors,
            default=[],
            help="Filter by sector. Leave empty for all sectors.",
            placeholder="All Sectors",
        )
        
        # Multi-select for exchanges in results
        available_exchanges = sorted(set([r.exchange for r in rows if r.exchange]))
        
        exchange_result_filter = st.multiselect(
            "Exchange",
            options=available_exchanges,
            default=[],
            help="Filter by exchange. Leave empty for all exchanges.",
            placeholder="All Exchanges",
        )
    
    with filter_col3:
        ticker_search = st.text_input(
            "Search Ticker/Company",
            placeholder="e.g., AAPL or Apple",
            help="Search by ticker or company name"
        )
        
        # Show full table toggle
        show_full_table = st.checkbox(
            "Show all results (including errors)",
            value=True,
            help="When unchecked, hides stocks that failed analysis"
        )

    # Auto-apply filters (no button needed!)
    filtered_rows = apply_filters(
        rows, 
        min_score, 
        verdict_filter, 
        sector_result_filter, 
        exchange_result_filter,
        ticker_search, 
        show_full_table,
        mcap_min=min_mcap,
        mcap_max=max_mcap,
    )
    
    # Show count
    st.caption(f"📊 Showing **{len(filtered_rows)}** of **{len(rows)}** results")
    
    # Display filtered results
    if filtered_rows:
        df_filtered = rows_to_dataframe(filtered_rows)
        st.dataframe(
            style_screener_table(df_filtered),
            use_container_width=True,
            hide_index=True,
            height=600,
        )
    else:
        st.warning("No results match your filters. Try adjusting the criteria above.")
        df_filtered = pd.DataFrame()

    # ========== EXPORT AND CLEAR ==========
    col_export, col_clear = st.columns(2)
    
    with col_export:
        if st.button("📥 Export to CSV"):
            # Export the filtered results
            if filtered_rows:
                csv_data = rows_to_dataframe(filtered_rows)
            else:
                csv_data = rows_to_dataframe(rows)
            csv = csv_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"screener_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    with col_clear:
        if st.button("🗑️ Clear Results"):
            st.session_state.pop("screener_results", None)
            st.session_state.pop("screener_time", None)
            st.rerun()

    # ========== ERRORS ==========
    errors = [r for r in rows if r.error]
    if errors and st.checkbox(f"⚠️ Show {len(errors)} errors"):
        with st.expander(f"❌ {len(errors)} ticker(s) had errors"):
            for r in errors[:50]:
                st.markdown(f"**{r.ticker}** — {r.error}")
            if len(errors) > 50:
                st.markdown(f"... and {len(errors) - 50} more errors")