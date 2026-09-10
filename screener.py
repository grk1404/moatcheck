"""Stock Screener — Uses built-in universe + MoatCheck deep analysis (No API Key Needed)."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd
import streamlit as st
import yfinance as yf

from moatcheck import compute_big5, fetch, value_price
from moatcheck.fetcher import FetchError
from moatcheck.tickerlist import get_us_stock_universe


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

EXCHANGES = ["All Exchanges", "NYSE", "NASDAQ", "AMEX", "BATS", "OTC"]

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
def fmt_money(v: float | None) -> str:
    if v is None:
        return "n/a"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.2f}"


# ==================== LOAD UNIVERSE (NO API KEY) ====================
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_universe() -> list[str]:
    """Load stock universe from moatcheck (no API key needed)."""
    with st.spinner("📊 Loading stock universe..."):
        df = get_us_stock_universe()
        tickers = df["ticker"].tolist()
        st.success(f"✅ Loaded **{len(tickers):,}** stocks")
        return tickers


@st.cache_data(ttl=3600, show_spinner=False)
def pre_filter_stocks_quick(
    tickers: list[str],
    max_stocks: int = 2000,
    min_mcap: float = 0,
    max_mcap: float = float("inf"),
    exchange: str = "All Exchanges",
    sectors: list[str] = None,
) -> list[str]:
    """Quick pre-filter using yFinance info (no full financials)."""
    st.write("🔄 Pre-filtering stocks (quick check)...")

    if not sectors:
        sectors = []

    qualified = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    batch_size = 50
    total = min(len(tickers), max_stocks)
    tickers_subset = tickers[:total]

    for i in range(0, total, batch_size):
        batch = tickers_subset[i : i + batch_size]
        status_text.text(f"Checking: {i+1}-{min(i+batch_size, total)} of {total}")

        for ticker in batch:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Basic checks
                price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
                market_cap = info.get("marketCap", 0)
                pe = info.get("trailingPE", 0)
                volume = info.get("volume", 0) or info.get("regularMarketVolume", 0)
                stock_exchange = info.get("exchange", "")
                stock_sector = info.get("sector", "")

                # Apply filters
                if price <= 0 or market_cap <= 0 or pe <= 0 or pe >= 50 or volume < 100_000:
                    continue

                if market_cap < min_mcap or market_cap > max_mcap:
                    continue

                if exchange != "All Exchanges" and stock_exchange.upper() != exchange.upper():
                    continue

                if sectors and stock_sector not in sectors:
                    continue

                qualified.append(ticker)

            except Exception:
                pass

        progress_bar.progress(min((i + batch_size) / total, 1.0))

    progress_bar.empty()
    status_text.empty()

    st.write(f"✅ Pre-filter complete: **{len(qualified)}** stocks passed")
    return qualified


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
    if fin.eps.empty:
        return None
    eps_ttm = float(fin.eps.iloc[-1])
    if eps_ttm > 0:
        return eps_ttm
    positive_history = fin.eps[fin.eps > 0]
    if positive_history.empty:
        return None
    return float(positive_history.tail(3).mean())


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
def deep_analyze_ticker(ticker: str) -> ScreenerRow:
    """Full MoatCheck analysis on a single ticker."""
    try:
        # Get the financial data
        fin = fetch(ticker)
        big5 = compute_big5(fin)
        score = big5.wonderfulness().overall

        # Get valuation
        current_eps = positive_eps(fin)
        big5_eps_g = big5_eps_growth(big5)

        val = value_price(
            current_eps=current_eps,
            big5_eps_growth=big5_eps_g,
            analyst_growth=fin.analyst_5yr_growth,
            historical_pe=fin.pe_ratio_ttm,
        )

        value_price_val = val.value_price if val else None
        mos_price_val = val.mos_price if val else None
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
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get("sector")
            industry = info.get("industry")
            
            if not exchange:
                exchange = info.get("exchange", "US")
            
            if market_cap is None:
                market_cap = info.get("marketCap")
            
            if stock_price is None:
                stock_price = info.get("currentPrice") or info.get("regularMarketPrice")
            
            if company_name == ticker:
                company_name = info.get("longName") or info.get("shortName") or ticker
                
        except Exception:
            pass

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
            stock = yf.Ticker(ticker)
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
) -> list[ScreenerRow]:
    """Run deep analysis in parallel with LIVE results display."""
    if not tickers:
        return []

    st.write(f"🔄 Deep analysis of **{len(tickers)}** candidates (parallel processing)...")

    # --- Live results container ---
    live_container = st.empty() if show_live else None

    results = []
    completed = 0
    total = len(tickers)

    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(deep_analyze_ticker, ticker): ticker for ticker in tickers}

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
                            "Score": f"{r.score:.1f}/10" if r.score is not None else "n/a",
                            "Price": fmt_money(r.stock_price),
                            "Value": fmt_money(r.value_price),
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

    results.sort(key=lambda r: (r.score is None, -(r.score or 0)))
    return results


# ==================== DISPLAY ====================
def rows_to_dataframe(rows: list[ScreenerRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Company": r.company,
                "Ticker": r.ticker,
                "Sector": r.sector or "n/a",
                "Industry": r.industry or "n/a",
                "Exchange": r.exchange or "n/a",
                "Market Cap": fmt_money(r.market_cap),
                "Wonderfulness": f"{r.score:.1f}/10" if r.score is not None else "n/a",
                "Stock Price": fmt_money(r.stock_price),
                "Value Price": fmt_money(r.value_price),
                "MOS Price": fmt_money(r.mos_price),
                "P/E": f"{r.pe_ratio:.1f}" if r.pe_ratio else "n/a",
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
    show_full_table: bool
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
    
    # Filter by search term
    if ticker_search:
        search = ticker_search.upper()
        filtered = [
            r for r in filtered
            if search in r.ticker.upper() or search in r.company.upper()
        ]
    
    # Filter out errors
    if not show_full_table:
        filtered = [r for r in filtered if r.error is None]
    
    return filtered


# ==================== MAIN SCREENER UI ====================
def render_stock_screener() -> None:
    """Main screener UI — shows results LIVE as companies are analyzed."""

    st.markdown("### 🏦 Stock Screener - Find Wonderful Businesses")
    st.caption(
        "**Two-phase screening:** "
        "1️⃣ Quick pre-filter using yFinance (basic metrics) "
        "2️⃣ Deep MoatCheck analysis on top candidates — **results appear LIVE!**"
    )

    with st.expander("⚙️ Screener Settings", expanded=True):
        # Elegant single-row filter layout
        col1, col2, col3, col4 = st.columns(4)

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

        with col4:
            st.markdown("**Analysis Depth**")
            max_candidates = st.slider(
                "Candidates",
                min_value=10,
                max_value=200,
                value=100,
                step=10,
                label_visibility="collapsed",
                help="Number of stocks to deep-analyze"
            )

        # Performance settings in a secondary row
        col5, col6, col7 = st.columns([1, 1, 2])
        with col5:
            max_workers = st.slider(
                "Workers",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="Parallel processing speed"
            )
        with col6:
            max_to_check = st.slider(
                "Pre-Filter Range",
                min_value=100,
                max_value=5000,
                value=2000,
                step=100,
                help="Stocks to scan initially"
            )
        with col7:
            show_live_results = st.checkbox(
                "📡 Show LIVE results as they come in",
                value=True,
                help="Display each company immediately after analysis (recommended)"
            )

    if st.button("🔍 Start Screening", type="primary"):
        start_time = time.time()

        # Load universe
        all_tickers = load_stock_universe()

        # Phase 1: Pre-filter
        st.info("📊 **Phase 1:** Quick pre-filtering...")
        
        # If multiple sectors selected, pass them as a list
        sectors_to_filter = sector_filter if sector_filter else []
        
        filtered_tickers = pre_filter_stocks_quick(
            all_tickers,
            max_stocks=max_to_check,
            min_mcap=min_mcap if min_mcap != float("inf") else 0,
            max_mcap=max_mcap if max_mcap != float("inf") else float("inf"),
            exchange=exchange,
            sectors=sectors_to_filter,
        )

        if not filtered_tickers:
            st.warning("No stocks passed pre-filter. Try increasing the range or choosing 'All Sectors'.")
            return

        # Limit candidates
        candidates = filtered_tickers[:max_candidates]
        st.info(f"📊 **Phase 2:** Deep analysis of **{len(candidates)}** candidates...")

        # Phase 2: Deep analysis with live results
        results = deep_analyze_batch(candidates, max_workers, show_live_results)

        elapsed = time.time() - start_time

        st.session_state["screener_results"] = results
        st.session_state["screener_time"] = elapsed
        st.session_state["screener_total_filtered"] = len(filtered_tickers)

        st.success(f"✅ Screening complete in **{elapsed:.1f} seconds**!")

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
            default=["BARGAIN BUY", "BUY", "WATCH", "AVOID"],
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
        show_full_table
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