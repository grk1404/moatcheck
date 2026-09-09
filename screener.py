"""Stock Screener tab — scan Value style company universes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from moatcheck import compute_big5, fetch, value_price
from moatcheck.fetcher import FetchError

from moatcheck.tickerlist import get_us_stock_tickers, get_us_stock_universe

# Curated universe of wide-moat businesses commonly used in Value analysis.
# moat_TICKERS: list[str] = get_us_stock_tickers()

# @st.cache_data(ttl=86400)
# def get_all_us_stocks() -> list[str]:
#     st.write("Searching for all US Stock Universe...")
#     df = get_us_stock_universe()
#     # Displays: Dataset Size: (3, 2)
#     st.write("Dataset Size:", df.shape)
#     return df["ticker"].tolist()

# def get_all_us_stock_tickers() -> list[str]:
#     #Return a list of all US stock tickers.
#     st.write("Searching for all US Stock Tickers...")
#     tlist = []
#     tlist = get_us_stock_tickers()
#     st.write("Dataset Size:", len(tlist))
#     return tlist

# #SCREENER_UNIVERSES: dict[str, list[str]] = {
# SCREENER_UNIVERSES = {
#     "All US Stocks": get_all_us_stock_tickers(),
# }

SCREENER_UNIVERSES = {
    "Wide-Moat Companies": {
        "description": "High-quality businesses with strong Big 5 metrics",
        "min_score": 7.0,
    },
    "Growth Companies": {
        "description": "Companies with strong growth characteristics",
        "min_score": 6.0,
    },
    "Value Companies": {
        "description": "Companies trading near or below Value Price",
        "min_score": 5.0,
    },
    "My Fav Stocks": {
        "description": "Broad US stock universe",
        "min_score": 0.0,
    },
}

_VERDICT_COLORS = {
    "BARGAIN BUY": "background-color: #0d3d1f; color: #00E676;",
    "BUY": "background-color: #1e4620; color: #b6f0b6;",
    "WATCH": "background-color: #4a3a1e; color: #f0d8a6;",
    "AVOID": "background-color: #4b1e1e; color: #f0b6b6;",
    "Unknown": "background-color: #2a2a2a; color: #9aa0a6;",
    "Error": "background-color: #4b1e1e; color: #f0b6b6;",
}


@dataclass
class ScreenerRow:
    company: str
    ticker: str
    score: float | None
    stock_price: float | None
    value_price: float | None
    verdict: str
    error: str | None = None


def _fmt_money(v: float | None) -> str:
    if v is None:
        return "n/a"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.2f}"


def _big5_eps_growth(big5) -> float | None:
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


def _verdict_price(
    current: float | None, mos: float | None, value: float | None
) -> str:
    if current is None or mos is None or value is None or value <= 0 or mos <= 0:
        return "Unknown"
    if current <= mos * 0.5:
        return "BARGAIN BUY"
    if current <= mos:
        return "BUY"
    if current <= value:
        return "WATCH"
    return "AVOID"


def _positive_eps(fin) -> float | None:
    if fin.eps.empty:
        return None
    eps_ttm = float(fin.eps.iloc[-1])
    if eps_ttm > 0:
        return eps_ttm
    positive_history = fin.eps[fin.eps > 0]
    if positive_history.empty:
        return None
    return float(positive_history.tail(3).mean())


@st.cache_data(ttl=3600, show_spinner=False)
def _screen_ticker(ticker: str) -> ScreenerRow:
    try:
        fin = fetch(ticker)
        big5 = compute_big5(fin)
        score = big5.wonderfulness().overall
        current_eps = _positive_eps(fin)
        big5_eps_g = _big5_eps_growth(big5)
        val = value_price(
            current_eps=current_eps,
            big5_eps_growth=big5_eps_g,
            analyst_growth=fin.analyst_5yr_growth,
            historical_pe=fin.pe_ratio_ttm,
        )
        value = val.value_price if val else None
        mos = val.mos_price if val else None
        verdict = _verdict_price(fin.current_price, mos, value)
        return ScreenerRow(
            company=fin.company_name or ticker,
            ticker=fin.ticker,
            score=score,
            stock_price=fin.current_price,
            value_price=value,
            verdict=verdict,
        )
    except (FetchError, Exception) as e:
        return ScreenerRow(
            company=ticker,
            ticker=ticker,
            score=None,
            stock_price=None,
            value_price=None,
            verdict="Error",
            error=str(e),
        )


def _run_screen(tickers: list[str]) -> list[ScreenerRow]:
    rows: list[ScreenerRow] = []
    progress = st.progress(0, text="Screening companies…")
    for i, ticker in enumerate(tickers):
        progress.progress((i + 1) / len(tickers), text=f"Screening {ticker}…")
        rows.append(_screen_ticker(ticker))
    progress.empty()
    rows.sort(
        key=lambda r: (
            r.score is None,
            -(r.score or 0),
            r.ticker,
        )
    )
    return rows


def _rows_to_dataframe(rows: list[ScreenerRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Company": r.company,
                "Ticker": r.ticker,
                "Company Score": f"{r.score:.1f}/10" if r.score is not None else "n/a",
                "Stock Price": _fmt_money(r.stock_price),
                "Value Price": _fmt_money(r.value_price),
                "Verdict": r.verdict,
            }
            for r in rows
        ]
    )


def _style_screener_table(df: pd.DataFrame):
    def _style_row(row):
        style = _VERDICT_COLORS.get(row["Verdict"], "")
        return [style] * len(row)

    return df.style.apply(_style_row, axis=1)


def render_stock_screener() -> None:
    st.markdown("### Stock Screener")
    st.caption(
        "Scan for high-quality businesses aka Great Businesses. Each row shows the wonderfulness "
        "score (Big 5 griwth rates), current stock price, Value price, and buy/watch/avoid verdict."
    )

    universe = st.radio(
        "Universe",
        options=list(SCREENER_UNIVERSES.keys()),
        horizontal=True,
        key="screener_universe",
    )

    search = st.button("Search", type="primary", key="screener_search")

    if search:
        st.write("Loading US stock universe...")
        universe_df = get_us_stock_universe()

        st.write(f"Universe contains {len(universe_df):,} securities.")

        # TEMPORARY TEST — only screen first 20
        tickers = universe_df["ticker"].head(20).tolist()

        st.write("Testing tickers:", tickers)
        # End Test
    
        st.session_state["screener_results"] = _run_screen(SCREENER_UNIVERSES[universe])
        st.session_state["screener_universe_ran"] = universe

    if "screener_results" not in st.session_state:
        st.info("Select **Value companies** and click **Search** to screen the universe.")
        return

    if st.session_state.get("screener_universe_ran") != universe:
        st.warning("Universe changed — click **Search** to refresh results.")

    rows: list[ScreenerRow] = st.session_state["screener_results"]
    errors = [r for r in rows if r.error]
    st.markdown(f"**{len(rows)} companies** · sorted by company score (highest first)")

    df = _rows_to_dataframe(rows)
    st.dataframe(
        _style_screener_table(df),
        use_container_width=True,
        hide_index=True,
    )

    if errors:
        with st.expander(f"{len(errors)} ticker(s) could not be screened"):
            for r in errors:
                st.markdown(f"**{r.ticker}** — {r.error}")
