from __future__ import annotations

import pandas as pd
import streamlit as st

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
# import inspect

# st.write("value_price loaded from:", inspect.getsourcefile(value_price))
# st.write("value_price signature:", inspect.signature(value_price))

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


def _suggest_ticker(symbol: str) -> str | None:
    symbol = symbol.strip().upper()
    return _COMMON_TICKER_FIXES.get(symbol)


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%"


def _fmt_money(v: float | None, exchange: str = "") -> str:
    if v is None:
        return "n/a"

    # Determine currency symbol
    if exchange in ("NSE", "BSE"):
        currency = "₹ "
    else:
        currency = "$ "
    if abs(v) >= 1e9:
        return f"{currency}{v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{currency}{v / 1e6:.2f}M"
    return f"{currency}{v:,.2f}"


def _color_pass(val: float | None) -> str:
    if val is None:
        return "color: #888;"
    if val >= PASS_THRESHOLD:
        return "background-color: #1e4620; color: #b6f0b6;"
    return "background-color: #4b1e1e; color: #f0b6b6;"


# def _render_big5_table(big5: Big5Result) -> None:
#     df = big5.as_dataframe()
#     display = df.copy()
#     for w in WINDOWS:
#         col = f"{w}yr"
#         display[col] = display[col].map(_fmt_pct)
#     display["pass"] = display["pass"].map(lambda p: "PASS" if p else "FAIL")
#     display = display.rename(columns={"metric": "Metric", "pass": "Verdict"})

#     def _style(row):
#         original = df.loc[row.name]
#         out = [""]  # Metric column
#         for w in WINDOWS:
#             out.append(_color_pass(original[f"{w}yr"]))
#         out.append(
#             "background-color: #1e4620; color: #b6f0b6;"
#             if original["pass"]
#             else "background-color: #4b1e1e; color: #f0b6b6;"
#         )
#         return out
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
    st.dataframe(styled, use_container_width=True, hide_index=True)


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
            return stable[n // 2]
        return (stable[n // 2 - 1] + stable[n // 2]) / 2
    # Fallback: 1yr only when nothing longer is computable
    one = windows.get(1)
    return one if one is not None else None


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


# #def _inject_tab_styles() -> None:
#     """Rounded-square tab styling (also in .streamlit/style.css for reload)."""
#     st.markdown(
#         """
#         <style>
#         .stTabs [data-baseweb="tab-list"] {
#             gap: 0.75rem;
#             background-color: transparent;
#             border-bottom: none;
#             padding-bottom: 0.25rem;
#         }
#         .stTabs [data-baseweb="tab-border"] { display: none; }
#         .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"],
#         .stTabs [data-baseweb="tab-list"] button[role="tab"] {
#             height: auto;
#             min-height: 2.75rem;
#             background-color: rgba(255, 255, 255, 0.05);
#             border: 1.5px solid rgba(255, 255, 255, 0.14);
#             border-radius: 10px;
#             padding: 0.65rem 1.5rem;
#             color: #9aa0a6;
#             font-weight: 600;
#             font-size: 0.95rem;
#         }
#         .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"]:hover,
#         .stTabs [data-baseweb="tab-list"] button[role="tab"]:hover {
#             background-color: rgba(255, 255, 255, 0.09);
#             border-color: rgba(76, 175, 80, 0.5);
#             color: #e8eaed;
#         }
#         .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"],
#         .stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
#             background-color: rgba(76, 175, 80, 0.2);
#             border-color: #4CAF50;
#             color: #b6f0b6;
#             box-shadow: 0 0 0 1px rgba(76, 175, 80, 0.3);
#         }
#         .stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

# RG new function replaces the above function for makign TABS prominent.
def _inject_tab_styles() -> None:
    """Inject attached light-grey rounded tab headers with a full content border box."""
    st.markdown(
        """
        <style>
        /* 1. Main Tab Bar Container Alignment */
        div[data-testid="stTabs"] {
            margin-top: 0.5rem;
        }

        /* Remove spacing gap between tabs and flush them together */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px !important;
            background-color: transparent !important;
            border-bottom: none !important;
            padding: 0px !important;
            margin-bottom: -2px !important; /* Overlap header seamlessly with panel border */
            z-index: 2 !important;
        }

        /* Hide Streamlit's default red/blue underline */
        .stTabs [data-baseweb="tab-border"],
        .stTabs [data-baseweb="tab-highlight-title"] {
            display: none !important;
        }

        /* 2. Inactive Tabs: Light Grey attached tiles */
        .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"],
        .stTabs [data-baseweb="tab-list"] button[role="tab"],
        div[data-testid="stTab"] {
            height: auto !important;
            min-height: 48px !important;
            padding: 12px 28px !important;
            /* Top rounded corners only so they sit flat on the content box */
            border-top-left-radius: 10px !important;
            border-top-right-radius: 10px !important;
            border-bottom-left-radius: 0px !important;
            border-bottom-right-radius: 0px !important;
            background-color: #E2E8F0 !important;         /* Light Grey card fill */
            border: 2px solid #CBD5E1 !important;          /* Slate grey border */
            border-bottom: none !important;                /* Flush against content panel */
            color: #1E293B !important;                    /* Dark slate readable text */
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            cursor: pointer !important;
            margin-right: -2px !important;                 /* Overlap borders so they connect */
            transition: background-color 0.2s ease !important;
        }

        /* Enforce child text color */
        .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] *,
        .stTabs [data-baseweb="tab-list"] button[role="tab"] * {
            color: #1E293B !important;
            font-weight: 700 !important;
        }

        /* 3. Hover State */
        .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"]:hover,
        .stTabs [data-baseweb="tab-list"] button[role="tab"]:hover {
            background-color: #F8FAFC !important;
        }

        /* 4. Active Selected Tab: Soft Light Green with Bold Border */
        .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"],
        .stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
            background-color: #E8F5E9 !important;         /* Light pastel green */
            border: 2px solid #4CAF50 !important;          /* Green border matching panel */
            border-bottom: 2px solid #E8F5E9 !important;   /* Conceal bottom border to merge into panel */
            position: relative !important;
            z-index: 3 !important;                         /* Sits above the panel border line */
        }

        /* Active tab text color */
        .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] *,
        .stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] * {
            color: #1B5E20 !important;
            font-weight: 800 !important;
        }

        /* 5. Full Border Enclosure around the Tab Content Below */
        .stTabs [data-baseweb="tab-panel"] {
            background-color: #0E1117 !important;         /* Dark background interior */
            border: 2px solid #4CAF50 !important;          /* Full border outline around content */
            border-radius: 0px 12px 12px 12px !important;  /* Rounded corners except top-left under first tab */
            padding: 1.5rem !important;
            margin-top: 0px !important;
            position: relative !important;
            z-index: 1 !important;
            box-shadow: 0px 8px 24px rgba(0, 0, 0, 0.4) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

_inject_tab_styles()

st.markdown(
    """
    <style>
    /* Override all Streamlit buttons */
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
    </style>
    """,
    unsafe_allow_html=True,
)

def render_analyzer() -> None:
    st.caption(
        "Big 5 growth screening plus DCF, Peter Lynch Fair Value, Graham Number, "
        "Graham Formula, and PEG. Enter a ticker to compute all methods side-by-side."
    )

    with st.sidebar:
        st.header("Valuation knobs")
        dcf_discount = st.slider(
            "DCF discount rate",
            min_value=6.0,
            max_value=15.0,
            value=10.0,
            step=0.5,
            format="%.1f%%",
            help="Required annual return. 10% = long-run S&P 500 average; 15% = Buffet's aggressive rate.",
        ) / 100
        dcf_terminal = st.slider(
            "DCF terminal growth",
            min_value=0.0,
            max_value=4.0,
            value=2.5,
            step=0.5,
            format="%.1f%%",
            help="Perpetual growth rate after the fade period. Should be <= long-run GDP growth (~2.5-3%).",
        ) / 100
        aaa_yield = st.slider(
            "AAA corporate bond yield (for Graham Formula)",
            min_value=2.0,
            max_value=10.0,
            value=4.5,
            step=0.5,
            format="%.1f%%",
            help="Current AAA corporate bond yield. Used in Graham's revised 1974 formula.",
        ) / 100
        mos_pct = st.slider(
            "Margin of Safety (Lynch/Graham/PEG)",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=5.0,
            format="%.1f%%",
            help="Discount applied to fair value to get a buy price. MOS uses a fixed 50% on Value Price.",
        ) / 100

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
                submitted = st.form_submit_button("⏳ Analyzing...", use_container_width=True, disabled=True)
            else:
                submitted = st.form_submit_button("Analyze", use_container_width=True)

    if submitted:
        st.session_state["analyzing"] = True
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

    with st.spinner(f"Fetching 10-year financials for {symbol}..."):
        try:
            fin = _cached_fetch(symbol)
        except FetchError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Unexpected error fetching {symbol}: {e}")
            return

    st.subheader(f"{fin.company_name} ({fin.ticker})")
    # ADD THIS LINE:
    if fin.exchange in ("NSE", "BSE"):
        st.caption(f"🇮🇳 Listed on {fin.exchange} (₹ INR)")
    elif fin.exchange:
        st.caption(f"US Listed on {fin.exchange}  ($ USD)")
    _eps_ttm = float(fin.eps.iloc[-1]) if not fin.eps.empty else None
    top1, top2, top3, top4, top5, top6, top7 = st.columns(7)
    top1.metric("Current Price", _fmt_money(fin.current_price, fin.exchange))
    top2.metric("Market Cap", _fmt_money(fin.market_cap,fin.exchange))
    top3.metric("EPS (TTM)", _fmt_money(_eps_ttm, fin.exchange))
    top4.metric("TTM P/E", f"{fin.pe_ratio_ttm:.1f}" if fin.pe_ratio_ttm else "n/a")
    top5.metric("Div Yield", _fmt_pct(fin.dividend_yield))
    top6.metric("BVPS", _fmt_money(fin.book_value_per_share, fin.exchange))
    top7.metric("Years of data", str(fin.years_available))

    # Data-source note is shown in the footer only — not up top. Split-artifact
    # flag is precomputed here and rendered alongside the footer note.
    _show_split_warning = False
    if fin.data_source.startswith("edgar") and not fin.eps.empty and len(fin.eps) >= 3:
        ratios = fin.eps.diff() / fin.eps.shift(1).abs()
        _show_split_warning = bool((ratios.abs() > 0.5).any())

    big5 = compute_big5(fin)

    st.markdown(
        '### Big 5 Growth Rates — <span style="color: #00E676;">Is this a Wonderful Company?</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "As per Buffett, a wonderful business has a durable competitive advantage — a "
        '"moat". Ideally, all five metrics below should show historical growth rates of '
        "10% or more per year over the 10, 5, 3, and 1-year windows. All these numbers are Compounded Annual Growth Rates (CAGRs)."
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
                    <div style="color:##FFFFFF; font-size: 0.7rem;">{_ws.consistency_reason}</div>                    
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

    st.markdown("### Valuation")

    # --- EPS selection: prefer latest year, but fall back to a 3yr positive-EPS
    # average when the latest year is negative (value Price and other formulas
    # can't handle negative earnings). This surfaces to the user as an amber note.
    _eps_ttm_raw = float(fin.eps.iloc[-1]) if not fin.eps.empty else None
    _normalized_eps = None
    _normalized_years: list[int] = []
    if _eps_ttm_raw is not None and _eps_ttm_raw <= 0 and not fin.eps.empty:
        positive_history = fin.eps[fin.eps > 0]
        if not positive_history.empty:
            # Take the 3 most recent positive years (or fewer if less available).
            recent_pos = positive_history.tail(3)
            _normalized_eps = float(recent_pos.mean())
            _normalized_years = [int(y) for y in recent_pos.index]

    current_eps = _normalized_eps if _normalized_eps is not None else _eps_ttm_raw

    if _normalized_eps is not None:
        yrs_txt = ", ".join(str(y) for y in _normalized_years)
        st.warning(
            f":warning: Latest reported EPS is negative (${_eps_ttm_raw:.2f}). "
            f"value Price and other formulas need positive EPS to work — the app "
            f"has substituted a **normalized EPS of ${_normalized_eps:.2f}**, computed as the "
            f"average of the last {len(_normalized_years)} positive years ({yrs_txt}). "
            f"Treat these valuations as guidance about what this business *could* be worth "
            f"if it returns to prior profitability, not what it's worth today."
        )

    big5_eps_g = _big5_eps_growth(big5)

    def _moat_conservative_growth() -> float | None:
        candidates = [g for g in (big5_eps_g, fin.analyst_5yr_growth) if g is not None and g > 0]
        if not candidates:
            return None
        growth = min(candidates)
        return min(growth, 0.15)


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

    mode_options = _valuation_growth_mode_options()
    if "valuation_growth_mode" not in st.session_state or st.session_state["valuation_growth_mode"] not in mode_options:
        st.session_state["valuation_growth_mode"] = "Value conservative"
    selected_growth_mode = st.session_state["valuation_growth_mode"]

    if "custom_growth_rate" not in st.session_state:
        st.session_state["custom_growth_rate"] = 0


    def _compute_valuation_value(mode: str):
        st.write(
            "DEBUG COMPUTE:",
            mode,
            "analyst=",
            fin.analyst_5yr_growth,
            "custom=",
            st.session_state.get("custom_growth_rate"),
        )
        if mode == "Custom":
            custom_growth_pct = float(st.session_state.get("custom_growth_rate", 0) or 0)
            if custom_growth_pct <= 0:
                return None
            custom_growth = custom_growth_pct / 100.0
            return value_price(
                current_eps=current_eps,
                big5_eps_growth=None,
                analyst_growth=None,
                historical_pe=fin.pe_ratio_ttm,
                custom_growth=custom_growth,
            )

        if mode == "Value conservative":
            use_big5_growth = big5_eps_g is not None
            use_analyst_growth = fin.analyst_5yr_growth is not None
            if not (use_big5_growth or use_analyst_growth):
                return None
            return value_price(
                current_eps=current_eps,
                big5_eps_growth=big5_eps_g if use_big5_growth else None,
                analyst_growth=fin.analyst_5yr_growth if use_analyst_growth else None,
                historical_pe=fin.pe_ratio_ttm,
            )

        if mode == "Big 5 EPS growth":
            if big5_eps_g is None:
                return None
            return value_price(
                current_eps=current_eps,
                big5_eps_growth=big5_eps_g,
                analyst_growth=None,
                historical_pe=fin.pe_ratio_ttm,
            )

        if mode == "Analyst 5Y growth":
            if fin.analyst_5yr_growth is None:
                return None
            return value_price(
                current_eps=current_eps,
                big5_eps_growth=None,
                analyst_growth=fin.analyst_5yr_growth,
                historical_pe=fin.pe_ratio_ttm,
            )

        return None


    # The valuation result is intentionally recomputed from the current selected
    # growth source only after the user presses RE-VALUATE. A stale cached result
    # would otherwise keep showing the old value price even after the radio
    # selection changes.

    _VERDICT_TOOLTIPS = {
        "BARGAIN BUY": (
            "Current price is at or below 50% of the MOS Buy Price — a rare deep "
            "discount. In Value terms this is a 'back up the truck' setup: the "
            "formula says you get a huge margin of safety on top of an already "
            "conservative fair value. BUT — verify the company is still healthy. "
            "Bargain prices often reflect real risks the market is pricing in "
            "(pending lawsuit, industry decline, accounting concerns). Check "
            "recent news and the Big 5 trend before buying."
        ),
        "BUY": (
            "Current price is below the Margin of Safety (MOS) Buy Price. This "
            "means the formula says you can buy today with the safety cushion "
            "Buffett / Lynch / Graham built into their method. This is the "
            "signal each method was designed to produce — but it's still just "
            "one formula. Cross-check against the other valuation methods in "
            "the table below, verify the Big 5 are strong, and confirm the "
            "business quality before acting."
        ),
        "WATCH": (
            "Current price is between the MOS Buy Price and the Intrinsic Value. "
            "The formula thinks the stock is reasonably valued but not offering "
            "a margin of safety. Add to your watchlist and wait for a pullback "
            "toward MOS — Buffett's discipline is patience. If fundamentals "
            "improve (Big 5 accelerating) or the stock drops, this can quickly "
            "become a BUY."
        ),
        "AVOID": (
            "AVOID here doesn't mean the stock is a bad buy in absolute terms — "
            "it only means the current price is above the intrinsic value "
            "produced by this specific formula (Value / Peter Lynch / etc.). "
            "These formulas are deliberately conservative (Buffett caps growth "
            "at 15%, uses a 15% discount rate, and applies a 50% margin of "
            "safety). A stock flagged AVOID may still be a good long-term buy "
            "if you believe the company will outgrow the conservative "
            "assumptions, if you require a lower return than 15%, or if "
            "qualitative factors (moat, management, industry) justify paying "
            "above the formula price. Use these numbers as guardrails, not gospel."
        ),
        "Unknown": (
            "Not enough data to compute a verdict — usually missing current price "
            "or a computable fair value. Check the Data footnote at the bottom "
            "of the page."
        ),
    }

    # Section-level tooltip (used above the Other Intrinsic Value Methods table,
    # where a per-cell tooltip isn't possible). Covers all four verdict outcomes.
    _VERDICT_TOOLTIP_SECTION = (
        "Verdicts compare current price against each formula's fair value. "
        "BARGAIN BUY = current is <= 50% of MOS Buy Price (deep discount). "
        "BUY = current is below the MOS Buy Price. "
        "WATCH = between MOS and fair value. "
        "AVOID = above fair value — but each formula is deliberately conservative, "
        "so AVOID doesn't necessarily mean 'bad company' — it means 'the formula's "
        "guardrails say wait'. Cross-check with the Big 5 and the other methods."
    )


    _ANALYST_STYLE = {
        # yfinance recommendationKey -> (display, color)
        "strong_buy":  ("STRONG BUY",  "bargain"),
        "buy":         ("BUY",         "green"),
        "outperform":  ("BUY",         "green"),   # legacy label some tickers still use
        "hold":        ("HOLD",        "orange"),
        "underperform":("SELL",        "red"),
        "sell":        ("SELL",        "red"),
        "strong_sell": ("STRONG SELL", "red"),
    }


    def _render_analyst_cell(col, fin) -> None:
        """Analyst consensus rating from yfinance, with count and price target upside."""
        key = fin.analyst_rec_key
        count = fin.analyst_count
        label, color = _ANALYST_STYLE.get(key or "", (None, "gray"))

        if label is None or not count:
            col.markdown(
                """
                <div style="color: rgba(250,250,250,0.6); font-size: 0.875rem;">Analyst Rating</div>
                <div style="font-size: 1.5rem; color: #9AA0A6;">n/a</div>
                <div style="font-size: 0.72rem; color: #9aa0a6;">no analyst coverage</div>
                """,
                unsafe_allow_html=True,
            )
            return

        color_hex = _COLOR_HEX[color]
        # Detail: N analysts, plus price target upside vs current if available.
        detail_parts = [f"{count} analyst{'s' if count != 1 else ''}"]
        if fin.analyst_target_mean and fin.current_price and fin.current_price > 0:
            tgt = fin.analyst_target_mean
            upside = (tgt - fin.current_price) / fin.current_price * 100
            detail_parts.append(f"target ${tgt:.0f} ({upside:+.0f}%)")
        detail = " · ".join(detail_parts)

        col.markdown(
            f"""
            <div style="color: rgba(250,250,250,0.6); font-size: 0.875rem;">Analyst Rating</div>
            <div style="font-size: 2.0rem; font-weight: 400; line-height: 1.2; color: {color_hex};">
                {label}
            </div>
            <div style="font-size: 0.72rem; color: #9aa0a6; line-height: 1.3; margin-top: 0.2rem;">
                {detail}
            </div>
            """,
            unsafe_allow_html=True,
        )


    def _render_verdict_cell(col, verdict: str, detail: str, color: str) -> None:
        # &#9432; is the ⓘ (info) glyph. HTML `title` gives us a native browser tooltip
        # on hover — works on all platforms without needing any Streamlit component.
        tip_raw = _VERDICT_TOOLTIPS.get(verdict, _VERDICT_TOOLTIPS["Unknown"])
        tip = tip_raw.replace('"', "&quot;")
        col.markdown(
            f"""
            <div style="color: rgba(250,250,250,0.6); font-size: 0.875rem;">
                Verdict
                <span title="{tip}" style="cursor: help; color: #4CAF50; margin-left: 4px; font-size: 0.9rem;">&#9432;</span>
            </div>
            <div style="font-size: 2.0rem; font-weight: 400; line-height: 1.2; color: {_COLOR_HEX[color]};">
                {verdict}
            </div>
            <div style="font-size: 0.72rem; color: #9aa0a6; line-height: 1.3; margin-top: 0.2rem;">
                {detail}
            </div>
            """,
            unsafe_allow_html=True,
        )


    if "valuation_growth_result" not in st.session_state:
        st.session_state["valuation_growth_result"] = None

    if "custom_growth_rate" not in st.session_state:
        st.session_state["custom_growth_rate"] = 0

    # Custom growth rate is stored as a whole-number percentage for UI simplicity,
    # but the valuation model expects a decimal fraction (e.g. 15 => 0.15).
    if isinstance(st.session_state["custom_growth_rate"], float):
        st.session_state["custom_growth_rate"] = int(round(st.session_state["custom_growth_rate"] * 100))

    val = st.session_state.get("valuation_growth_result")
    if val is None:
        val = _compute_valuation_value(selected_growth_mode)
        st.session_state["valuation_growth_result"] = val

    if val is None:
        st.info(
            "Cannot compute value Price — need positive current EPS and at least one "
            "growth estimate (Big 5 EPS growth or analyst 5yr growth)."
        )
    else:
        v_verdict, v_detail, v_color = _verdict_price(fin.current_price, val.mos_price, val.value_price)
        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("Value Price / Intrinsic Value", _fmt_money(val.value_price, fin.exchange))
        v2.metric("Margin of Safety (MOS) Buy Price", _fmt_money(val.mos_price, fin.exchange))
        v3.metric("Current Price", _fmt_money(fin.current_price, fin.exchange))
        _render_verdict_cell(v4, v_verdict, v_detail, v_color)
        _render_analyst_cell(v5, fin)

        growth_group_col, button_col = st.columns([8.2, 1.8])
        with growth_group_col:
            st.markdown(
                "<div style='font-size: 0.875rem; margin-bottom: 0.25rem;'>Growth source</div>",
                unsafe_allow_html=True,
            )
            radio_col, custom_col = st.columns([6.5, 1.7])
            #with radio_col:
            #    st.markdown("<div style='height: 0.65rem;'></div>", unsafe_allow_html=True)
            # RADIO BUTTONS
            selected_growth_mode = st.radio(
                "Growth source",
                options=mode_options,
                horizontal=True,
                format_func=_valuation_growth_rate_label,
                key="valuation_growth_mode",
                label_visibility="collapsed",
            )
            custom_col, button_col, spacer_col = st.columns([1.7, 1.8, 6.5])
            with custom_col:
                st.markdown(
                    "<div style='font-size: 0.75rem; color: #9aa0a6; margin-bottom: 0.15rem;'>Custom %</div>",
                    unsafe_allow_html=True,
                )
                st.number_input(
                    "Custom %",
                    key="custom_growth_rate",
                    min_value=1,
                    max_value=100,
                    step=1,
                    format="%d",
                    disabled=(selected_growth_mode != "Custom"),
                    label_visibility="collapsed",
                    help="Select the 'Custom' radio option, then set this to force that whole-number growth rate.",
                )
        with button_col:
            st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
            revalue_button = st.button("RE-VALUATE", use_container_width=True)

        st.caption(f"DEBUG — mode: {selected_growth_mode} | custom box: {st.session_state.get('custom_growth_rate')}")

        if revalue_button:
            st.session_state["valuation_growth_result"] = _compute_valuation_value(selected_growth_mode)
            st.session_state["valuation_calculated_mode"] = selected_growth_mode
            st.rerun()

        with st.expander("How the value Price was calculated"):
            _big5_g_txt = _fmt_pct(big5_eps_g)
            _analyst_g_txt = _fmt_pct(fin.analyst_5yr_growth)
            _hist_pe_txt = f"{fin.pe_ratio_ttm:.1f}" if fin.pe_ratio_ttm else "n/a"
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
    - = **{_fmt_money(val.future_price, fin.exchange)}**

    **Step 5 — Discount back to today at {_fmt_pct(val.discount_rate)} (Buffett's required return)**
    - Future price ÷ (1 + {val.discount_rate:.2f})^{val.horizon_years}
    - = **{_fmt_money(val.value_price, fin.exchange)}**  ← *Value Price*

    **Step 6 — Apply {_mos_pct_int}% Margin of Safety**
    - value × {1 - _mos_pct_int/100:.2f}
    - = **{_fmt_money(val.mos_price, fin.exchange)}**  ← *MOS Buy Price*
    """.strip()
            )

    _tip_other = _VERDICT_TOOLTIP_SECTION.replace('"', "&quot;")
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
            current_eps=current_eps,
            growth_rate=_growth_for_intrinsic,
            dividend_yield=fin.dividend_yield,
            current_price=fin.current_price,
            mos=mos_pct,
        )
    )
    _add(
        graham_number(
            current_eps=current_eps,
            book_value_per_share=fin.book_value_per_share,
            current_price=fin.current_price,
            mos=mos_pct,
        )
    )
    _add(
        graham_formula(
            current_eps=current_eps,
            growth_rate=_growth_for_intrinsic,
            current_price=fin.current_price,
            aaa_bond_yield=aaa_yield,
            mos=mos_pct,
        )
    )
    _add(
        peg_fair_value(
            current_eps=current_eps,
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
                "Fair Value": _fmt_money(m.fair_value, fin.exchange),
                "MOS Buy": _fmt_money(m.mos_price, fin.exchange),
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
            use_container_width=True,
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
        st.dataframe(combined.sort_index(ascending=False), use_container_width=True)

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


st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem;
        background-color: transparent;
        border-bottom: none;
        padding-bottom: 0.25rem;
    }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"],
    .stTabs [data-baseweb="tab-list"] button[role="tab"] {
        height: auto;
        min-height: 2.75rem;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1.5px solid rgba(255, 255, 255, 0.14);
        border-radius: 10px;
        padding: 0.65rem 1.5rem;
        color: #9aa0a6;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"]:hover,
    .stTabs [data-baseweb="tab-list"] button[role="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.09);
        border-color: rgba(76, 175, 80, 0.5);
        color: #e8eaed;
    }
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
        background-color: rgba(76, 175, 80, 0.2);
        border-color: #4CAF50;
        color: #b6f0b6;
        box-shadow: 0 0 0 1px rgba(76, 175, 80, 0.3);
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

tab_analyzer, tab_screener = st.tabs(["🔍 Stock Analyzer", "📊 Stock Screener"])

with tab_screener:
    render_stock_screener()

with tab_analyzer:
    render_analyzer()
