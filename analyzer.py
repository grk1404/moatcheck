from __future__ import annotations
import streamlit as st
import shutil
import os
import statistics
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import logging
from data_provider import get_ticker
from investments import _metric_html, get_holding_tickers
from moatcheck.valuation import get_model_growth, calculate_model_valuation, get_ui_growth_rate
from moatcheck import (
    Big5Result, Financials, compute_big5, dcf_two_stage, debt_to_fcf, fetch, 
    graham_formula, graham_number, payback_time, peg_fair_value, peter_lynch_fair, value_price
)
from moatcheck.big5 import WINDOWS, PASS_THRESHOLD
from moatcheck.fetcher import FetchError
from moatcheck.intrinsic import MethodResult

warnings.filterwarnings('ignore')
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

_COLOR_HEX = {
    "bargain": "#00E676", "green": "#4CAF50", "orange": "#FFA726", "red": "#EF5350", "gray": "#9AA0A6",
}

_COMMON_TICKER_FIXES = {"APPL": "AAPL", "GOOL": "GOOG", "MSFTT": "MSFT", "AMZNN": "AMZN"}

def clear_yfinance_cache(ticker: str | None = None) -> bool:
    potential_dirs = [
        Path(os.path.expanduser("~")) / "AppData" / "Local" / "py-yfinance-cache",
        Path(os.path.expanduser("~")) / ".cache" / "py-yfinance-cache",
        Path("py-yfinance-cache"),
    ]
    cleared = False
    for base_dir in potential_dirs:
        if not base_dir.exists(): continue
        if ticker:
            ticker_dir = base_dir / ticker.upper()
            if ticker_dir.exists():
                shutil.rmtree(ticker_dir)
                cleared = True
        else:
            shutil.rmtree(base_dir)
            cleared = True
    return cleared

def _first_n_sentences(text: str, n: int = 2) -> str:
    if not text: return ""
    import re
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(parts[:n])

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fetch(symbol: str) -> Financials:
    return fetch(symbol)

def _suggest_ticker(symbol: str) -> str | None:
    return _COMMON_TICKER_FIXES.get(symbol.strip().upper())

def _fmt_pct(v: float | None) -> str:
    if v is None or pd.isna(v): return "no data"
    return f"{v * 100:.1f}%"

def get_currency_symbol(exchange: str, ticker: str = "") -> str:
    if exchange and exchange.upper() in ("NSE", "BSE"): return "₹"
    if ticker and (ticker.endswith(".NS") or ticker.endswith(".BO")): return "₹"
    return "$"

def _fmt_money(v: float | None, exchange: str = "", ticker: str = "") -> str:
    if v is None: return "n/a"
    currency = get_currency_symbol(exchange, ticker)
    if abs(v) >= 1e12: return f"{currency}{v / 1e12:.2f}T"
    if abs(v) >= 1e9: return f"{currency}{v / 1e9:.2f}B"
    if abs(v) >= 1e6: return f"{currency}{v / 1e6:.2f}M"
    return f"{currency}{v:,.2f}"

def _as_fraction(value, *, is_percent: bool) -> float | None:
    if value is None: return None
    try: v = float(value)
    except (TypeError, ValueError): return None
    return v / 100.0 if is_percent else v

def _color_pass(val: float | None) -> str:
    if val is None or pd.isna(val): return "color: #888;"
    if val >= PASS_THRESHOLD: return "background-color: #1e4620; color: #b6f0b6;"
    return "background-color: #4b1e1e; color: #f0b6b6;"

def _render_big5_table(big5: Big5Result) -> None:
    df = big5.as_dataframe()
    avg_row = {"metric": "Average (All Metrics)"}
    for w in WINDOWS:
        col = f"{w}yr"
        valid_vals = [v for v in df[col] if pd.notna(v)]
        avg_row[col] = sum(valid_vals) / len(valid_vals) if valid_vals else None
    avg_pass_count = sum(1 for w in WINDOWS if avg_row[f"{w}yr"] is not None and avg_row[f"{w}yr"] >= PASS_THRESHOLD)
    avg_row["pass"] = avg_pass_count >= len(WINDOWS) // 2
    df_with_avg = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    def _calc_row_avg(row):
        vals = [row[f"{w}yr"] for w in WINDOWS if pd.notna(row[f"{w}yr"])]
        return sum(vals) / len(vals) if vals else None
    df_with_avg["Average"] = df_with_avg.apply(_calc_row_avg, axis=1)
    display = df_with_avg.copy()
    for w in WINDOWS:
        display[f"{w}yr"] = display[f"{w}yr"].map(_fmt_pct)
    display["Average"] = display["Average"].map(_fmt_pct)
    display["pass"] = display["pass"].map(lambda p: "PASS" if p else "FAIL")
    cols = ["metric"] + [f"{w}yr" for w in WINDOWS] + ["Average", "pass"]
    display = display[cols].rename(columns={"metric": "Metric", "pass": "Verdict"})
    
    def _style(row):
        original = df_with_avg.loc[row.name]
        is_avg_row = row["Metric"] == "Average (All Metrics)"
        styles = pd.Series("", index=row.index)
        if is_avg_row: styles["Metric"] = "font-weight: bold; background-color: rgba(255,255,255,0.05);"
        for w in WINDOWS:
            col_name = f"{w}yr"
            if col_name in styles.index:
                cell_style = _color_pass(original[col_name])
                if is_avg_row: cell_style += " font-weight: bold;"
                styles[col_name] = cell_style
        if "Average" in styles.index:
            styles["Average"] = _color_pass(original["Average"]) + " font-weight: bold;"
        verdict_style = "background-color: #1e4620; color: #b6f0b6;" if original["pass"] else "background-color: #4b1e1e; color: #f0b6b6;"
        if is_avg_row: verdict_style += " font-weight: bold;"
        styles["Verdict"] = verdict_style
        return styles
    st.dataframe(display.style.apply(_style, axis=1), width="stretch", hide_index=True)

def _verdict_price(current: float | None, mos: float | None, value: float | None) -> tuple[str, str, str]:
    if current is None or mos is None or value is None or value <= 0 or mos <= 0: return ("Unknown", "", "gray")
    if current <= mos * 0.5: return ("BARGAIN BUY", f"{(mos - current) / mos * 100:.0f}% below MOS Buy Price — rare deep discount", "bargain")
    if current <= mos: return ("BUY", f"{(mos - current) / mos * 100:.0f}% below MOS Buy Price", "green")
    if current <= value: return ("WATCH", f"between MOS and Intrinsic Value ({(value - current) / value * 100:.0f}% below Intrinsic)", "orange")
    return ("AVOID", f"above Intrinsic Value by {(current - value) / value * 100:.0f}%", "red")

def render_etf_dashboard(symbol: str, yft) -> None:
    try: info = yft.info or {}
    except: info = {}
    name = info.get("longName") or info.get("shortName") or symbol
    st.subheader(f"{name} ({symbol.upper()})")
    st.caption(f"🏦 Exchange-Traded Fund · {info.get('category', 'ETF')} · Listed on {info.get('exchange', 'n/a')}")
    st.info("ETFs don't report financial statements, so Big 5 growth, DCF, and the value-investing formulas don't apply. Below are the fund-level metrics that matter for an ETF. For entry/exit signals, use **📈 Technical Analysis**.")
    
    _last_price = info.get("regularMarketPrice") or info.get("previousClose") or info.get("navPrice")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Price", f"${_last_price:,.2f}" if _last_price else "n/a")
    m2.metric("Fund Size (AUM)", _fmt_money(info.get("totalAssets"), "USD") if info.get("totalAssets") else "n/a")
    _expense = _as_fraction(info.get("netExpenseRatio") or info.get("annualReportExpenseRatio"), is_percent=True)
    m3.metric("Expense Ratio", f"{_expense * 100:.2f}%" if _expense is not None else "n/a")
    _yield = _as_fraction(info.get("yield"), is_percent=False)
    m4.metric("Distribution Yield", f"{_yield * 100:.2f}%" if _yield is not None else "n/a")
    m5.metric("Beta (3yr)", f"{info.get('beta3Year') or info.get('beta'):.2f}" if info.get("beta3Year") or info.get("beta") else "n/a")
    _ytd = _as_fraction(info.get("ytdReturn"), is_percent=True)
    m6.metric("YTD Return", f"{_ytd * 100:+.1f}%" if _ytd is not None else "n/a")

def render_analyzer() -> None:
    st.markdown('<h2 style="margin-bottom: 0.25rem;">🔍 Stock Analyzer <span style="font-size: 0.95rem; font-weight: 500; color: #4CAF50; letter-spacing: 0.03em;">— Separating Wonderful Businesses from the Merely Good</span></h2>', unsafe_allow_html=True)
    st.caption("Enter a US-listed ticker (e.g. AAPL, MSFT, KO) or an Indian ticker (e.g. RELIANCE, TCS). Results cover 5 key growth rates, five valuation methods, plus Buffett-style health checks.")
    
    _holdings = get_holding_tickers()
    _NONE = "(none)"
    st.session_state.setdefault("analyzer_ticker_box", "AAPL")

    def _apply_pick() -> None:
        chosen = st.session_state.get("analyzer_pick", _NONE)
        if chosen and chosen != _NONE: st.session_state["analyzer_ticker_box"] = chosen
        st.session_state["analyzer_pick"] = _NONE

    with st.form("analyze"):
        col_a, col_h, col_b, col_c = st.columns([2, 3, 1, 3])
        with col_a: ticker = st.text_input("Ticker symbol", max_chars=10, key="analyzer_ticker_box").strip().upper()
        with col_h: st.selectbox("Or pick from My Investments", key="analyzer_pick", options=[_NONE] + [sym for sym, _ in _holdings], format_func=lambda s_: s_ if s_ == _NONE else f"{s_}  -  {dict(_holdings).get(s_, '')[:28]}", disabled=not _holdings)
        with col_b:
            st.markdown("<div style='height: 28px;color: #FFFFFF'></div>", unsafe_allow_html=True)
            if "analyzing" in st.session_state and st.session_state["analyzing"]:
                submitted = st.form_submit_button("⏳ Analyzing...", width="stretch", disabled=True)
            else:
                submitted = st.form_submit_button("Analyze", width="stretch", on_click=_apply_pick)

    if submitted:
        st.session_state["analyzing"] = True
        if st.session_state.get("last_ticker") != ticker:
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

    if submitted: st.session_state["last_ticker"] = ticker
    symbol = st.session_state.get("last_ticker", ticker)

    if "valuation_ticker" not in st.session_state or st.session_state["valuation_ticker"] != symbol:
        st.session_state["valuation_ticker"] = symbol
        st.session_state.pop("valuation_growth_mode", None)
        st.session_state["valuation_growth_result"] = None
        st.session_state.pop("valuation_calculated_mode", None)

    with st.spinner(f"Looking up {symbol}..."):
        try: _qtype = (get_ticker(symbol).info or {}).get("quoteType", "").upper()
        except: _qtype = ""

    if _qtype in ("ETF", "MUTUALFUND", "INDEX", "CRYPTOCURRENCY", "CURRENCY"):
        st.session_state["analyzing"] = False
        render_etf_dashboard(symbol, get_ticker(symbol))
        return

    with st.spinner(f"Fetching 10-year financials for {symbol}..."):
        try:
            if st.session_state.get("valuation_ticker") != symbol:
                st.cache_data.clear()
                st.session_state["valuation_ticker"] = symbol
            fin = _cached_fetch(symbol)
            st.session_state["analyzing"] = False
        except Exception as e:
            st.session_state["analyzing"] = False
            st.error(f"Unexpected error fetching {symbol}: {e}")
            if st.button(f"🧹 Clear cache for {symbol} and try again", key=f"fix_{symbol}"):
                if clear_yfinance_cache(symbol):
                    st.cache_data.clear()
                    st.rerun()
            return

    st.subheader(f"{fin.company_name} ({fin.ticker})")
    
    if fin.company_summary or fin.company_sector or fin.company_industry or fin.company_country:
        with st.expander("Company Summary:", expanded=True):
            _label_parts = [fin.company_industry or fin.company_sector]
            if fin.company_country:
                _label_parts.append({"United States": "USA", "USA": "USA", "US": "USA"}.get(fin.company_country, fin.company_country))
            _label = " · ".join(p for p in _label_parts if p)
            if _label: st.markdown(f'<div style="color:#9aa0a6; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">{_label}</div>', unsafe_allow_html=True)
            if fin.company_summary: st.markdown(_first_n_sentences(fin.company_summary, 2))
            else: st.caption("Detailed summary unavailable for this ticker (common for foreign filers and small caps).")

    if fin.exchange in ("NSE", "BSE"): st.caption(f"🇮🇳 Listed on {fin.exchange} (₹ INR)")
    elif fin.exchange: st.caption(f"US Listed on {fin.exchange}  ($ USD)")
    _eps_ttm = float(fin.eps.iloc[-1]) if not fin.eps.empty else None

    big5 = compute_big5(fin)
    big5_eps_g = get_model_growth(fin, big5.eps.values)

    _fg = getattr(fin, "analyst_forward_growth", None)
    if _fg is not None:
        _analyst_growth_raw_display = _fg
        _analyst_growth_label_src = getattr(fin, "analyst_forward_growth_source", "") or "+1y"
    elif getattr(fin, "analyst_5yr_growth", None) is not None:
        _analyst_growth_raw_display = fin.analyst_5yr_growth
        _analyst_growth_label_src = "trailing"
    else:
        _analyst_growth_raw_display = None
        _analyst_growth_label_src = ""

    fin._analyst_growth_for_model = get_model_growth(fin, big5.eps.values)
    fin._analyst_growth_raw_display = _analyst_growth_raw_display
    fin._analyst_growth_label_src = _analyst_growth_label_src    

    _eps_ttm_raw = float(fin.eps.iloc[-1]) if not fin.eps.empty else None
    _normalized_eps = None
    _normalized_years: list[int] = []
    if _eps_ttm_raw is not None and _eps_ttm_raw <= 0 and not fin.eps.empty:
        positive_history = fin.eps[fin.eps > 0]
        if not positive_history.empty:
            _normalized_eps = float(positive_history.mean())
            _normalized_years = [int(y) for y in positive_history.index]

    current_eps = _normalized_eps if _normalized_eps is not None else _eps_ttm_raw
    eps_for_valuation = _normalized_eps if _normalized_eps is not None else _eps_ttm_raw

    if _normalized_eps is not None:
        yrs_txt = ", ".join(str(y) for y in _normalized_years)
        st.warning(f":warning: Latest reported EPS is negative (${_eps_ttm_raw:.2f}). To calculate intrinsic value, the app has substituted a **historical average EPS of ${_normalized_eps:.2f}**, computed from the last {len(_normalized_years)} profitable years ({yrs_txt}). Treat these valuations as guidance about what this business *could* be worth if it returns to prior profitability, not what it's worth today.")

    def _moat_conservative_growth() -> float | None: return 0.15
    _MODEL_MODE = "Model (lower of Big 5 & analyst)"
    def _model_growth() -> float | None: return get_model_growth(fin, big5.eps.values)

    def _valuation_growth_rate_label(mode: str) -> str:
        if mode == _MODEL_MODE: return f"Model: lower of Big 5 & analyst: {_fmt_pct(_model_growth()) if _model_growth() is not None else 'n/a'}"
        if mode == "Value conservative": return f"Value conservative: {_fmt_pct(_moat_conservative_growth())}"
        if mode == "Big 5 EPS growth": return f"Big 5 EPS growth: {_fmt_pct(big5_eps_g) if big5_eps_g is not None else 'n/a'}"
        if mode == "Analyst 5Y growth": return f"Analyst 5Y growth: {_fmt_pct(fin.analyst_5yr_growth) if fin.analyst_5yr_growth is not None else 'n/a'}"
        if mode == "Custom": return "Custom"
        return mode

    def _valuation_growth_mode_options() -> list[str]:
        options = [_MODEL_MODE] if _model_growth() is not None else []
        options.append("Value conservative")
        if big5_eps_g is not None: options.append("Big 5 EPS growth")
        if fin.analyst_5yr_growth is not None: options.append("Analyst 5Y growth")
        options.append("Custom")
        return options

    if "valuation_growth_result" not in st.session_state: st.session_state["valuation_growth_result"] = None
    if "custom_growth_rate" not in st.session_state: st.session_state["custom_growth_rate"] = 0
    if isinstance(st.session_state["custom_growth_rate"], float): st.session_state["custom_growth_rate"] = int(round(st.session_state["custom_growth_rate"] * 100))

    mode_options = _valuation_growth_mode_options()
    if "valuation_growth_mode" not in st.session_state or st.session_state["valuation_growth_mode"] not in mode_options:
        st.session_state["valuation_growth_mode"] = mode_options[0]
    selected_growth_mode = st.session_state.get("valuation_growth_mode", _MODEL_MODE)
    
    val = st.session_state.get("valuation_growth_result")
    if val is None:
        calc_growth = get_ui_growth_rate(mode=selected_growth_mode, fin=fin, big5_eps_values=big5.eps.values, custom_pct=float(st.session_state.get("custom_growth_rate", 0) or 0))
        val_res = calculate_model_valuation(current_eps=eps_for_valuation, fcf_ttm=float(fin.free_cash_flow.iloc[-1]) if not fin.free_cash_flow.empty else None, shares_outstanding=fin.shares_outstanding, book_value_per_share=fin.book_value_per_share, dividend_yield=fin.dividend_yield, current_price=fin.current_price, growth_rate=calc_growth, historical_pe=fin.pe_ratio_ttm)
        if val_res["details"] and "Value Price" in val_res["details"]:
            val = val_res["details"]["Value Price"]
            st.session_state["valuation_growth_result"] = val
            st.session_state["valuation_calculated_mode"] = selected_growth_mode
    
    _VERDICT_TOOLTIPS = {
        "BARGAIN BUY": "Current price is at or below 50% of the MOS Buy Price...",
        "BUY": "Current price is below the Margin of Safety (MOS) Buy Price...",
        "WATCH": "Current price is between the MOS Buy Price and the Intrinsic Value...",
        "AVOID": "AVOID here doesn't mean the stock is a bad buy...",
        "Unknown": "Not enough data to compute a verdict.",
    }
    _ANALYST_STYLE = {"strong_buy": ("STRONG BUY", "bargain"), "buy": ("BUY", "green"), "outperform": ("BUY", "green"), "hold": ("HOLD", "orange"), "underperform": ("SELL", "red"), "sell": ("SELL", "red"), "strong_sell": ("STRONG SELL", "red")}

    def _render_analyst_cell(col, fin) -> None:
        key = fin.analyst_rec_key
        count = fin.analyst_count
        label, color = _ANALYST_STYLE.get(key or "", (None, "gray"))
        if label is None or not count:
            _ag = getattr(fin, "_analyst_growth_for_model", None)
            _src = getattr(fin, "_analyst_growth_label_src", "")
            _growth_html = ""
            if _ag is not None:
                if _src == "trailing": _growth_html = f'<div style="font-size: 0.72rem; color: #9aa0a6;">Trailing EPS growth: {_ag * 100:.1f}%</div>'
                else:
                    _src_label = {"+1y": "FY+1", "0y": "FY0"}.get(_src, _src)
                    _growth_html = f'<div style="font-size: 0.72rem; color: #9aa0a6;">Est. growth ({_src_label}): {_ag * 100:.1f}%</div>'
            col.markdown('<div style="color: rgba(250,250,250,0.6); font-size: 0.875rem;">Analyst Rating</div><div style="font-size: 1.5rem; color: #9AA0A6;">n/a</div><div style="font-size: 0.72rem; color: #9aa0a6;">no analyst coverage</div>' + _growth_html, unsafe_allow_html=True)
            return
        color_hex = _COLOR_HEX[color]
        detail_parts = [f"{count} analyst{'s' if count != 1 else ''}"]
        if fin.analyst_target_mean and fin.current_price and fin.current_price > 0: detail_parts.append(f"target ${fin.analyst_target_mean:.0f} ({(fin.analyst_target_mean - fin.current_price) / fin.current_price * 100:+.0f}%)")
        detail = " · ".join(detail_parts)
        _ag = getattr(fin, "_analyst_growth_for_model", None)
        _raw = getattr(fin, "_analyst_growth_raw_display", None)
        _src = getattr(fin, "_analyst_growth_label_src", "")
        if _ag is not None:
            if _src == "trailing": _label = "Trailing EPS growth"
            else:
                _src_label = {"+1y": "FY+1", "0y": "FY0"}.get(_src, _src)
                _label = f"Est. growth ({_src_label})"
            if _raw is not None and abs(_raw - _ag) > 1e-9: detail += f"<br>{_label}: {_raw * 100:.1f}% → capped {_ag * 100:.1f}%"
            else: detail += f"<br>{_label}: {_ag * 100:.1f}%"
        col.markdown(f'<div style="color: rgba(250,250,250,0.6); font-size: 0.875rem;">Analyst Rating</div><div style="font-size: 2.0rem; font-weight: 400; line-height: 1.2; color: {color_hex};">{label}</div><div style="font-size: 0.72rem; color: #9aa0a6; line-height: 1.3; margin-top: 0.2rem;">{detail}</div>', unsafe_allow_html=True)

    def _render_verdict_cell(col, verdict: str, detail: str, color: str) -> None:
        tip_raw = _VERDICT_TOOLTIPS.get(verdict, _VERDICT_TOOLTIPS["Unknown"])
        tip = tip_raw.replace('"', "&quot;")
        col.markdown(f'<div style="color: rgba(250,250,250,0.6); font-size: 0.875rem;">Verdict <span title="{tip}" style="cursor: help; color: #4CAF50; margin-left: 4px; font-size: 0.9rem;">&#9432;</span></div><div style="font-size: 2.0rem; font-weight: 400; line-height: 1.2; color: {_COLOR_HEX[color]};">{verdict}</div><div style="font-size: 0.72rem; color: #9aa0a6; line-height: 1.3; margin-top: 0.2rem;">{detail}</div>', unsafe_allow_html=True)

    if val is None:
        _has_positive_eps = (not fin.eps.empty) and bool((fin.eps > 0).any())
        if not _has_positive_eps: st.warning("⚠️ No Value Price: this company has no profitable year on record, so an earnings-based valuation isn't meaningful. See the P/S Valuation further down.")
        else: st.warning("⚠️ Cannot compute a Value Price with the selected growth source — the growth data it needs is missing. Pick another source above, or use **Custom** to enter a growth rate.")

    v_verdict, v_detail, v_color = _verdict_price(fin.current_price, val.mos_price if val else None, val.value_price if val else None)

    _intrinsic_sub = ""
    _intrinsic_color = "#9aa0a6"
    if val and fin.current_price and val.value_price:
        _upside = (val.value_price - fin.current_price) / fin.current_price
        if _upside < 0:
            _intrinsic_sub = f"{abs(_upside)*100:.1f}% above Value Price"
            _intrinsic_color = "#FFA726"
        else:
            _intrinsic_sub = f"{_upside*100:.1f}% below Value Price"
            _intrinsic_color = "#00E676"

    top1, top2, top3, top4, top5, top6, top7, top8 = st.columns([1, 1.4, 1, 1, 1, 1, 1, 1.2])
    top1.markdown(_metric_html("Current Price", _fmt_money(fin.current_price, fin.exchange, fin.ticker), sub=_intrinsic_sub, sub_color=_intrinsic_color), unsafe_allow_html=True)
    top2.metric("Market Cap", _fmt_money(fin.market_cap, fin.exchange, fin.ticker))
    
    _eps_sub = ""
    if getattr(fin, "headline_eps_ttm", None) is not None and abs(fin.headline_eps_ttm - _eps_ttm) / abs(_eps_ttm) > 0.02:
        _eps_sub = f"Non-GAAP: {_fmt_money(fin.headline_eps_ttm, fin.exchange or 'USD')}"

    top3.markdown(_metric_html("EPS (Diluted)", _fmt_money(_eps_ttm, fin.exchange or "USD"), sub=_eps_sub, sub_color="#9aa0a6", help_text="Diluted EPS is pulled directly from the SEC 10-K/10-Q income statement (GAAP). Non-GAAP is Yahoo's headline adjusted EPS."), unsafe_allow_html=True)
    top4.metric("TTM P/E", f"{fin.pe_ratio_ttm:.1f}" if fin.pe_ratio_ttm else "n/a")

    _ltd_ttm = float(fin.long_term_debt.iloc[-1]) if not fin.long_term_debt.empty else None
    _fcf_ttm = float(fin.free_cash_flow.iloc[-1]) if not fin.free_cash_flow.empty else None

    if _ltd_ttm is None: _debt_value, _debt_color, _debt_sub = "n/a", "#9AA0A6", ""
    elif _ltd_ttm <= 0: _debt_value, _debt_color, _debt_sub = "$0", "#9AA0A6", "debt-free"
    elif _fcf_ttm is None: _debt_value, _debt_color, _debt_sub = _fmt_money(_ltd_ttm, fin.exchange, fin.ticker), "#9AA0A6", "no FCF data"
    elif _fcf_ttm <= 0: _debt_value, _debt_color, _debt_sub = _fmt_money(_ltd_ttm, fin.exchange, fin.ticker), "#EF5350", "⚠ SEVERE · negative FCF"
    else:
        _years_to_pay = _ltd_ttm / _fcf_ttm
        _debt_value = _fmt_money(_ltd_ttm, fin.exchange, fin.ticker)
        if _years_to_pay > 5: _debt_color, _debt_sub = "#EF5350", f"{_years_to_pay:.1f}× FCF · Heavy"
        elif _years_to_pay > 3: _debt_color, _debt_sub = "#FFC107", f"{_years_to_pay:.1f}× FCF · Elevated"
        else: _debt_color, _debt_sub = "#00E676", f"{_years_to_pay:.1f}× FCF · Healthy"

    top5.markdown(_metric_html("LT Debt", _debt_value, sub=_debt_sub, sub_color=_debt_color, help_text="Long-term debt ÷ TTM free cash flow, shown as 'years to pay off'. Green ≤ 3×, yellow 3–5×, red > 5×."), unsafe_allow_html=True)
    top6.metric("BVPS", _fmt_money(fin.book_value_per_share, fin.exchange, fin.ticker))
    
    _hist_high, _hist_low = None, None
    if fin.years_available > 0:
        try:
            _hist = get_ticker(symbol).history(period=f"{fin.years_available}y")
            if not _hist.empty:
                _hist_high = float(_hist["High"].max())
                _hist_low = float(_hist["Low"].min())
        except: pass

    _yr_sub = f"High: {_fmt_money(_hist_high, fin.exchange, fin.ticker)} | Low: {_fmt_money(_hist_low, fin.exchange, fin.ticker)}" if _hist_high and _hist_low else ""
    top7.markdown(_metric_html("Years of data", str(fin.years_available), sub=_yr_sub, sub_color="#9aa0a6", help_text=f"Stock price range over the {fin.years_available} years of available SEC data."), unsafe_allow_html=True)

    if fin.week52_high and fin.week52_low and fin.week52_high > fin.week52_low:
        _w52_pct = (fin.current_price - fin.week52_low) / (fin.week52_high - fin.week52_low) * 100 if fin.current_price else None
        _w52_sub = f"Current ${fin.current_price:,.2f} ({_w52_pct:.0f}%)" if fin.current_price and _w52_pct is not None else ""
        top8.markdown(_metric_html("52-Week Range", f"${fin.week52_low:,.0f} – ${fin.week52_high:,.0f}", sub=_w52_sub, sub_color="#9aa0a6", help_text=f"52-week low ${fin.week52_low:,.2f}, high ${fin.week52_high:,.2f}."), unsafe_allow_html=True)
    elif fin.week52_high and fin.week52_low: top8.markdown(_metric_html("52-Week Range", f"${fin.week52_low:,.0f} – ${fin.week52_high:,.0f}"), unsafe_allow_html=True)
    else: top8.metric("52-Week Range", "n/a")

    _show_split_warning = False
    if fin.data_source.startswith("edgar") and not fin.eps.empty and len(fin.eps) >= 3:
        ratios = fin.eps.diff() / fin.eps.shift(1).abs()
        _show_split_warning = bool((ratios.abs() > 0.5).any())

    st.markdown('### Big 5 Growth Rates — <span style="color: #00E676;">Is this a Wonderful Company?</span>', unsafe_allow_html=True)
    st.caption(f'As per Buffett, a wonderful business has a durable competitive advantage — a "moat". Ideally, all five metrics below should show historical growth rates of 10% or more per year over the 10, 5, 3, and 1-year windows. All these numbers are Compounded Annual Growth Rates (CAGRs).')
    st.caption("Note: ROIC is shown as a period-average return, not a CAGR.")
    _render_big5_table(big5)

    st.markdown("##### Quality Metrics")
    
    def _build_quality_card(label, metric, threshold, is_pct=True):
        if metric is None or metric.values.get(10) is None: return f'<div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 0.6rem 1rem;"><div style="font-size: 0.75rem; color: #9aa0a6; text-transform: uppercase; letter-spacing: 0.05em;">{label}</div><div style="font-size: 1.1rem; font-weight: 600; color: #9aa0a6; margin: 0.2rem 0;">n/a</div></div>'
        val = metric.values.get(10)
        passed = val >= threshold
        val_str = f"{val * 100:.1f}%" if is_pct else f"{val:.2f}x"
        tgt_str = f"Target: >{threshold * 100:.0f}%" if is_pct else f"Target: >{threshold:.2f}x"
        color = "#00E676" if passed else "#EF5350"
        status = "PASS" if passed else "FAIL"
        return f'<div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-left: 3px solid {color}; border-radius: 8px; padding: 0.6rem 1rem;"><div style="font-size: 0.75rem; color: #9aa0a6; text-transform: uppercase; letter-spacing: 0.05em;">{label}</div><div style="display: flex; align-items: baseline; justify-content: space-between; margin: 0.2rem 0;"><div style="font-size: 1.2rem; font-weight: 600; color: {color};">{val_str}</div><div style="font-size: 0.75rem; font-weight: 600; color: {color};">{status}</div></div><div style="font-size: 0.65rem; color: #9aa0a6;">{tgt_str}</div></div>'

    card_cc = _build_quality_card("Cash Conversion", big5.cash_conversion, 0.80, is_pct=True)
    card_fcf = _build_quality_card("Discount to Fair Value (FCF)", big5.discount_fcf, 0.30, is_pct=True)
    card_np = _build_quality_card("Discount to Fair Value (NP)", big5.discount_np, 0.30, is_pct=True)
    st.markdown(f'<div style="display: flex; gap: 1rem; margin-bottom: 1.5rem;">{card_cc}{card_fcf}{card_np}</div>', unsafe_allow_html=True)

    _ws = big5.wonderfulness(time_horizon="10y")
    _ws_color_hex = _COLOR_HEX[_ws.color]
    _bar_pct = int(round(_ws.overall * 10))
    _strength_html = " · ".join(f"<span style='color:#b6f0b6'>{s}</span>" for s in _ws.strengths) if _ws.strengths else "<span style='color:#9aa0a6'>—</span>"
    _weakness_html = " · ".join(f"<span style='color:#f0b6b6'>{s}</span>" for s in _ws.weaknesses) if _ws.weaknesses else "<span style='color:#9aa0a6'>—</span>"

    st.markdown(f"""
        <div style="border-radius: 10px; padding: 1.25rem 1.5rem; margin: 0.75rem 0 1rem; background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.06)); border: 1px solid rgba(255,255,255,0.08);">
            <div style="font-size: 0.85rem; color: #9aa0a6; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;">How wonderful is the company:</div>
            <div style="display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap;">
                <div style="font-size: 3.5rem; font-weight: 700; line-height: 1; color: {_ws_color_hex};">{_ws.overall:.1f}<span style="font-size: 1.5rem; color: #9aa0a6; font-weight: 400;">/10</span></div>
                <div style="flex: 1;"><div style="font-size: 1.5rem; font-weight: 600; color: {_ws_color_hex};">{_ws.label}</div><div style="font-size: 0.85rem; color: #9aa0a6; margin-top: 0.15rem;">{_ws.checks_passed}/{_ws.checks_total} checks pass the 10% bar</div></div>
            </div>
            <div style="height: 8px; background: rgba(255,255,255,0.06); border-radius: 4px; margin: 0.75rem 0 0.9rem; overflow: hidden;"><div style="height: 100%; width: {_bar_pct}%; background: {_ws_color_hex}; border-radius: 4px;"></div></div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; font-size: 0.85rem;">
                <div><div style="color:#9aa0a6; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;">Pass Rate</div><div style="font-size: 1.25rem; color: {_ws_color_hex};">{_ws.pass_rate:.1f}/10</div><div style="color:#9aa0a6; font-size: 0.7rem;">How many checks beat 10%</div></div>
                <div><div style="color:#9aa0a6; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;">Magnitude</div><div style="font-size: 1.25rem; color: {_ws_color_hex};">{_ws.magnitude:.1f}/10</div><div style="color:#9aa0a6; font-size: 0.7rem;">How far above 10% they sit</div></div>
                <div><div style="color:#9aa0a6; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;">Consistency</div><div style="font-size: 1.25rem; color: {_ws_color_hex};">{_ws.consistency:.1f}/10</div><div style="color:#9aa0a6; font-size: 0.7rem;">How stable growth is across four time windows</div><div style="color:#FFFFFF; font-size: 0.7rem;">{_ws.consistency_reason}</div></div>
            </div>
            <div style="margin-top: 0.9rem; font-size: 0.8rem; color: #c5c8cd;"><div><strong style="color:#4CAF50;">Strengths:</strong> {_strength_html}</div><div style="margin-top: 0.2rem;"><strong style="color:#EF5350;">Weaknesses:</strong> {_weakness_html}</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Valuation")
    v1, v2, v3, v4, v5 = st.columns(5)
    v1.metric("Value Price / Intrinsic Value", _fmt_money(val.value_price, fin.exchange, fin.ticker) if val else "n/a")
    if val:
        _src_label = {_MODEL_MODE: "model", "Value conservative": "conservative", "Custom": "custom", "Big 5 EPS growth": "Big 5 EPS", "Analyst 5Y growth": "analyst"}.get(st.session_state.get("valuation_calculated_mode"), val.growth_source)
        v1.markdown(f'<div style="font-size:0.8rem; color:#9aa0a6; white-space:nowrap; overflow:visible;">Growth used: {val.growth_rate * 100:.1f}% ({_src_label})</div>', unsafe_allow_html=True)
    v2.metric("Margin of Safety (MOS) Buy Price", _fmt_money(val.mos_price, fin.exchange, fin.ticker) if val else "n/a")
    v3.metric("Current Price", _fmt_money(fin.current_price, fin.exchange, fin.ticker))
    _render_verdict_cell(v4, v_verdict, v_detail, v_color)
    _render_analyst_cell(v5, fin)

    growth_group_col, button_col = st.columns([4, 1])
    with growth_group_col:
        st.markdown("<div style='font-size: 0.875rem; margin-bottom: 0.25rem;'>Growth source</div>", unsafe_allow_html=True)
        col_radio, col_input, col_pct, col_btn, col_spacer = st.columns([2.1, 1.1, 0.2, 1.0, 1.8])
        with col_radio: selected_growth_mode = st.radio("Growth source", options=mode_options, horizontal=True, format_func=_valuation_growth_rate_label, key="valuation_growth_mode", label_visibility="collapsed")
        with col_input:
            st.markdown("<style>[data-testid=\"stNumberInput\"] { width: 100% !important; }</style>", unsafe_allow_html=True)
            st.markdown("<div style='height: 36px;'></div>", unsafe_allow_html=True)
            st.number_input("Custom %", key="custom_growth_rate", min_value=1, max_value=100, step=1, format="%d", disabled=(selected_growth_mode != "Custom"), label_visibility="collapsed")
        with col_pct: st.markdown("<div style='height: 44px;'></div><div style='font-size: 0.875rem; color: #9aa0a6;'>%</div>", unsafe_allow_html=True)
        with col_btn:
            st.markdown("<div style='height: 36px;'></div>", unsafe_allow_html=True)
            revalue_button = st.button("RE-VALUATE", key="revaluate_btn", help="Recalculate with selected growth rate")

    if revalue_button:
        st.session_state["valuation_growth_result"] = None
        st.rerun()

    if val is not None:
        with st.expander("How the value Price was calculated"):
            _big5_g_txt = _fmt_pct(get_model_growth(fin, big5.eps.values))
            _analyst_g_txt = _fmt_pct(fin.analyst_5yr_growth)
            _hist_pe_txt = f"{fin.pe_ratio_ttm:.1f}" if fin.pe_ratio_ttm else "no data"
            _default_pe = val.growth_rate * 100 * 2
            _mos_pct_int = int(round(100 * (1 - val.mos_price / val.value_price))) if val.value_price else 50
            st.markdown(f"""
**Step 1 — Growth rate (from the source selected above; not capped). The default "Model" option uses the lower of the Big 5 and analyst rates**
- Big 5 EPS growth (median of the 10/5/3-year windows): **{_big5_g_txt}**
- Analyst growth: **{_analyst_g_txt}**
- Used: **{_fmt_pct(val.growth_rate)}** *(source: {val.growth_source}; "Value conservative" fixes this at 15%)*

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
""".strip())

    st.markdown("#### DCF Method Valuation")
    st.caption("Discounted Cash Flow — projects free cash flow forward and discounts it back to today. Adjust the two assumptions below; only this section reacts to the sliders.")
    dcf_col1, dcf_col2 = st.columns(2)
    with dcf_col1: dcf_discount = st.slider("DCF discount rate", min_value=6.0, max_value=15.0, value=10.0, step=0.5, format="%.1f%%", help="Required annual return.", key="dcf_discount_slider") / 100
    with dcf_col2: dcf_terminal = st.slider("DCF terminal growth", min_value=0.0, max_value=4.0, value=2.5, step=0.5, format="%.1f%%", help="Perpetual growth rate after the fade period.", key="dcf_terminal_slider") / 100

    _fcf_ttm_dcf = float(fin.free_cash_flow.iloc[-1]) if not fin.free_cash_flow.empty else None
    _growth_dcf = big5_eps_g if big5_eps_g is not None else fin.analyst_5yr_growth
    _dcf_val = dcf_two_stage(fcf_ttm=_fcf_ttm_dcf, shares_out=fin.shares_outstanding, current_price=fin.current_price, growth_rate=_growth_dcf, discount_rate=dcf_discount, terminal_growth=dcf_terminal)
    if _dcf_val is None: st.info("DCF not computable — needs positive TTM free cash flow and shares outstanding.")
    else:
        d1, d2, d3, d4 = st.columns(4)
        with d1: st.metric("DCF Fair Value", _fmt_money(_dcf_val.fair_value, fin.exchange, fin.ticker))
        with d2: st.metric("DCF MOS Buy Price", _fmt_money(_dcf_val.mos_price, fin.exchange, fin.ticker))
        with d3: st.metric("DCF Upside", _fmt_pct(_dcf_val.upside_pct) if _dcf_val.upside_pct is not None else "n/a")
        with d4: st.metric("DCF Verdict", _dcf_val.verdict)

    st.markdown("---")
    if current_eps is None or current_eps <= 0:
        st.markdown("#### Price-to-Sales (P/S) Valuation")
        st.caption("Since this company has negative earnings or missing data, we use a Price-to-Sales multiple to estimate fair value.")
        _revenue_ttm = float(fin.revenue.iloc[-1]) if not fin.revenue.empty else None
        _shares = fin.shares_outstanding
        if _revenue_ttm and _revenue_ttm > 0 and _shares and _shares > 0:
            _sales_per_share = _revenue_ttm / _shares
            ps_col1, ps_col2 = st.columns(2)
            with ps_col1: _fair_ps_multiple = st.slider("Fair P/S Multiple", min_value=1.0, max_value=30.0, value=10.0, step=1.0, help="A 10x multiple means you are willing to pay $10 for every $1 of revenue.")
            _fair_value_ps = _sales_per_share * _fair_ps_multiple
            _mos_ps = _fair_value_ps * 0.75
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Revenue (TTM)", _fmt_money(_revenue_ttm, fin.exchange, fin.ticker))
            col2.metric("Sales Per Share", _fmt_money(_sales_per_share, fin.exchange, fin.ticker))
            col3.metric("Fair Value (P/S)", _fmt_money(_fair_value_ps, fin.exchange, fin.ticker))
            col4.metric("P/S MOS Buy", _fmt_money(_mos_ps, fin.exchange, fin.ticker))
            if fin.current_price:
                _upside_ps = (_fair_value_ps - fin.current_price) / fin.current_price
                if _upside_ps > 0: st.success(f"Stock is **{_upside_ps*100:.1f}% below** its P/S fair value.")
                else: st.warning(f"Stock is **{abs(_upside_ps)*100:.1f}% above** its P/S fair value.")
        else: st.info("Cannot calculate P/S — missing Revenue or Shares Outstanding.")
        st.markdown("---")

    with st.expander("⚙️ Assumptions for the other methods", expanded=False):
        a_col1, a_col2 = st.columns(2)
        with a_col1: aaa_yield = st.slider("AAA corporate bond yield (Graham Formula)", min_value=2.0, max_value=10.0, value=4.5, step=0.5, format="%.1f%%", help="Used in Graham's revised 1974 formula.", key="aaa_yield_slider") / 100
        with a_col2: mos_pct = st.slider("Margin of Safety (Lynch/Graham/PEG)", min_value=10.0, max_value=60.0, value=25.0, step=5.0, format="%.1f%%", help="Discount applied to fair value to get a buy price.", key="mos_pct_slider") / 100

    _tip_other = _VERDICT_TOOLTIPS.get("BUY", "").replace('"', "&quot;")
    st.markdown(f'<h3 style="margin-bottom: 0.25rem;">Other Intrinsic Value Methods <span title="{_tip_other}" style="cursor: help; color: #4CAF50; font-size: 0.9rem; margin-left: 6px;">&#9432;</span></h3>', unsafe_allow_html=True)
    st.caption("Multiple valuation lenses on the same ticker. Each shows fair value, a margin-of-safety buy price, upside vs current, and a verdict. No single method is right — look for consensus. Hover the ⓘ icon for how to interpret the verdicts.")

    _fcf_ttm_for_dcf = float(fin.free_cash_flow.iloc[-1]) if not fin.free_cash_flow.empty else None
    _growth_for_intrinsic = big5_eps_g if big5_eps_g is not None else fin.analyst_5yr_growth

    from moatcheck.valuation import calculate_median_valuation
    val_res = calculate_median_valuation(current_eps=eps_for_valuation, fcf_ttm=_fcf_ttm_for_dcf, shares_outstanding=fin.shares_outstanding, book_value_per_share=fin.book_value_per_share, dividend_yield=fin.dividend_yield, current_price=fin.current_price, growth_rate=_growth_for_intrinsic, historical_pe=fin.pe_ratio_ttm, discount_rate=dcf_discount, terminal_growth=dcf_terminal, aaa_bond_yield=aaa_yield, mos=mos_pct)

    if not val_res["details"]: st.info("No intrinsic-value methods computable.")
    else:
        def _verdict_with_pct(m: MethodResult) -> str:
            if m.upside_pct is None: return m.verdict
            pct = m.upside_pct * 100
            if m.verdict == "BUY": return f"{m.verdict} ({pct:+.0f}% below fair)"
            if m.verdict == "WATCH": return f"{m.verdict} ({pct:+.0f}% vs fair)"
            if m.verdict == "AVOID": return f"{m.verdict} ({-pct:.0f}% above fair)"
            return m.verdict

        methods_rows = [{"Method": name, "Fair Value": _fmt_money(getattr(m, "fair_value", getattr(m, "value_price", None)), fin.exchange, fin.ticker), "MOS Buy": _fmt_money(m.mos_price, fin.exchange, fin.ticker), "Upside vs Current": _fmt_pct(m.upside_pct), "Verdict": _verdict_with_pct(m)} for name, m in val_res["details"].items()]
        methods_df = pd.DataFrame(methods_rows)
        
        def _style_row(row):
            v = row["Verdict"]
            if v.startswith("BUY"): return ["background-color: #1e4620; color: #b6f0b6;"] * len(row)
            if v.startswith("WATCH"): return ["background-color: #4a3a1e; color: #f0d8a6;"] * len(row)
            if v.startswith("AVOID"): return ["background-color: #4b1e1e; color: #f0b6b6;"] * len(row)
            return [""] * len(row)

        st.dataframe(methods_df.style.apply(_style_row, axis=1), width="stretch", hide_index=True)
        vals = [m.fair_value for m in val_res["details"].values() if m.fair_value is not None]
        if vals:
            median_val = statistics.median(vals)
            average_val = sum(vals) / len(vals)
            st.markdown(f'<div style="background-color: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; display: flex; justify-content: space-around;"><div style="text-align: center;"><div style="color: #9aa0a6; font-size: 0.8rem; text-transform: uppercase;">Median Fair Value</div><div style="font-size: 1.5rem; font-weight: 600; color: #e8eaed;">{_fmt_money(median_val, fin.exchange, fin.ticker)}</div></div><div style="text-align: center;"><div style="color: #9aa0a6; font-size: 0.8rem; text-transform: uppercase;">Average Fair Value</div><div style="font-size: 1.5rem; font-weight: 600; color: #e8eaed;">{_fmt_money(average_val, fin.exchange, fin.ticker)}</div></div></div>', unsafe_allow_html=True)
        
        buys = sum(1 for m in val_res["details"].values() if m.verdict == "BUY")
        watches = sum(1 for m in val_res["details"].values() if m.verdict == "WATCH")
        avoids = sum(1 for m in val_res["details"].values() if m.verdict == "AVOID")
        total = buys + watches + avoids
        if total:
            if buys >= total / 2: st.success(f"Consensus: **BUY** ({buys}/{total} methods say below MOS)")
            elif avoids >= total / 2: st.error(f"Consensus: **AVOID** ({avoids}/{total} methods say above fair value)")
            else: st.warning(f"Consensus: **WATCH** ({buys} buy / {watches} watch / {avoids} avoid)")

        with st.expander("Per-method assumptions"):
            for name, m in val_res["details"].items():
                st.markdown(f"**{name}**")
                st.write(m.assumptions)

    st.markdown("### Health checks")
    fcf_ttm = float(fin.free_cash_flow.iloc[-1]) if not fin.free_cash_flow.empty else None
    ltd_ttm = float(fin.long_term_debt.iloc[-1]) if not fin.long_term_debt.empty else None
    big5_growth_for_payback = big5_eps_g if big5_eps_g is not None else fin.analyst_5yr_growth
    pay = payback_time(fcf_ttm, big5_growth_for_payback, fin.market_cap)
    dtf = debt_to_fcf(ltd_ttm, fcf_ttm)

    h1, h2 = st.columns(2)
    with h1:
        if pay is None: st.metric("Payback Time", "n/a"); st.caption("Needs positive TTM FCF and a valid market cap.")
        else: st.metric("Payback Time", f"{pay:.0f} years", delta=("PASS" if pay <= 8 else "FAIL")); st.caption("Buffett's rule: 8 years or fewer.")
    with h2:
        if dtf is None: st.metric("Debt / FCF", "n/a"); st.caption("Needs positive TTM FCF.")
        else: st.metric("Debt / FCF", f"{dtf:.2f}", delta=("PASS" if dtf < 3 else "FAIL")); st.caption("Buffett's rule: less than 3 years to pay off all long-term debt.")

    st.markdown("### Underlying data (annual)")
    with st.expander("Show raw statement values used"):
        frames = {"Revenue": fin.revenue, "Net Income": fin.net_income, "EPS (Diluted)": fin.eps, "Equity": fin.equity, "Long-Term Debt": fin.long_term_debt, "Operating Cash Flow": fin.operating_cash_flow, "CapEx": fin.capex, "Free Cash Flow": fin.free_cash_flow, "ROIC (per year)": big5.roic_by_year}
        combined = pd.DataFrame({k: v for k, v in frames.items() if not v.empty})
        combined.index.name = "Year"
        st.dataframe(combined.sort_index(ascending=False), width="stretch")

    st.divider()
    _split_line = "EDGAR reports EPS as-filed (not split-adjusted); large year-over-year EPS jumps may reflect a stock split rather than an earnings change. " if _show_split_warning else ""
    st.markdown(f'<div style="color: #6c6f75; font-size: 0.72rem; line-height: 1.5; margin-top: 0.5rem;"><em>{fin.data_source_note} {_split_line} Not investment advice. Cross-check with 10-K filings before any capital decision.</em></div>', unsafe_allow_html=True)