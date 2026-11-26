from __future__ import annotations
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from data_provider import get_ticker
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
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# Import the technical indicator analyzer
from technical_indicators import TechnicalIndicatorAnalyzer

# Global constants
MIN_TRADE_PCT = 2.0   # 2% of account value  - TQQQ/SQQQ Strategy Configuration

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
            clear_yfinance_cache("^NDX")  # sleeve strategy signal source
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
        from sleeves_strategy import get_sleeve_target
        _sl = get_sleeve_target()
        c1, c2, c3 = st.columns(3)

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

        with c3:
            st.markdown("**Sleeves (NDX Trend, Long-or-Cash)**")
            if _sl.get("error"):
                st.warning(_sl["error"])
            else:
                l1, l2, l3 = st.columns(3)
                l1.metric("TQQQ Target", f"{_sl['tqqq_pct']:.1f}%")
                l2.metric("SQQQ Target", f"{_sl['sqqq_pct']:.1f}%")
                l3.metric("Cash",        f"{_sl['cash_pct']:.1f}%")
                _stale = _sl["as_of"] is not None and _sl["as_of"] < datetime.now().date() - timedelta(days=3)
                st.caption(
                    f"NDX regime: {_sl['regime']} · data as of {_sl['as_of']}"
                    + (" ⚠️ stale" if _stale else "")
                )

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
            st.session_state["tqqq_capital"] = net_cap if net_cap >= 100.0 else 5000.0

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

            # --- Sleeve strategy (trend-regime sleeve, long-or-cash) ---
            st.markdown("---")
            st.markdown("**Sleeve Strategy** *(trend regime on NDX, long-or-cash)*")
            if _sl.get("error"):
                st.warning(_sl["error"])
            else:
                _sl_t = compute_target_shares(
                    account_value=account_value,
                    tqqq_target_pct=_sl["tqqq_pct"],
                    sqqq_target_pct=_sl["sqqq_pct"],
                    tqqq_price=tqqq_price,
                    sqqq_price=sqqq_price,
                )
                _sl_r = compute_rebalance(
                    current_tqqq_shares=current_tqqq,
                    current_sqqq_shares=current_sqqq,
                    tqqq_target_shares=_sl_t["tqqq_target_shares"],
                    sqqq_target_shares=_sl_t["sqqq_target_shares"],
                    tqqq_price=tqqq_price,
                    sqqq_price=sqqq_price,
                )
                s1, s2, s3 = st.columns(3)
                s1.metric("TQQQ", f"{_sl['tqqq_pct']:.1f}%", f"${_sl_t['tqqq_target_value']:,.2f}")
                s2.metric("SQQQ", f"{_sl['sqqq_pct']:.1f}%", f"${_sl_t['sqqq_target_value']:,.2f}")
                s3.metric("Cash", f"{_sl['cash_pct']:.1f}%", f"${_sl_t['cash_value']:,.2f}")
                st.caption(
                    f"NDX regime: {_sl['regime']} · as of {_sl['as_of']} · "
                    f"target {_sl_t['tqqq_target_shares']:.4f} TQQQ / {_sl_t['sqqq_target_shares']:.4f} SQQQ shares"
                )
                render_action_block(_sl_r)

            # Log trade buttons
            # Snapshot signal + target so a refresh between display and click
            # can't desync what gets logged
            # Logged against the SLEEVE strategy (the one being traded). If the
            # sleeve is unavailable, fall back to the dial and label it as such.
            if _sl.get("error"):
                _log_rb = rebalance
                signal_snapshot = f"DIAL-FALLBACK {tqqq_signals['signal']}"
                target_snapshot = tqqq_signals["target_tqqq_pct"]
                sqqq_target_snapshot = tqqq_signals["target_sqqq_pct"]
                cash_target_snapshot = tqqq_signals["target_cash_pct"]
                log_notes = "sleeve unavailable"
            else:
                _log_rb = _sl_r
                signal_snapshot = f"SLEEVE {_sl['regime']}"
                target_snapshot = _sl["tqqq_pct"]
                sqqq_target_snapshot = _sl["sqqq_pct"]
                cash_target_snapshot = _sl["cash_pct"]
                log_notes = f"sleeve as_of={_sl['as_of']}; dial={tqqq_signals['signal']} {tqqq_signals['target_tqqq_pct']:.1f}%"

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
                        delta = _log_rb["tqqq_delta_shares"]
                        action = "BUY" if delta > 0 else "SELL"

                        log_trade(
                            ticker="TQQQ", action=action,
                            shares=abs(actual_tqqq_fill),
                            price=tqqq_price,
                            signal=signal_snapshot,
                            tqqq_target_pct=target_snapshot,
                            sqqq_target_pct=sqqq_target_snapshot,
                            cash_target_pct=cash_target_snapshot,
                            account_value=account_value,
                            notes=log_notes,
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
                        delta = _log_rb["sqqq_delta_shares"]
                        action = "BUY" if delta > 0 else "SELL"

                        log_trade(
                            ticker="SQQQ", action=action,
                            shares=abs(actual_sqqq_fill),
                            price=sqqq_price,
                            signal=signal_snapshot,
                            tqqq_target_pct=target_snapshot,
                            sqqq_target_pct=sqqq_target_snapshot,
                            cash_target_pct=cash_target_snapshot,
                            account_value=account_value,
                            notes=log_notes,
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

# Leverage vs QQQ for the Sleeves vol overlay when trend signals are read from QQQ.
LEVERAGED_NASDAQ = {"TQQQ": 3.0, "QLD": 2.0}


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
        "- **Sleeves (traded live)** — trend-regime sleeve: 80% weight above both the 50- and 250-day MA, 40% between, 0% below both, scaled down when volatility is high and moved at most 10 points a day. Long-or-cash.\n"
        "- **7-Step Dial (live rules)** — the live 7-vote dial, calling the same vote functions as the app. Holds TQQQ at the dial weight when bullish, cash otherwise (the short leg is not traded).\n"
        "- **One Percent A Week** — weekly swing. Buys a 1% close below the week's first open; exits on +1%, a 0.5% stop, or at the week's last close. Never held over a weekend.\n"
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
    c0, c0b, c1, c2, c3, c4 = st.columns([1.2, 1.5, 1.2, 1.4, 1.1, 1.0])

    with c0:
        ticker_input = st.text_input("Ticker", value="TQQQ", key="backtest_ticker")
        backtest_ticker = ticker_input.strip().upper()

    with c0b:
        signal_choice = st.selectbox(
            "Trend signals read",
            options=["Traded ticker's own price", "QQQ (Nasdaq-100)"],
            index=1 if backtest_ticker in LEVERAGED_NASDAQ else 0,
            help="The regime logic (50/250-day MAs, Bollinger, 200-day filter) reads this price series. "
                 "Use QQQ for leveraged Nasdaq ETFs like TQQQ, as the live app does. "
                 "The swing and One Percent A Week rules always use the traded ticker's own price.",
            key=f"backtest_signal_choice_{backtest_ticker in LEVERAGED_NASDAQ}",
        )
        signal_symbol = backtest_ticker if signal_choice.startswith("Traded") else "QQQ"
        
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
            run_clicked = _btn_slot.button("▶ Run All", type="primary", width="stretch", key="backtest_btn")
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
                    prices = load_backtest_data(backtest_ticker, start_date, end_date, signal_symbol=signal_symbol)
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

            start_ts = pd.Timestamp(start_date)
            # Sleeves' vol overlay assumes the traded ticker is levered vs the signal series (TQQQ vs QQQ = 3x).
            leverage = LEVERAGED_NASDAQ.get(backtest_ticker, 1.0) if signal_symbol == "QQQ" else 1.0

            window = prices[prices.index >= start_ts]
            if len(window) < 30:
                st.warning(f"Only {len(window)} bars in the window for {backtest_ticker}; pick a longer period.")
                st.session_state["backtest_running"] = False
                return

            results = {}
            for name, fn in bs.STRATEGIES.items():
                # History before start_ts only warms up indicators; all trade state starts clean there.
                sig = fn(prices, trade_start=start_ts, swing_pct=swing_pct / 100.0, leverage=leverage)
                results[name] = run_backtest(
                    window, sig.weight, starting_cash=starting_cash, warmup_days=0,
                    exit_at_close=sig.exit_at_close, ticker=backtest_ticker,
                )

            st.session_state["backtest_meta"] = {"ticker": backtest_ticker, "signal_symbol": signal_symbol, "leverage": leverage}
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
    meta = st.session_state.get("backtest_meta", {})
    bt_ticker = first.ticker or "Ticker"
    st.caption(
        f"Window: **{first.start_date.date()} → {first.end_date.date()}** "
        f"({(first.end_date - first.start_date).days} days) · "
        f"Starting cash: **${first.starting_cash:,.0f}** · "
        f"Ticker: **{bt_ticker}** · Trend signals from: **{meta.get('signal_symbol', bt_ticker)}**"
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
            "Avg Exposure": f"{r.avg_exposure*100:.0f}%",
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
        width="stretch",
        hide_index=True,
    )
    
    st.caption(
        "\\* **Sharpe** = √252 × mean daily return ÷ std dev of daily returns (risk-free rate 0). "
        "Measures return per unit of day-to-day risk. Higher is better. Realistic range is 0.5 - 2.0 where 2 is excellent. Anything greater than 2 is Very Rare and should be treated with caution.\n\n"
        "\\*\\* **Calmar** = CAGR ÷ Max Drawdown. Measures return per unit of "
        "worst-case loss. Higher is better. Realistic range is 0.5 - 1.0 (Good). Between 1-2 is Very good but very rare.\n\n"
        "\\^ **TASC** = Technical Analysis of Stock Charts. a monthly magazine that has been published since 1982. The 1% per week strategy is inspired by its teachings."
    )    

    # ---------- Benchmarks ----------
    st.markdown("#### Buy & Hold Benchmarks")
    bc1, bc2 = st.columns(2)
    bc1.metric(f"{bt_ticker} Buy & Hold", f"{first.ticker_buy_hold_return*100:+.2f}%", delta=f"${first.starting_cash * (1 + first.ticker_buy_hold_return):,.0f}", delta_color="normal" if first.ticker_buy_hold_return >= 0 else "inverse")
    bc2.metric("QQQ Buy & Hold", f"{first.qqq_buy_hold_return*100:+.2f}%", delta=f"${first.starting_cash * (1 + first.qqq_buy_hold_return):,.0f}", delta_color="normal" if first.qqq_buy_hold_return >= 0 else "inverse")

    # ---------- Equity curve — all strategies on one chart ----------
    st.markdown("#### Equity Curves")
    import plotly.graph_objects as go
    fig = go.Figure()

    strategy_colors = ["#4DABF7", "#FF6B6B", "#FFD43B", "#B388FF", "#00E676", "#FFA726", "#26C6DA", "#F06292"]
    for i, (name, r) in enumerate(results.items()):
        color = strategy_colors[i % len(strategy_colors)]
        fig.add_trace(go.Scatter(x=r.equity_curve.index, y=r.equity_curve, name=name, line=dict(color=color, width=2.5)))

    # Buy-and-hold of the traded ticker as a dashed reference line
    fig.add_trace(go.Scatter(x=first.benchmark_ticker.index, y=first.benchmark_ticker, name=f"{bt_ticker} Buy & Hold", line=dict(color="#9AA0A6", width=1.5, dash="dot")))

    fig.update_layout(
        height=440,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Portfolio Value ($)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, width="stretch")

    # ---------- 15% Swing Analysis ----------
    st.markdown("#### 15% Swing Analysis")
    st.caption(
        "Peak-to-trough and trough-to-peak moves of at least the threshold "
        "below. Shows how often large swings occur and how much time "
        "separates them."
    )

    from swing_analysis import find_swings
    tqqq_series = prices.loc[prices.index >= first.start_date, "close"]

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
        st.dataframe(display, width="stretch", hide_index=True)

        # Chart with markers
        # Chart with markers
        import plotly.graph_objects as go
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=tqqq_series.index, y=tqqq_series, name=f"{bt_ticker} Price", line=dict(color="#4DABF7", width=2)))

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
            yaxis_title=f"{bt_ticker} Price ($)", 
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig2, width="stretch")

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
            st.dataframe(display, width="stretch", hide_index=True)

    st.caption(
        "Backtest is hypothetical. Past performance does not guarantee future "
        "results. No transaction costs, slippage, or taxes are modeled."
    )