"""My Investments page — read a Fidelity positions CSV and render a portfolio dashboard."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import json
from datetime import datetime
import json
import statistics
from pathlib import Path

DATA_DIR = Path("data")
# Look for any of these in priority order; first one that exists wins.
CSV_CANDIDATES = [
    DATA_DIR / "fidelity_positions.csv",
]

def _market_is_open() -> bool:
    """Return True if US equity markets are currently open (ET, Mon–Fri, 9:30–16:00)."""
    from datetime import datetime, time as dtime
    from zoneinfo import ZoneInfo

    try:
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return False

    if now_et.weekday() >= 5:          # Sat / Sun
        return False
    open_t = dtime(9, 30)
    close_t = dtime(16, 0)
    return open_t <= now_et.time() <= close_t

def _fetch_mover_data(symbol: str, quantity: float, position_value: float) -> dict | None:
    """Fetch OHLCV and compute mover metrics for one symbol.

    Returns None if the symbol can't be fetched or has no usable data.
    """
    from data_provider import get_ticker
    import math

    try:
        hist = get_ticker(symbol).history(period="30d")
    except Exception:
        return None

    if hist is None or len(hist) < 2:
        return None

    # Which bar to use for "today"?
    # - If market is open, today's partial bar is the current session.
    # - Otherwise, the last complete bar (yesterday) is what matters.
    # We use the last row either way — yfinance returns the most recent bar.
    today = hist.iloc[-1]
    yesterday = hist.iloc[-2]

    prev_close = float(yesterday["Close"])
    current = float(today["Close"])
    if prev_close <= 0:
        return None

    pct_change = (current - prev_close) / prev_close * 100.0
    dollar_impact = position_value * pct_change / 100.0

    # RVOL: today's volume so far vs. what we'd expect by now.  During the session, we scale the 20-day average by the fraction of
    # the trading day that has elapsed. Without this, every stock looks like RVOL 0.4–0.6 until 4 PM.
    rvol = 1.0
    try:
        vol_today = float(today["Volume"])
        tail = hist["Volume"].iloc[-21:-1]
        vol_avg = float(tail.mean()) if len(tail) > 0 else 0.0
        elapsed = _elapsed_session_fraction()
        vol_expected = vol_avg * elapsed if vol_avg > 0 else 0.0
        if vol_expected > 0 and vol_today > 0:
            rvol = vol_today / vol_expected
    except Exception:
        pass

    # RVOL multiplier: normalized log, floors at 0.5 for very thin volume
    # ln(rvol + 1) / ln(2) maps rvol=1 to 1.0, rvol=2 to ~1.58, rvol=5 to ~2.58
    rvol_mult = max(0.5, math.log(rvol + 1.0) / math.log(2.0)) if rvol > 0 else 1.0

    # Freshness: where is the current price within today's range?
    # For gainers, near the high = fresh; for losers, near the low = fresh.
    high = float(today["High"])
    low = float(today["Low"])
    rng = high - low
    if rng > 0:
        pos_in_range = (current - low) / rng            # 0 = at low, 1 = at high
    else:
        pos_in_range = 0.5

    if pct_change >= 0:
        freshness = 0.5 + pos_in_range                  # [0.5, 1.5]
    else:
        freshness = 1.5 - pos_in_range                  # [0.5, 1.5]
    freshness = max(0.5, min(1.5, freshness))

    # Composite score
    score = pct_change * rvol_mult * freshness

    return {
        "Symbol": symbol,
        "Quantity": quantity,
        "Value": position_value,
        "Pct": pct_change,
        "Dollar": dollar_impact,
        "RVOL": rvol,
        "Freshness": freshness,
        "Score": score,
        "AtHigh": pct_change >= 0 and pos_in_range >= 0.9,
        "AtLow": pct_change < 0 and pos_in_range <= 0.1,
    }

def _top_movers(df: pd.DataFrame, n: int = 7) -> tuple[list[dict], list[dict], dict]:
    """Rank positions by mover score, aggregated across accounts.

    Returns (gainers, losers, summary). Each mover is a dict with
    Symbol, Pct, Dollar, RVOL, Freshness, Score, AtHigh, AtLow, Accounts.
    """
    # Only real equity symbols with non-zero value
    view = df[
        df["Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)
        & (df["_value"] > 0)
        & (df["Quantity"] > 0)
    ]

    if view.empty:
        return [], [], {}

    # Aggregate across accounts: NVDA in Rollover + Roth is one row
    agg: dict[str, dict] = {}
    for _, r in view.iterrows():
        sym = str(r["Symbol"])
        qty = float(r["Quantity"]) if r["Quantity"] == r["Quantity"] else 0.0
        val = float(r["_value"])
        if sym not in agg:
            agg[sym] = {"qty": 0.0, "value": 0.0, "accounts": []}
        agg[sym]["qty"] += qty
        agg[sym]["value"] += val
        agg[sym]["accounts"].append(str(r["Account name"]))

    # Fetch + score
    rows = []
    for sym, info in agg.items():
        m = _fetch_mover_data(sym, info["qty"], info["value"])
        if m is None:
            continue
        m["Accounts"] = info["accounts"]
        rows.append(m)

    if not rows:
        return [], [], {}

    movers = pd.DataFrame(rows)

    gainers = (
        movers[movers["Pct"] > 0]
        .sort_values("Dollar", ascending=False)
        .head(n)
        .to_dict("records")
    )
    losers = (
        movers[movers["Pct"] < 0]
        .sort_values("Dollar", ascending=True)
        .head(n)
        .to_dict("records")
    )

    total_value = float(movers["Value"].sum())
    net_dollar = float(movers["Dollar"].sum())
    net_pct = (net_dollar / (total_value - net_dollar) * 100.0) if total_value > 0 else 0.0

    summary = {
        "net_dollar": net_dollar,
        "net_pct": net_pct,
        "n_up": int((movers["Pct"] > 0).sum()),
        "n_down": int((movers["Pct"] < 0).sum()),
        "bar_source": "today" if _market_is_open() else "yesterday's close",
    }
    return gainers, losers, summary

def _elapsed_session_fraction() -> float:
    """Fraction of the 6.5-hour US trading session that has elapsed (0.0–1.0).

    Returns 1.0 outside market hours (so the calc falls back to full-day
    comparison on weekends, after-hours, and pre-market).
    """
    from datetime import datetime, time as dtime
    from zoneinfo import ZoneInfo

    try:
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return 1.0

    if now_et.weekday() >= 5:
        return 1.0

    t = now_et.time()
    if t < dtime(9, 30):
        return 1.0          # pre-market — treat as full day (yesterday's close)
    if t >= dtime(16, 0):
        return 1.0          # after hours — full day

    # Minutes since 9:30 ET
    minutes = (now_et.hour * 60 + now_et.minute) - (9 * 60 + 30)
    minutes = max(0, min(int(6.5 * 60), minutes))
    return max(0.05, minutes / (6.5 * 60))


def _rvol_html(rvol: float) -> str:
    """Format the RVOL cell with a hover tooltip explaining the badge."""
    if rvol >= 2.0:
        tier = "High conviction — trading well above normal volume"
        color = "#FFA726"
        label = f"· RVOL {rvol:.1f}×"
    elif rvol >= 1.5:
        tier = "Elevated volume — move has participation"
        color = "#9aa0a6"
        label = f"· RVOL {rvol:.1f}×"
    elif rvol >= 0.6:
        tier = "Normal volume range"
        color = "#6c6f75"
        label = f"· {rvol:.1f}×"
    else:
        tier = "Light volume — move has little participation, treat as noise"
        color = "#6c6f75"
        label = f"· {rvol:.1f}× 💤"

    tip = (
        f"RVOL = today's volume / expected volume by this time of day. "
        f"{tier}."
    ).replace('"', "&quot;")

    return (
        f'<span title="{tip}" style="color:{color}; cursor: help;">'
        f"{label}</span>"
    )


def _render_top_movers(df: pd.DataFrame) -> None:
    st.markdown("#### Top Movers")
    st.caption(
        "Ranked by dollar impact on your portfolio. "
        "RVOL ≥ 2× means the move has real volume behind it."
    )

    with st.spinner("Computing movers…"):
        gainers, losers, summary = _top_movers(df, n=7)

    if not gainers and not losers:
        st.info("No mover data available (yfinance may be down or tickers unlisted).")
        return

    left, mid, right = st.columns([2, 1, 2])

    def _card(row: dict) -> str:
        pct = row["Pct"]
        dollar = row["Dollar"]
        color = "#4CAF50" if pct >= 0 else "#EF5350"
        arrow = "▲" if pct >= 0 else "▼"

        rvol_html = _rvol_html(row["RVOL"])

        fresh = ""
        if row.get("AtHigh"):
            fresh = ' <span style="color:#00E676; font-size:0.7rem;">HOD</span>'
        elif row.get("AtLow"):
            fresh = ' <span style="color:#EF5350; font-size:0.7rem;">LOD</span>'

        accounts = row.get("Accounts", [])
        acct_txt = accounts[0][:22] if accounts else ""

        return f"""
        <div style="padding:0.4rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <div style="font-weight:600;">{row['Symbol']}{fresh}</div>
                <div style="color:{color}; font-weight:600;">{arrow} {pct:+.2f}%</div>
            </div>
            <div style="display:flex; justify-content:space-between;
                        font-size:0.72rem; color:#9aa0a6; margin-top:0.15rem;">
                <div>{acct_txt}</div>
                <div>${dollar:+,.0f} {rvol_html}</div>
            </div>
        </div>
        """

    with left:
        st.markdown(
            '<div style="font-weight:600; color:#4CAF50; margin-bottom:0.4rem;">🟢 Top Gainers</div>',
            unsafe_allow_html=True,
        )
        if gainers:
            for g in gainers:
                st.markdown(_card(g), unsafe_allow_html=True)
        else:
            st.caption("No gainers today.")

    with mid:
        net = summary.get("net_dollar", 0.0)
        net_pct = summary.get("net_pct", 0.0)
        color = "#4CAF50" if net >= 0 else "#EF5350"
        st.markdown(
            f"""
            <div style="text-align:center; padding:1.25rem 0.5rem;
                        border:1px solid rgba(255,255,255,0.08); border-radius:10px;">
                <div style="font-size:0.7rem; color:#9aa0a6;
                            text-transform:uppercase; letter-spacing:0.05em;">
                    Net Portfolio Move
                </div>
                <div style="font-size:1.9rem; font-weight:700; color:{color}; margin-top:0.5rem;">
                    ${net:+,.0f}
                </div>
                <div style="font-size:1.1rem; color:{color};">
                    {net_pct:+.2f}%
                </div>
                <div style="font-size:0.72rem; color:#9aa0a6; margin-top:0.7rem;">
                    {summary.get('n_up', 0)} up · {summary.get('n_down', 0)} down
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div style="font-weight:600; color:#EF5350; margin-bottom:0.4rem;">🔴 Top Losers</div>',
            unsafe_allow_html=True,
        )
        if losers:
            for l in losers:
                st.markdown(_card(l), unsafe_allow_html=True)
        else:
            st.caption("No losers today.")

def _money(v) -> str:
    """Compact dollar formatter: $2.52M, $12.1K, $45.00."""
    if v is None or v != v:
        return "—"
    av = abs(v)
    if av >= 1e6:
        return f"${v / 1e6:.2f}M"
    if av >= 1e3:
        return f"${v / 1e3:.1f}K"
    return f"${v:,.2f}"


def _pct_frac(v) -> str:
    """Growth rates stored as fractions: 0.224 → +22.4%."""
    return f"{v * 100:+.1f}%" if v is not None and v == v else "—"


def _pct_num(v) -> str:
    """Fidelity's Total gain/loss percent, already a percent number."""
    return f"{v:+.1f}%" if v is not None and v == v else "—"

def _color_vs_intrinsic(v):
    """Color the vs Intrinsic cell.

    Convention: negative = price below fair value = cheap = GREEN.
                positive = price above fair value = expensive = RED.

    Thresholds are on the absolute gap:
      <= -15%  bright green  (clearly undervalued)
      <=  -5%  green         (modest discount)
      -5..+5%  grey          (near fair value)
      >=  +5%  orange        (modest premium)
      >= +15%  red           (clearly overvalued)
    """
    if v is None or v != v:
        return ""
    if v <= -0.15:
        return "color: #00E676;"   # bright green — undervalued
    if v <= -0.05:
        return "color: #4CAF50;"   # green — modest discount
    if v >= 0.15:
        return "color: #EF5350;"   # red — overvalued
    if v >= 0.05:
        return "color: #FFA726;"   # orange — modest premium
    return "color: #9AA0A6;"       # grey — near fair value


def _color_wonder(v):
    """Green when wonderfullness is high, red when low."""
    if v is None or v != v:
        return ""
    if v >= 8.0:
        return "color: #00E676;"
    if v >= 6.5:
        return "color: #4CAF50;"
    if v >= 5.0:
        return "color: #FFA726;"
    return "color: #EF5350;"

def _build_signals(df: pd.DataFrame) -> dict:
    """Compute Big5 growth + multi-method valuation for every equity symbol.

    Slow on first run (~1–3 min for 30–40 symbols). Writes nothing itself;
    the caller caches the returned dict.
    """
    from moatcheck import (
        fetch, compute_big5, dcf_two_stage, peter_lynch_fair,
        graham_number, graham_formula, peg_fair_value,
    )
    from moatcheck.fetcher import FetchError

    symbols = sorted({
        s for s in df["Symbol"].dropna().unique()
        if isinstance(s, str) and s.isalpha() and 1 <= len(s) <= 5
    })

    out: dict[str, dict] = {}
    if not symbols:
        return out

    progress = st.progress(0.0, text="Analyzing positions…")
    for i, sym in enumerate(symbols):
        progress.progress(
            (i + 1) / len(symbols),
            text=f"Analyzing {sym} ({i+1}/{len(symbols)})…",
        )

        try:
            fin = fetch(sym)
        except FetchError as e:
            print(f"[signals] {sym} FetchError: {e}")
            out[sym] = {}
            continue
        except Exception as e:
            import traceback
            print(f"[signals] {sym} EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()
            out[sym] = {}
            continue

        try:
            big5 = compute_big5(fin)
            eps_w = big5.eps.values          # {10: v, 5: v, 3: v, 1: v}
            rev_w = big5.sales.values

            eps_ttm = float(fin.eps.iloc[-1]) if not fin.eps.empty else None
            current_eps = eps_ttm if (eps_ttm and eps_ttm > 0) else None
            fcf_ttm = (
                float(fin.free_cash_flow.iloc[-1])
                if not fin.free_cash_flow.empty else None
            )
            growth = eps_w.get(5) or eps_w.get(3)

            vals: list[tuple[str, float]] = []
            if current_eps and growth:
                if fcf_ttm is not None and fcf_ttm > 0:
                    dcf = dcf_two_stage(
                        fcf_ttm=fcf_ttm,
                        shares_out=fin.shares_outstanding,
                        current_price=fin.current_price,
                        growth_rate=growth,
                        discount_rate=0.10,
                        terminal_growth=0.025,
                    )
                    if dcf and dcf.fair_value:
                        vals.append(("DCF", dcf.fair_value))
                

                lynch = peter_lynch_fair(
                    current_eps=current_eps, growth_rate=growth,
                    dividend_yield=fin.dividend_yield,
                    current_price=fin.current_price, mos=0.25,
                )
                if lynch and lynch.fair_value:
                    vals.append(("Lynch", lynch.fair_value))

                gn = graham_number(
                    current_eps=current_eps,
                    book_value_per_share=fin.book_value_per_share,
                    current_price=fin.current_price, mos=0.25,
                )
                if gn and gn.fair_value:
                    vals.append(("Graham #", gn.fair_value))

                gf = graham_formula(
                    current_eps=current_eps, growth_rate=growth,
                    current_price=fin.current_price,
                    aaa_bond_yield=0.045, mos=0.25,
                )
                if gf and gf.fair_value:
                    vals.append(("Graham F", gf.fair_value))

                peg = peg_fair_value(
                    current_eps=current_eps, growth_rate=growth,
                    current_price=fin.current_price, mos=0.25,
                )
                if peg and peg.fair_value:
                    vals.append(("PEG", peg.fair_value))

            if len(vals) >= 2:
                intrinsic = statistics.median(v for _, v in vals)
                method = f"median of {len(vals)}"
            elif len(vals) == 1:
                intrinsic = vals[0][1]
                method = vals[0][0]
            else:
                intrinsic = None
                method = None

            out[sym] = {
                "rev_10y": rev_w.get(10),
                "rev_5y":  rev_w.get(5),
                "rev_3y":  rev_w.get(3),
                "rev_1y":  rev_w.get(1),
                "eps_10y": eps_w.get(10),
                "eps_5y":  eps_w.get(5),
                "eps_3y":  eps_w.get(3),
                "eps_1y":  eps_w.get(1),
                "wonderfulness": big5.wonderfulness().overall,
                "intrinsic_value": intrinsic,
                "intrinsic_method": method,
            }
        except Exception as e:
            import traceback
            print(f"[signals] {sym} inner EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()
            out[sym] = {}

    progress.empty()
    return out


def _load_or_build_signals(df: pd.DataFrame) -> dict:
    """Load signals from cache, or build and cache if missing/stale."""
    cache_path = Path("data/investments_signals.json")
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    sig = _build_signals(df)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sig, indent=2, default=str))
        tmp.replace(cache_path)
    except Exception:
        pass
    return sig

def _fmt_compact(v: float | None) -> str:
    """Format a dollar amount compactly: $2.52M, $12.1K, $45.00."""
    if v is None:
        return "n/a"
    sign = "-" if v < 0 else ""
    av = abs(v)
    if av >= 1e9:
        return f"{sign}${av / 1e9:.2f}B"
    if av >= 1e6:
        return f"{sign}${av / 1e6:.2f}M"
    if av >= 1e3:
        return f"{sign}${av / 1e3:.1f}K"
    return f"{sign}${av:,.0f}"

def _find_csv() -> Path | None:
    for p in CSV_CANDIDATES:
        if p.exists():
            return p
    # Fall back to the newest Fidelity*.csv in data/
    if DATA_DIR.exists():
        matches = sorted(DATA_DIR.glob("Fidelity*.csv"), key=lambda p: p.stat().st_mtime)
        if matches:
            return matches[-1]
    return None

def _metric_html(label: str, value: str, sub: str = "", help_text: str = "",
                 sub_color: str = "#9aa0a6") -> str:
    """Render a metric card as HTML so we can control the help tooltip glyph."""
    tip = help_text.replace('"', "&quot;")
    info = (
        f'<span title="{tip}" style="cursor: help; color: #4CAF50; '
        f'margin-left: 4px; font-size: 0.85rem;">&#9432;</span>'
        if help_text else ""
    )
    sub_html = (
        f'<div style="font-size: 0.8rem; color: {sub_color}; margin-top: 0.15rem;">{sub}</div>'
        if sub else ""
    )
    return f"""
    <div style="line-height: 1.25;">
        <div style="color: rgba(250,250,250,0.6); font-size: 0.875rem; margin-bottom: 0.2rem;">
            {label}{info}
        </div>
        <div style="font-size: 1.75rem; font-weight: 400;">{value}</div>
        {sub_html}
    </div>
    """

def _parse_fidelity_csv(path: Path) -> pd.DataFrame:
    """Parse the Fidelity positions export into a clean dataframe."""
    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        skipinitialspace=True,
        engine="python",
        on_bad_lines="skip",
        index_col=False,          # <-- THE FIX: don't use first column as index
    )

    # Strip BOM and whitespace from column names
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]

    # Drop any Unnamed columns (Fidelity's trailing comma produces one)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Drop rows where Symbol is empty or looks like a disclaimer
    df = df[df["Symbol"].astype(str).str.strip() != ""]
    df = df[~df["Symbol"].astype(str).str.startswith('"', na=False)]

    # Numeric conversion — non-numeric becomes NaN
    numeric_cols = [
        "Quantity", "Last price", "Last price change", "Current value",
        "Today's gain/loss dollar", "Today's gain/loss percent",
        "Total gain/loss dollar", "Total gain/loss percent",
        "Percent of account", "Cost basis total", "Average cost basis",
    ]

    def _num(series: pd.Series) -> pd.Series:
        s = (
            series.astype(str)
            .str.strip()
            .str.replace(r"[$,%+]", "", regex=True)
            .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
            .replace({"": None, "--": None, "n/a": None, "N/A": None, "Cash": None})
        )
        return pd.to_numeric(s, errors="coerce")

    for col in numeric_cols:
        if col in df.columns:
            df[col] = _num(df[col])

    return df


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns the dashboard needs."""
    out = df.copy()

    # Only positions with a market value participate in the totals.
    out["_value"] = out["Current value"].fillna(0.0)
    out["_basis"] = out["Cost basis total"].fillna(0.0)

    total_value = out["_value"].sum()
    out["% of portfolio"] = out["_value"] / total_value if total_value else 0.0

    # Concentration flag — over 10% of portfolio in a single name
    out["Concentration"] = out["% of portfolio"].apply(
        lambda w: "⚠ over 10%" if w >= 0.10 else ""
    )

    # Position size bucket
    out["Size"] = pd.cut(
        out["% of portfolio"],
        bins=[-1, 0.001, 0.01, 0.05, 0.10, 1.0],
        labels=["", "<1%", "1–5%", "5–10%", ">10%"],
    ).astype(str)

    return out

def _load_portfolio_capital() -> dict | None:
    """Load data/portfolio_capital.json if it exists."""
    path = Path("data/portfolio_capital.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _compute_net_deposits(cap: dict | None) -> float | None:
    """Total cash put into the portfolio, net of withdrawals."""
    if cap is None:
        return None
    starting = float(cap.get("starting_capital", 0) or 0)
    deposits = sum(float(d.get("amount", 0)) for d in cap.get("deposits", []))
    withdrawals = sum(float(w.get("amount", 0)) for w in cap.get("withdrawals", []))
    net = starting + deposits - withdrawals
    return net if net > 0 else None


def _years_since(date_str: str | None) -> float | None:
    if not date_str:
        return None
    try:
        inception = datetime.fromisoformat(date_str)
        delta = datetime.now() - inception
        years = delta.days / 365.25
        return years if years > 0 else None
    except Exception:
        return None


def _hero_strip(df: pd.DataFrame) -> None:
    equities = df[df["_value"] > 0]
    total_value = equities["_value"].sum()
    total_basis = equities["_basis"].sum()
    lifetime_gain = total_value - total_basis
    lifetime_gain_pct = (lifetime_gain / total_basis * 100) if total_basis else 0.0

    cap = _load_portfolio_capital()
    net_deposits = _compute_net_deposits(cap)
    total_ror = (
        (total_value - net_deposits) / net_deposits
        if net_deposits and net_deposits > 0
        else None
    )

    annualized = None
    if total_ror is not None and cap is not None:
        years = _years_since(cap.get("inception_date"))
        if years and years > 0:
            try:
                annualized = (1 + total_ror) ** (1 / years) - 1
            except (ValueError, OverflowError):
                annualized = None

    today_dollars = df["Today's gain/loss dollar"].fillna(0.0).sum()
    today_pct = (
        today_dollars / (total_value - today_dollars) * 100
        if (total_value - today_dollars) > 0
        else 0.0
    )

    # Seven cards, one row.
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    c1.metric("Portfolio Value", _fmt_compact(total_value))
    c2.metric("Cost Basis", _fmt_compact(total_basis))

    c3.markdown(_metric_html(
        "Total Lifetime Gain",
        _fmt_compact(lifetime_gain),
        sub=f"↑ {lifetime_gain_pct:+.1f}%",
        sub_color="#4CAF50" if lifetime_gain >= 0 else "#EF5350",
        help_text="Unrealized gain on current holdings: Portfolio Value − Cost Basis.",
    ), unsafe_allow_html=True)

    c4.markdown(_metric_html(
        "Total ROR",
        f"{total_ror * 100:+.1f}%" if total_ror is not None else "n/a",
        help_text=(
            "(Portfolio Value − Net Deposits) / Net Deposits. "
            "Add data/portfolio_capital.json to enable."
            if total_ror is None else
            "Your actual return on all invested capital since inception."
        ),
    ), unsafe_allow_html=True)

    c5.markdown(_metric_html(
        "Annualized Return",
        f"{annualized * 100:+.1f}%" if annualized is not None else "n/a",
        help_text="Compounded annual return derived from Total ROR and holding period.",
    ), unsafe_allow_html=True)

    # Today's Change — colored UP / DOWN
    if today_dollars > 0:
        t_delta, t_color = "UP", "normal"
    elif today_dollars < 0:
        t_delta, t_color = "DOWN", "inverse"
    else:
        t_delta, t_color = "FLAT", "off"

    c6.metric("Today's Change", _fmt_compact(today_dollars), delta=f"{today_pct:+.2f}%", delta_color="normal" if today_dollars >= 0 else "inverse")

    c7.metric("Positions", f"{len(equities)}")

def _account_breakdown(df: pd.DataFrame) -> None:
    st.markdown("#### Account Breakdown")

    rows = []
    for acct, group in df.groupby("Account name", dropna=False):
        if pd.isna(acct) or str(acct).strip() == "":
            continue
        # Cash rows have no Symbol but a Current value — count them.
        val = group["_value"].sum()
        basis = group["_basis"].sum()
        gain = val - basis
        gain_pct = (gain / basis * 100) if basis else 0.0

        # Top non-cash holding by value
        holdings = group[group["Symbol"].str.match(r"^[A-Z]", na=False)]
        top_symbol = ""
        top_weight = 0.0
        if not holdings.empty and val > 0:
            top = holdings.sort_values("_value", ascending=False).iloc[0]
            top_symbol = str(top["Symbol"])
            top_weight = top["_value"] / val

        rows.append({
            "Account": str(acct),
            "Value": val,
            "Cost Basis": basis,
            "Gain/Loss": gain,
            "Gain %": gain_pct,
            "Positions": int((group["_value"] > 0).sum()),
            "Top Holding": f"{top_symbol} ({top_weight:.0%})" if top_symbol else "—",
        })

    acct_df = pd.DataFrame(rows).sort_values("Value", ascending=False).reset_index(drop=True)

    acct_df_display = acct_df.copy()
    for col in ("Value", "Cost Basis", "Gain/Loss"):
        if col in acct_df_display.columns:
            acct_df_display[col] = acct_df_display[col].map(_fmt_compact)
    if "Gain %" in acct_df_display.columns:
        acct_df_display["Gain %"] = acct_df_display["Gain %"].map(lambda v: f"{v:+.1f}%")

    st.dataframe(acct_df_display, width="stretch", hide_index=True)


def _action_summary_placeholder() -> None:
    """Placeholder — phase 2 will fill these based on valuation + wonderfullness."""
    st.markdown("#### Action Summary")
    st.caption(
        "Valuation-based BUY / HOLD / SELL signals are coming in phase 2. "
        "This section will rank each holding by Wonderfulness and compare the "
        "current price against DCF / Lynch / Graham / PEG fair values."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 BUY candidates", "—")
    c2.metric("🟡 HOLD", "—")
    c3.metric("🔴 SELL / TRIM", "—")


def _positions_table(df: pd.DataFrame, signals: dict) -> None:
    st.markdown("#### All Positions")

    f1, f2, f3 = st.columns([2, 2, 3])
    with f1:
        account_filter = st.multiselect(
            "Account",
            options=sorted(df["Account name"].dropna().unique().tolist()),
            default=[],
            key="pos_account_filter",
        )
    with f2:
        size_filter = st.multiselect(
            "Position size",
            options=["<1%", "1–5%", "5–10%", ">10%"],
            default=[],
            key="pos_size_filter",
        )
    with f3:
        search = st.text_input("Search symbol", "", key="pos_search").strip().upper()

    view = df.copy()
    if account_filter:
        view = view[view["Account name"].isin(account_filter)]
    if size_filter:
        view = view[view["Size"].isin(size_filter)]
    if search:
        view = view[view["Symbol"].str.contains(search, na=False)]

    view = view[view["Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)]

    rows = []
    for _, r in view.iterrows():
        sym = str(r["Symbol"])
        sig = signals.get(sym, {}) or {}
        intrinsic = sig.get("intrinsic_value")
        price = r["Last price"]

        vs = None
        if intrinsic and price and price == price and intrinsic == intrinsic:
            vs = (price - intrinsic) / intrinsic

        rows.append({
            "Symbol": sym,
            "Account": r["Account name"],
            "Qty": r["Quantity"],
            "Value": r["_value"],
            "Gain %": r["Total gain/loss percent"],
            "Price": price,
            "Intrinsic": intrinsic,
            "vs Intrinsic": vs,
            "Method": sig.get("intrinsic_method") or "",
            "Rev 5y": sig.get("rev_5y"),
            "Rev 3y": sig.get("rev_3y"),
            "Rev 1y": sig.get("rev_1y"),
            "EPS 5y": sig.get("eps_5y"),
            "EPS 3y": sig.get("eps_3y"),
            "EPS 1y": sig.get("eps_1y"),
            "Wonder": sig.get("wonderfulness"),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        st.info("No positions to display.")
        return

    out = out.sort_values("Value", ascending=False).reset_index(drop=True)

    # ---- Build the styled dataframe: numeric values in, formatted display out ----
    styled = (
        out.style
        .format({
            "Qty": lambda v: f"{v:,.0f}" if v == v else "—",
            "Value": _money,
            "Gain %": _pct_num,
            "Price": _money,
            "Intrinsic": _money,
            "vs Intrinsic": lambda v: f"{v * 100:+.0f}%" if v == v else "—",
            "Rev 5y": _pct_frac,
            "Rev 3y": _pct_frac,
            "Rev 1y": _pct_frac,
            "EPS 5y": _pct_frac,
            "EPS 3y": _pct_frac,
            "EPS 1y": _pct_frac,
            "Wonder": lambda v: f"{v:.1f}/10" if v == v else "—",
        }, na_rep="—")
        .map(_color_vs_intrinsic, subset=["vs Intrinsic"])
        .map(_color_wonder, subset=["Wonder"])
    )

    st.dataframe(styled, width="stretch", hide_index=True)
    st.caption(
        f"Showing {len(out)} rows. "
        "Growth columns are CAGRs over each window. "
        "Wonder is the 0–10 quality score. "
        "Intrinsic is the median of valuation methods that succeeded — see Method. "
        "vs Intrinsic: green = trading below fair value, red = above."
    )


def _concentration_warnings(df: pd.DataFrame) -> None:
    st.markdown("#### Concentration Warnings")

    warnings: list[str] = []

    # 1. Single-name concentration
    over_10 = df[df["% of portfolio"] >= 0.10].sort_values("% of portfolio", ascending=False)
    for _, row in over_10.iterrows():
        warnings.append(
            f"**{row['Symbol']}** is {row['% of portfolio']:.1%} of your portfolio "
            f"({row['Account name']}) — over the 10% single-name threshold."
        )

    # 2. Account concentration
    acct_totals = df.groupby("Account name")["_value"].sum()
    grand_total = acct_totals.sum()
    if grand_total:
        for acct, val in acct_totals.items():
            pct = val / grand_total
            if pct >= 0.50:
                warnings.append(
                    f"**{acct}** holds {pct:.0%} of your total invested assets."
                )

    # 3. Losers over 10% of portfolio cost
    losers = df[
        (df["Total gain/loss percent"] < -20) & (df["% of portfolio"] > 0.005)
    ].sort_values("Total gain/loss percent")
    for _, row in losers.iterrows():
        warnings.append(
            f"**{row['Symbol']}** is down {row['Total gain/loss percent']:.0f}% "
            f"({row['% of portfolio']:.1%} of portfolio)."
        )

    if not warnings:
        st.success("No concentration warnings triggered.")
        return

    for w in warnings:
        st.warning(w)


def render_my_investments() -> None:
    st.header("📁 My Investments")
    st.caption(
        "Read-only view of your Fidelity positions. "
        "Drop the latest export into `data/fidelity_positions.csv` to refresh."
    )

    csv_path = _find_csv()
    if csv_path is None:
        st.warning(
            "No Fidelity positions CSV found. Export from Fidelity → "
            "Accounts & Trade → Portfolio → Positions → Download, then save it as "
            "`data/fidelity_positions.csv`."
        )
        return

    st.caption(
        f"Loaded: `{csv_path}` "
        f"(modified {pd.Timestamp(csv_path.stat().st_mtime, unit='s'):%Y-%m-%d %H:%M})"
    )

    try:
        raw = _parse_fidelity_csv(csv_path)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        return

    if raw.empty:
        st.warning("CSV parsed to zero rows.")
        return

    df = _derive(raw)

    _hero_strip(df)
    st.divider()

    _render_top_movers(df)          # <-- new Top Movers
    st.divider()

    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("🔄 Refresh analysis", key="refresh_signals"):
            Path("data/investments_signals.json").unlink(missing_ok=True)
            st.rerun()
    with c2:
        st.caption(
            "Refresh recomputes growth + valuations for every position. "
            "First run takes 1–3 minutes; results are cached."
        )

    with st.spinner("Loading signals…"):
        signals = _load_or_build_signals(df)

    _positions_table(df, signals)
    st.divider()
    _concentration_warnings(df)
    st.divider()
    _account_breakdown(df)

    st.caption(
        "Read-only snapshot. Not investment advice. "
        "Intrinsic values are model estimates, not predictions."
    )