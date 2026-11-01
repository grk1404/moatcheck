# backtest_40y.py
"""
Standalone 40-Year Backtest Engine.
Strategy logic is defined locally in this file so you can tweak it and run 
simulations without affecting your live Streamlit app.

Usage:
  python backtest_40y.py --mode synthetic
  python backtest_40y.py --mode real --start 2022-07-23
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd
import numpy as np

from data_provider import get_ticker
from synthetic import synthetic_levered_series
from execute_backtest import execute
from data_quality import check_data_quality

# ============================================================
# 1. STANDALONE STRATEGY LOGIC (Tweak these values!)
# ============================================================
NEUTRAL = 0.5
REGIME_LONG_MA = 250

# --- BACKTEST GLOBALS ---
DEAD_ZONE = 0.00
HIGH_VOL_MULT = 1.0
LOW_VOL_MULT = 1.0

def dial_to_allocation(dial: float) -> dict:
    dial = max(0.0, min(1.0, dial))
    
    if dial >= 0.50 + DEAD_ZONE:
        tqqq, sqqq = dial, 0.0
    elif dial <= 0.50 - DEAD_ZONE:
        tqqq, sqqq = 0.0, 1.0 - dial
    else:
        tqqq, sqqq = 0.0, 0.0
        
    cash = max(0.0, 1.0 - tqqq - sqqq)
    return {"tqqq": tqqq, "sqqq": sqqq, "cash": cash}


def build_strategy_targets(qqq_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily targets by running the 7 strategies."""
    close = qqq_df["Close"]
    
    # Calculate indicators
    sma_50 = close.rolling(50).mean()
    sma_250 = close.rolling(250).mean()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    rows = []
    for d in close.index:
        if pd.isna(sma_250.loc[d]): continue
            
        price = close.loc[d]
        s50 = sma_50.loc[d]
        s250 = sma_250.loc[d]
        b_upper = bb_upper.loc[d]
        b_lower = bb_lower.loc[d]
        b_mid = bb_mid.loc[d]
        
        bb_width = (b_upper - b_lower) / b_mid if b_mid else 0.0
        bb_pos = ((price - b_lower) / (b_upper - b_lower) * 100) if (b_upper - b_lower) else 50.0
        dist_50 = (price - s50) / s50 if s50 else 0.0
        
        if price > s50 > s250: regime = "BULL"
        elif price < s50 < s250: regime = "BEAR"
        else: regime = "NEUTRAL"

        votes = []
        # 1. Momentum Long
        if regime == "BULL": votes.append(1.0)
        elif regime == "BEAR": votes.append(0.0)
        else: votes.append(NEUTRAL)
        # 2. Mean Reversion
        if dist_50 < -0.08 and regime != "BEAR": votes.append(0.8)
        elif dist_50 > 0.10: votes.append(0.2)
        else: votes.append(NEUTRAL)
        # 3. BB Reversion
        if bb_pos < 15: votes.append(0.9)
        elif bb_pos > 85: votes.append(0.1)
        else: votes.append(NEUTRAL)
        # 4. BB Width
        if regime == "BULL" and bb_width > 0.08: votes.append(0.9)
        else: votes.append(NEUTRAL)
        # 5. Momentum Short
        if regime == "BEAR": votes.append(0.0)
        else: votes.append(NEUTRAL)
        # 6. Short Reversion
        if dist_50 > 0.12 and regime != "BULL": votes.append(0.1)
        else: votes.append(NEUTRAL)
        # 7. Bear Bounce Fade
        if regime == "BEAR" and dist_50 > -0.02: votes.append(0.0)
        elif regime == "BEAR" and dist_50 < -0.10: votes.append(0.3)
        else: votes.append(NEUTRAL)

        base_dial = sum(votes) / len(votes)
        deviation = base_dial - 0.50
        multiplier = HIGH_VOL_MULT if (regime in ("BULL", "BEAR") and bb_width > 0.10) else LOW_VOL_MULT
        
        dial = max(0.0, min(1.0, 0.50 + (deviation * multiplier)))
        alloc = dial_to_allocation(dial)
        rows.append({"date": d, "tqqq": alloc["tqqq"], "sqqq": alloc["sqqq"], "cash": alloc["cash"]})
        
    return pd.DataFrame(rows).set_index("date")

def generate_mallik_signals(qqq_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate Mallik's continuous sizing formula."""
    close = qqq_df["Close"]
    sma_50 = close.rolling(50).mean()
    sma_250 = close.rolling(250).mean()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    
    rows = []
    for d in close.index:
        if pd.isna(sma_250.loc[d]): continue
        
        price = close.loc[d]
        s50 = sma_50.loc[d]
        s250 = sma_250.loc[d]
        b_mid = bb_mid.loc[d]
        b_std = bb_std.loc[d]
        
        bb_width = (4 * b_std) / b_mid if b_mid else 0.0
        dist_250 = (price - s250) / s250
        
        base_target = max(0.0, min(1.0, 0.5 + dist_250 * 5.0))
        
        if price > s50 > s250: target = base_target
        elif price < s50 < s250: target = base_target * 0.5
        else: target = 0.5
            
        if bb_width > 0.10:
            target = 0.5 + (target - 0.5) * 1.2
            
        target = max(0.0, min(1.0, target))
        
        if target >= 0.5:
            tq, sq = target, 0.0
        else:
            tq, sq = 0.0, 1.0 - target
            
        rows.append({"date": d, "tqqq": tq, "sqqq": sq, "cash": max(0.0, 1.0 - tq - sq)})
        
    return pd.DataFrame(rows).set_index("date")


# ============================================================
# 2. DATA FETCHING
# ============================================================
def fetch_qqq(start: str, end: str) -> pd.DataFrame:
    print(f"fetch_qqq - using ^NDX as the ticker and not QQQ for 40 years history")
    df = get_ticker("^NDX").history(start=start, end=end)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna(subset=["Close"])

def fetch_tqqq_sqqq(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = {}
    for tk in ("TQQQ", "SQQQ"):
        df = get_ticker(tk).history(start=start, end=end)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        out[tk] = df.dropna(subset=["Close"])
    return out["TQQQ"], out["SQQQ"]

def synth_tqqq_sqqq_from_qqq(qqq: pd.DataFrame, financing_spread: float = 0.02) -> tuple[pd.DataFrame, pd.DataFrame]:
    qqq_ret = qqq["Close"].pct_change().fillna(0.0)
    tq_nav = synthetic_levered_series(qqq_ret, leverage=+3.0, expense_ratio=0.0095, financing_spread=financing_spread)
    sq_nav = synthetic_levered_series(qqq_ret, leverage=-3.0, expense_ratio=0.0095, financing_spread=financing_spread)

    def to_ohlc(nav: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({"Open": nav.shift(1).fillna(nav.iloc[0]), "Close": nav}, index=nav.index)

    return to_ohlc(tq_nav), to_ohlc(sq_nav)

# ============================================================
# 3. EXECUTION & REPORTING
# ============================================================
def run_mode(label: str, qqq: pd.DataFrame, tqqq: pd.DataFrame, sqqq: pd.DataFrame,
             start_date: str | None, starting_cash: float) -> dict:
    
    quality = check_data_quality(qqq)
    print(f"  [Data Quality] {quality['start_date']} to {quality['end_date']} ({quality['total_rows']} rows)")
    if quality["bad_price_count"] > 0: print(f"  ⚠️ WARNING: Found {quality['bad_price_count']} zero/negative prices.")
    if quality["jump_count"] > 0: print(f"  ⚠️ WARNING: Found {quality['jump_count']} single-day jumps >50%.")
    if quality["is_healthy"]: print("  ✅ Data quality looks excellent.")
    
    print(f"\n[{label}] building daily targets...")    
    targets = build_strategy_targets(qqq)
    
    if targets.empty:
        print(f"[{label}] no targets produced — insufficient history")
        return {}
        
    start_ts = pd.to_datetime(start_date) if start_date else targets.index[0]
        
    targets_sliced = targets[targets.index >= start_ts]
    tqqq_sliced = tqqq[tqqq.index >= start_ts]
    sqqq_sliced = sqqq[sqqq.index >= start_ts]

    print(f"[{label}] {len(targets_sliced)} days, running execution...")
    result = execute(targets_sliced, tqqq_sliced, sqqq_sliced, starting_cash=starting_cash)
    
    return {
        "label": label, "targets": targets_sliced, "result": result,
        "qqq_df": qqq, "tqqq_df": tqqq, "sqqq_df": sqqq
    }

def _equity_from_targets(targets: pd.DataFrame, start_ts: pd.Timestamp, r: dict, starting_cash: float) -> pd.Series:
    """Run a target frame through the same execution engine, from start_ts on."""
    t = targets[targets.index >= start_ts]
    return execute(t, r["tqqq_df"], r["sqqq_df"], starting_cash=starting_cash).equity


def print_monthly_returns(equity: pd.Series, starting_cash: float, r: dict):
    from sleeves_strategy import build_sleeve_targets

    start_ts = equity.index[0]
    base_date = start_ts - pd.offsets.MonthEnd(1)  # prior month-end, so the first month is not dropped

    # All strategies use the same window, data and execution engine.
    mallik_eq = _equity_from_targets(generate_mallik_signals(r["qqq_df"]), start_ts, r, starting_cash)
    sleeve_eq = _equity_from_targets(build_sleeve_targets(r["qqq_df"]["Close"]), start_ts, r, starting_cash)

    bh_px = r["tqqq_df"]["Close"]
    bh_px = bh_px[bh_px.index >= start_ts]

    curves = {
        "MoatCheck": (equity, starting_cash),
        "Mallik Sim": (mallik_eq, starting_cash),
        "Sleeves": (sleeve_eq, starting_cash),
        "Buy&Hold": (bh_px, float(bh_px.iloc[0])),
    }
    monthly = {}
    for name, (curve, base) in curves.items():
        with_base = pd.concat([pd.Series([base], index=[base_date]), curve])
        monthly[name] = with_base.resample('ME').last().pct_change().dropna()

    names = list(curves)
    width = 12 + 14 * len(names)
    print(f"\n{'=' * width}\nMONTHLY RETURNS: MOATCHECK vs MALLIK SIM vs SLEEVES vs BUY & HOLD\n{'=' * width}")
    print(f"{'Month':<9}" + "".join(f"| {n:>11} " for n in names))
    print("-" * width)

    for date in monthly["MoatCheck"].index:
        cells = []
        for n in names:
            v = monthly[n].get(date)
            cells.append(f"| {v * 100:>+10.2f}% " if v is not None else f"| {'n/a':>11} ")
        print(f"{date.strftime('%Y-%m'):<9}" + "".join(cells))

    print("-" * width)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    totals, cagrs, dds = [], [], []
    for n in names:
        curve, base = curves[n]
        growth = curve.iloc[-1] / base
        totals.append(f"| {(growth - 1) * 100:>+10.2f}% ")
        cagrs.append(f"| {(growth ** (1 / years) - 1) * 100:>+10.2f}% ")
        dds.append(f"| {(curve / curve.cummax() - 1).min() * 100:>+10.2f}% ")
    print(f"{'TOTAL':<9}" + "".join(totals))
    print(f"{'CAGR':<9}" + "".join(cagrs))
    print(f"{'MAX DD':<9}" + "".join(dds))

def print_summary(r: dict, starting_cash: float = 10000.0) -> None:
    if not r: return
    d = r["result"].diagnostics
    print(f"\n{'=' * 60}\n{r['label']}\n{'=' * 60}")
    print(f"  CAGR:                 {d['cagr']*100:+.2f}%")
    print(f"  Volatility:           {d['vol']*100:.2f}%")
    print(f"  Sharpe:               {d['sharpe']:.2f}")
    print(f"  Max drawdown:         {d['max_dd']*100:.2f}%")
    print(f"  Num trades:           {d['num_trades']}")
    print(f"  Round-trip win rate:  {d['win_rate']*100:.1f}%")
    print(f"  Avg trade duration:   {d['avg_duration_days']:.1f} days")
    print(f"  Avg TQQQ weight:      {d['avg_tqqq_weight']*100:.1f}%")
    print(f"  Avg SQQQ weight:      {d['avg_sqqq_weight']*100:.1f}%")
    print(f"  Avg cash weight:      {d['avg_cash_weight']*100:.1f}%")
    print(f"  Days mostly in cash:  {d['pct_days_mostly_cash']*100:.1f}%")
    print(f"  Days short-heavy:     {d['pct_days_short_heavy']*100:.1f}%")
    print(f"  Total cost paid:      ${d['total_cost_paid']:,.2f}")
    
    print_monthly_returns(r["result"].equity, starting_cash, r)


# ============================================================
# 4. CLI MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "synthetic", "both"], default="both")
    ap.add_argument("--start", default=None, help="Real mode only; default = 2010-02-11")
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--starting-cash", type=float, default=10_000.0)
    ap.add_argument("--financing-spread", type=float, default=0.02)
    ap.add_argument("--out-prefix", default="backtest40y")
    args = ap.parse_args()

    if args.mode in ("real", "both"):
        start_str = args.start or "2010-02-11"
        fetch_start = (pd.to_datetime(start_str) - pd.Timedelta(365, unit='D')).strftime("%Y-%m-%d")
        
        print(f"[real] fetching QQQ/TQQQ/SQQQ {fetch_start} → {args.end} (warmup included)")
        qqq = fetch_qqq(fetch_start, args.end)
        tqqq, sqqq = fetch_tqqq_sqqq(fetch_start, args.end)
        r = run_mode(f"REAL prices {start_str} → {args.end}", qqq, tqqq, sqqq, start_date=start_str, starting_cash=args.starting_cash)
        
        print_summary(r, args.starting_cash)
        if r:
            r["result"].equity.to_csv(f"{args.out_prefix}_real_equity.csv")
            r["result"].trades.to_csv(f"{args.out_prefix}_real_trades.csv", index=False)
            pd.DataFrame({"tqqq": r["result"].tqqq_weight, "sqqq": r["result"].sqqq_weight, "cash": r["result"].cash_weight}).to_csv(f"{args.out_prefix}_real_weights.csv")

    if args.mode in ("synthetic", "both"):
        start = args.start or "1986-01-01"
        print(f"\n[synthetic] fetching QQQ {start} → {args.end} (QQQ inception 1999)")
        fetch_start = (pd.to_datetime(start) - pd.Timedelta(365, unit='D')).strftime("%Y-%m-%d") if args.start else start
        qqq = fetch_qqq(fetch_start, args.end)
        tqqq, sqqq = synth_tqqq_sqqq_from_qqq(qqq, financing_spread=args.financing_spread)
        r = run_mode(f"SYNTHETIC 3x {args.start or qqq.index[0].date()} → {qqq.index[-1].date()}", qqq, tqqq, sqqq, start_date=args.start, starting_cash=args.starting_cash)
        
        print_summary(r, args.starting_cash)
        if r:
            r["result"].equity.to_csv(f"{args.out_prefix}_synth_equity.csv")
            r["result"].trades.to_csv(f"{args.out_prefix}_synth_trades.csv", index=False)
            pd.DataFrame({"tqqq": r["result"].tqqq_weight, "sqqq": r["result"].sqqq_weight, "cash": r["result"].cash_weight}).to_csv(f"{args.out_prefix}_synth_weights.csv")

    print(f"\n{'=' * 60}\nVALIDATION TARGET — Mallik's Collective2 (2022-07-23 → today)\n{'=' * 60}")
    print("  CAGR:                 +20.7%\n  Max drawdown:         -37.58%\n  Win rate:             38.2%  (102 trades, avg 13.2 days)")
    print("\nIf REAL mode's CAGR is wildly different (e.g., +80% vs +22%),")
    print("our implementation is NOT Mallik's strategy.")

if __name__ == "__main__":
    main()