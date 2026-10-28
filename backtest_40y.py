# backtest_40y.py
"""
Run the TQQQ/SQQQ strategy over real prices (2010+) or synthetic (1986+).

Modes:
  --mode real        Real TQQQ/SQQQ prices (2010-02-11 onward).
  --mode synthetic   Synthetic 3x from QQQ (1986 onward).
  --mode both        Run both, compare.

Usage:
  python backtest_40y.py --mode both
  python backtest_40y.py --mode real --start 2022-07-23 --end 2026-10-27
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from data_provider import get_ticker
from synthetic import synthetic_levered_series
from run_strategy_history import build_daily_targets
from execute_backtest import execute
from data_quality import check_data_quality
from mallik_sim import generate_mallik_signals

def fetch_qqq(start: str, end: str) -> pd.DataFrame:
    print(f"fetch_qqq - using ^NDX as the ticker and not QQQ for 40 years history")
    #df = get_ticker("QQQ").history(start=start, end=end)
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
    """Synthesize Open and Close for TQQQ/SQQQ from QQQ returns.

    Simplification: we don't know intraday path, so we set synthetic
    Open = previous Close for both legs. This avoids any same-day
    look-ahead but understates real intraday moves. Documented; not a bug.
    """
    qqq_ret = qqq["Close"].pct_change().fillna(0.0)

    tq_nav = synthetic_levered_series(qqq_ret, leverage=+3.0,
                                      expense_ratio=0.0095,
                                      financing_spread=financing_spread)
    sq_nav = synthetic_levered_series(qqq_ret, leverage=-3.0,
                                      expense_ratio=0.0095,
                                      financing_spread=financing_spread)

    def to_ohlc(nav: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({
            "Open": nav.shift(1).fillna(nav.iloc[0]),  # prev close as open
            "Close": nav,
        }, index=nav.index)

    return to_ohlc(tq_nav), to_ohlc(sq_nav)


def run_mode(label: str, qqq: pd.DataFrame, tqqq: pd.DataFrame, sqqq: pd.DataFrame,
             start_date: str | None, starting_cash: float) -> dict:
    
    # --- DATA QUALITY CHECK ---
    quality = check_data_quality(qqq)
    print(f"  [Data Quality] {quality['start_date']} to {quality['end_date']} ({quality['total_rows']} rows)")
    if quality["bad_price_count"] > 0:
        print(f"  ⚠️ WARNING: Found {quality['bad_price_count']} zero/negative prices.")
    if quality["jump_count"] > 0:
        print(f"  ⚠️ WARNING: Found {quality['jump_count']} single-day jumps >50% (possible unadjusted splits).")
    if not quality["is_healthy"]:
        print(f"  ⚠️ WARNING: Data may have gaps ({quality['missing_days']} missing business days).")
    else:
        print("  ✅ Data quality looks excellent.")
    # ---------------------------
    
    print(f"\n[{label}] building daily targets...")    
    # Build targets for the whole dataset (starts at day 200)
    targets = build_daily_targets(qqq, start_date=None)
    
    if targets.empty:
        print(f"[{label}] no targets produced — insufficient history")
        return {}
        
    # Slice targets and prices to only include the requested start date onwards
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
    else:
        start_ts = targets.index[0]  # Use earliest available date for synthetic runs
        
    targets_sliced = targets[targets.index >= start_ts]
    tqqq_sliced = tqqq[tqqq.index >= start_ts]
    sqqq_sliced = sqqq[sqqq.index >= start_ts]

    print(f"[{label}] {len(targets_sliced)} days, running execution...")
    result = execute(targets_sliced, tqqq_sliced, sqqq_sliced, starting_cash=starting_cash)
    # RETURN THE DATA SO THE SUMMARY CAN USE IT
    return {
        "label": label, 
        "targets": targets_sliced, 
        "result": result,
        "qqq_df": qqq,
        "tqqq_df": tqqq,
        "sqqq_df": sqqq
    }


def print_monthly_returns(equity: pd.Series, starting_cash: float, r: dict):
    """Print a 40-year comparison: MoatCheck vs Mallik Sim vs Buy & Hold."""
    from execute_backtest import execute
    from mallik_sim import generate_mallik_signals
    
    qqq_df = r["qqq_df"]
    tqqq_df = r["tqqq_df"]
    sqqq_df = r["sqqq_df"]
    
    # 1. Run the Mallik Simulation
    mallik_eq = None
    if qqq_df is not None:
        mallik_targets = generate_mallik_signals(qqq_df)
        
        # Align dates
        common_dates = mallik_targets.index.intersection(tqqq_df.index)
        mallik_targets = mallik_targets.loc[common_dates]
        
        # Run Mallik Sim through the real execution engine using real synthetic prices
        res = execute(
            mallik_targets, 
            tqqq_df.loc[common_dates], 
            sqqq_df.loc[common_dates], 
            starting_cash=starting_cash
        )
        mallik_eq = res.equity

    # 2. Format our Equity
    start_date = equity.index[0] - pd.Timedelta(1, unit='D')
    eq_with_start = pd.concat([pd.Series([starting_cash], index=[start_date]), equity])
    monthly_eq = eq_with_start.resample('ME').last()
    monthly_returns = monthly_eq.pct_change().dropna()
    
    # 3. Format Mallik's Equity
    if mallik_eq is not None:
        mallik_with_start = pd.concat([pd.Series([starting_cash], index=[start_date]), mallik_eq])
        monthly_mallik = mallik_with_start.resample('ME').last().pct_change().dropna()
    else:
        monthly_mallik = None

    # 4. Format Buy & Hold
    monthly_bh = None
    if tqqq_df is not None:
        tqqq_px = tqqq_df["Close"]
        bh_with_start = tqqq_px.reindex([start_date]).ffill().fillna(100.0).iloc[0]
        bh_series = pd.concat([pd.Series([bh_with_start], index=[start_date]), tqqq_px])
        monthly_bh = bh_series.resample('ME').last().pct_change().dropna()
    
    print(f"\n{'=' * 95}")
    print(f"MONTHLY RETURNS: MOATCHECK vs MALLIK SIM vs BUY & HOLD")
    print(f"{'=' * 95}")
    
    print(f"{'Month':<10} | {'MoatCheck':<10} | {'Mallik Sim':<11} | {'Buy&Hold':<10} | {'MC vs Mallik':<12} | {'MC vs BH':<10}")
    print("-" * 90)
    
    for date, ret in monthly_returns.items():
        month_str = date.strftime("%Y-%m")
        our_ret = ret * 100
        
        # Mallik Sim
        m_ret = monthly_mallik.get(date) * 100 if monthly_mallik is not None and date in monthly_mallik.index else 0.0
        diff_m = our_ret - m_ret
        
        # Buy & Hold
        bh_ret = monthly_bh.get(date) * 100 if monthly_bh is not None and date in monthly_bh.index else 0.0
        diff_bh = our_ret - bh_ret
        
        print(f"{month_str:<10} | {our_ret:>+9.2f}% | {m_ret:>+10.2f}% | {bh_ret:>+9.2f}% | {diff_m:>+11.2f}% | {diff_bh:>+9.2f}%")
            
    print("-" * 90)
    
    import math
    our_total = (equity.iloc[-1] / starting_cash - 1.0) * 100
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    our_cagr = ((equity.iloc[-1] / starting_cash) ** (1 / years) - 1) * 100
    
    if mallik_eq is not None:
        mallik_total = (mallik_eq.iloc[-1] / starting_cash - 1.0) * 100
        mallik_cagr = ((mallik_eq.iloc[-1] / starting_cash) ** (1 / years) - 1) * 100
    else:
        mallik_total, mallik_cagr = 0.0, 0.0

    if monthly_bh is not None and not monthly_bh.empty:
        bh_total = (math.prod(1 + r for r in monthly_bh) - 1) * 100
        bh_cagr = ((1 + bh_total/100) ** (1 / years) - 1) * 100
    else:
        bh_total, bh_cagr = 0.0, 0.0
        
    print(f"{'TOTAL':<10} | {our_total:>+9.2f}% | {mallik_total:>+10.2f}% | {bh_total:>+9.2f}%")
    print(f"{'CAGR':<10} | {our_cagr:>+9.2f}% | {mallik_cagr:>+10.2f}% | {bh_cagr:>+9.2f}%")

def print_summary(r: dict, starting_cash: float = 10000.0, tqqq_px: pd.Series = None) -> None:
    if not r:
        return
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
    
    # Print the new monthly returns table
    print_monthly_returns(r["result"].equity, starting_cash, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "synthetic", "both"], default="both")
    ap.add_argument("--start", default=None,
                    help="Real mode only; default = 2010-02-11 (TQQQ inception)")
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--starting-cash", type=float, default=10_000.0)
    ap.add_argument("--financing-spread", type=float, default=0.02,
                    help="Synthetic-mode annual financing cost, e.g. 0.02 = 2%%")
    ap.add_argument("--out-prefix", default="backtest40y")
    args = ap.parse_args()

    if args.mode in ("real", "both"):
        start_str = args.start or "2010-02-11"
        # Fetch 1 year prior to the requested start date to allow 200-day SMA warmup
        fetch_start = (pd.to_datetime(start_str) - pd.Timedelta(365, unit='D')).strftime("%Y-%m-%d")
        
        print(f"[real] fetching QQQ/TQQQ/SQQQ {fetch_start} → {args.end} (warmup included)")
        qqq = fetch_qqq(fetch_start, args.end)
        tqqq, sqqq = fetch_tqqq_sqqq(fetch_start, args.end)
        r = run_mode(f"REAL prices {start_str} → {args.end}", qqq, tqqq, sqqq,
                     start_date=start_str, starting_cash=args.starting_cash)
        
        print_summary(r, args.starting_cash, tqqq_px=tqqq["Close"])
        if r:
            r["result"].equity.to_csv(f"{args.out_prefix}_real_equity.csv")
            r["result"].trades.to_csv(f"{args.out_prefix}_real_trades.csv", index=False)
            pd.DataFrame({
                "tqqq": r["result"].tqqq_weight,
                "sqqq": r["result"].sqqq_weight,
                "cash": r["result"].cash_weight,
            }).to_csv(f"{args.out_prefix}_real_weights.csv")

    if args.mode in ("synthetic", "both"):
        start = args.start or "1986-01-01"
        print(f"\n[synthetic] fetching QQQ {start} → {args.end} (QQQ inception 1999; "
              f"pre-1999 the synthetic run will start where data begins)")
        qqq = fetch_qqq(start, args.end)
        tqqq, sqqq = synth_tqqq_sqqq_from_qqq(qqq, financing_spread=args.financing_spread)
        r = run_mode(f"SYNTHETIC 3x {qqq.index[0].date()} → {qqq.index[-1].date()}",
                     qqq, tqqq, sqqq,
                     start_date=None, starting_cash=args.starting_cash)
        
        print_summary(r, args.starting_cash, tqqq_px=tqqq["Close"])
        
        if r:
            r["result"].equity.to_csv(f"{args.out_prefix}_synth_equity.csv")
            r["result"].trades.to_csv(f"{args.out_prefix}_synth_trades.csv", index=False)
            pd.DataFrame({
                "tqqq": r["result"].tqqq_weight,
                "sqqq": r["result"].sqqq_weight,
                "cash": r["result"].cash_weight,
            }).to_csv(f"{args.out_prefix}_synth_weights.csv")

    print(f"\n{'=' * 60}")
    print("VALIDATION TARGET — Mallik's Collective2 (2022-07-23 → today)")
    print(f"{'=' * 60}")
    print("  CAGR:                 +22.2%")
    print("  Max drawdown:         -37.58%")
    print("  Win rate:             39.2%")
    print("  Num trades:           102")
    print("  Avg trade duration:   13.2 days")
    print()
    print("If REAL mode's CAGR is wildly different (e.g., +80% vs +22%),")
    print("our implementation is NOT Mallik's strategy. If it's within ±10%,")
    print("the code faithfully implements something in his class.")


if __name__ == "__main__":
    main()
