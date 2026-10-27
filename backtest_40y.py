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


def fetch_qqq(start: str, end: str) -> pd.DataFrame:
    df = get_ticker("QQQ").history(start=start, end=end)
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
    print(f"\n[{label}] building daily targets...")
    # Build targets for the whole dataset (starts at day 200)
    targets = build_daily_targets(qqq, start_date=None)
    
    if targets.empty:
        print(f"[{label}] no targets produced — insufficient history")
        return {}
        
    # Slice targets and prices to only include the requested start date onwards
    start_ts = pd.to_datetime(start_date)
    targets_sliced = targets[targets.index >= start_ts]
    tqqq_sliced = tqqq[tqqq.index >= start_ts]
    sqqq_sliced = sqqq[sqqq.index >= start_ts]

    print(f"[{label}] {len(targets_sliced)} days, running execution...")
    result = execute(targets_sliced, tqqq_sliced, sqqq_sliced, starting_cash=starting_cash)
    return {"label": label, "targets": targets_sliced, "result": result}


def print_monthly_returns(equity: pd.Series, starting_cash: float):
    """Print a month-by-month breakdown comparing against Mallik's C2 page."""
    # Resample to month-end. Prepend starting cash to capture month 1 correctly.
    start_date = equity.index[0] - pd.Timedelta(1, unit='D')
    eq_with_start = pd.concat([pd.Series([starting_cash], index=[start_date]), equity])
    
    # Get the last value of each month
    monthly_eq = eq_with_start.resample('ME').last()
    
    # Calculate percent change
    monthly_returns = monthly_eq.pct_change().dropna()
    
    print(f"\n{'=' * 75}")
    print(f"MONTHLY RETURNS: OUR CODE vs MALLIK (COLLECTIVE2)")
    print(f"{'=' * 75}")
    
    # Mallik's exact Collective2 returns from July 2022 to Oct 2026
    mallik_c2 = {
        (2022, 7): 7.5,   (2022, 8): 5.5,   (2022, 9): -16.2, (2022, 10): -1.0,
        (2022, 11): 13.2, (2022, 12): -16.4,
        (2023, 1): 12.8,  (2023, 2): -5.1,  (2023, 3): 7.3,   (2023, 4): -0.9,
        (2023, 5): 23.8,  (2023, 6): 12.1,  (2023, 7): 10.1,  (2023, 8): -1.8,
        (2023, 9): -6.1,  (2023, 10): -2.9, (2023, 11): 9.7,  (2023, 12): 3.6,
        (2024, 1): 7.3,   (2024, 2): 4.5,   (2024, 3): -0.6,  (2024, 4): -8.6,
        (2024, 5): 19.2,  (2024, 6): 1.7,   (2024, 7): -3.3,  (2024, 8): 1.0,
        (2024, 9): -4.0,  (2024, 10): -0.7, (2024, 11): -9.1, (2024, 12): 5.2,
        (2025, 1): 2.8,   (2025, 2): -3.5,  (2025, 3): -21.2, (2025, 4): 2.6,
        (2025, 5): 17.7,  (2025, 6): 21.2,  (2025, 7): 5.4,   (2025, 8): 4.0,
        (2025, 9): 11.5,  (2025, 10): -0.7, (2025, 11): 1.4,  (2025, 12): -1.8,
        (2026, 1): -4.9,  (2026, 2): -4.9,  (2026, 3): -1.6,  (2026, 4): 23.8,
        (2026, 5): 10.4,  (2026, 6): -20.9, (2026, 7): -5.2,  (2026, 8): 2.6,
        (2026, 9): 7.8,   (2026, 10): -2.8,
    }
    
    print(f"{'Month':<10} | {'Our Backtest':<12} | {'Mallik (C2)':<12} | {'Diff':<10}")
    print("-" * 55)
    
    for date, ret in monthly_returns.items():
        month_str = date.strftime("%Y-%m")
        our_ret = ret * 100
        
        m_ret = mallik_c2.get((date.year, date.month))
        if m_ret is not None:
            diff = our_ret - m_ret
            print(f"{month_str:<10} | {our_ret:>+10.2f}% | {m_ret:>+10.2f}% | {diff:>+8.2f}%")
        else:
            print(f"{month_str:<10} | {our_ret:>+10.2f}% | {'N/A':>10} | {'N/A':>8}")
            
    print("-" * 55)
    import math
    
    # 1. Calculate Mallik's Total Return by compounding his monthly returns
    mallik_total = (math.prod(1 + (v / 100) for v in mallik_c2.values()) - 1) * 100
    
    # 2. Calculate Our Total Return
    our_total = (equity.iloc[-1] / starting_cash - 1.0) * 100
    diff_total = our_total - mallik_total

    # 3. Calculate Our CAGR
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    our_cagr = ((equity.iloc[-1] / starting_cash) ** (1 / years) - 1) * 100
    
    # 4. Mallik's CAGR (from his Collective2 page)
    mallik_cagr = 22.2
    diff_cagr = our_cagr - mallik_cagr

    print("-" * 55)
    # Print the Total Row
    print(f"{'TOTAL':<10} | {our_total:>+10.2f}% | {mallik_total:>+10.2f}% | {diff_total:>+8.2f}%")
    # Print the CAGR Row
    print(f"{'CAGR':<10} | {our_cagr:>+10.2f}% | {mallik_cagr:>+10.2f}% | {diff_cagr:>+8.2f}%")


def print_summary(r: dict, starting_cash: float = 10000.0) -> None:
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
    print_monthly_returns(r["result"].equity, starting_cash)


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
        
        print_summary(r, args.starting_cash)
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
        print_summary(r)
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
