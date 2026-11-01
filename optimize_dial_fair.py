"""
Fair dial optimizer: gives the 7-strategy dial its best shot, scored the same way as Sleeves.

- Votes/regime are computed once (vectorized); each config is a few array ops + one execute().
- Settings are chosen on TRAIN (synthetic 3x, start .. TRAIN_END) and reported on TEST
  (after TRAIN_END) and on REAL TQQQ/SQQQ prices (2010-02-11 ..). Sleeves are in the same table.

Usage: python optimize_dial_fair.py
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from backtest_40y import (build_strategy_targets, fetch_qqq, fetch_tqqq_sqqq,
                          synth_tqqq_sqqq_from_qqq)
from execute_backtest import execute
from sleeves_strategy import build_sleeve_targets

TRAIN_END = pd.Timestamp("2010-12-31")
REAL_START = pd.Timestamp("2010-02-11")
DEAD_ZONES = [0.00, 0.02, 0.05, 0.08]
HIGH_MULTS = [1.0, 1.5, 2.0, 3.0]
LOW_MULTS = [0.5, 1.0, 2.0, 3.0]
CASH = 10_000.0


def dial_inputs(close: pd.Series) -> pd.DataFrame:
    """Config-independent pieces: base dial, and whether the high-vol multiplier applies."""
    s50, s250 = close.rolling(50).mean(), close.rolling(250).mean()
    mid, sd = close.rolling(20).mean(), close.rolling(20).std()
    up, lo = mid + 2 * sd, mid - 2 * sd
    width = (up - lo) / mid
    rng = up - lo
    pos = ((close - lo) / rng * 100).where(rng != 0, 50.0)
    dist = (close - s50) / s50

    bull = (close > s50) & (s50 > s250)
    bear = (close < s50) & (s50 < s250)
    v = pd.DataFrame(index=close.index)
    v[1] = np.select([bull, bear], [1.0, 0.0], 0.5)
    v[2] = np.select([(dist < -0.08) & ~bear, dist > 0.10], [0.8, 0.2], 0.5)
    v[3] = np.select([pos < 15, pos > 85], [0.9, 0.1], 0.5)
    v[4] = np.where(bull & (width > 0.08), 0.9, 0.5)
    v[5] = np.where(bear, 0.0, 0.5)
    v[6] = np.where((dist > 0.12) & ~bull, 0.1, 0.5)
    v[7] = np.select([bear & (dist > -0.02), bear & (dist < -0.10)], [0.0, 0.3], 0.5)
    return pd.DataFrame({
        "dev": v.mean(axis=1) - 0.5,
        "hi": (bull | bear) & (width > 0.10),
    }).where(s250.notna()).dropna()


def dial_targets(inp: pd.DataFrame, dz: float, hi: float, lo: float) -> pd.DataFrame:
    mult = np.where(inp["hi"].astype(bool), hi, lo)
    dial = np.clip(0.5 + inp["dev"].to_numpy() * mult, 0.0, 1.0)
    tq = np.select([dial >= 0.5 + dz, dial <= 0.5 - dz], [dial, 0.0], 0.0)
    sq = np.select([dial >= 0.5 + dz, dial <= 0.5 - dz], [0.0, 1.0 - dial], 0.0)
    return pd.DataFrame({"tqqq": tq, "sqqq": sq, "cash": np.clip(1 - tq - sq, 0, None)}, index=inp.index)


def stats(eq: pd.Series) -> dict:
    if len(eq) < 50:
        return {"cagr": np.nan, "dd": np.nan, "sharpe": np.nan}
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    vol = r.std() * np.sqrt(252)
    return {"cagr": ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100,
            "dd": (eq / eq.cummax() - 1).min() * 100,
            "sharpe": r.mean() * 252 / vol if vol > 0 else 0.0}


def run(targets, tq, sq, start=None) -> pd.Series:
    t = targets if start is None else targets[targets.index >= start]
    return execute(t, tq, sq, starting_cash=CASH).equity


def score(label, tgt, s_tq, s_sq, r_tq, r_sq) -> dict:
    eq = run(tgt, s_tq, s_sq)
    train, test = eq[eq.index <= TRAIN_END], eq[eq.index > TRAIN_END]
    real = run(tgt, r_tq, r_sq, REAL_START)
    a, b, c = stats(train), stats(test), stats(real)
    return {"config": label,
            "TRAIN cagr": a["cagr"], "TRAIN dd": a["dd"], "TRAIN sharpe": a["sharpe"],
            "TEST cagr": b["cagr"], "TEST dd": b["dd"], "TEST sharpe": b["sharpe"],
            "REAL cagr": c["cagr"], "REAL dd": c["dd"], "REAL sharpe": c["sharpe"]}


def main() -> None:
    print("Fetching ^NDX and real TQQQ/SQQQ ...")
    ndx = fetch_qqq("1985-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    s_tq, s_sq = synth_tqqq_sqqq_from_qqq(ndx)
    r_tq, r_sq = fetch_tqqq_sqqq("2009-02-11", pd.Timestamp.today().strftime("%Y-%m-%d"))
    close = ndx["Close"]

    inp = dial_inputs(close)
    # Sanity: vectorized dial must reproduce the loop version in backtest_40y (dz=0, mult=1)
    ref = build_strategy_targets(ndx)
    mine = dial_targets(inp, 0.0, 1.0, 1.0).reindex(ref.index)
    err = (mine[["tqqq", "sqqq"]] - ref[["tqqq", "sqqq"]]).abs().max().max()
    print(f"Vectorized vs loop dial max abs diff: {err:.2e}  ({'OK' if err < 1e-9 else 'MISMATCH'})")
    if err >= 1e-9:
        return

    rows = []
    grid = list(itertools.product(DEAD_ZONES, HIGH_MULTS, LOW_MULTS))
    for i, (dz, hi, lo) in enumerate(grid, 1):
        rows.append(score(f"dz={dz:.2f} hi={hi:.1f} lo={lo:.1f}",
                          dial_targets(inp, dz, hi, lo), s_tq, s_sq, r_tq, r_sq))
        if i % 8 == 0:
            print(f"  {i}/{len(grid)} configs done", flush=True)
    df = pd.DataFrame(rows)

    live = score("LIVE app (dz=0.00 hi=1.2 lo=1.0)", dial_targets(inp, 0.0, 1.2, 1.0), s_tq, s_sq, r_tq, r_sq)
    sleeves = score("SLEEVES (long-or-cash)", build_sleeve_targets(close), s_tq, s_sq, r_tq, r_sq)

    # Pick on TRAIN only: require drawdown no worse than -70%, then best Sharpe.
    ok = df[df["TRAIN dd"] > -70]
    best = (ok if len(ok) else df).sort_values("TRAIN sharpe", ascending=False)
    top = best.head(5)

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:,.1f}" if abs(x) >= 10 else f"{x:,.2f}")
    print("\nTop 5 dial configs chosen on TRAIN Sharpe (dd > -70%), shown with out-of-sample TEST and REAL:")
    print(top.to_string(index=False))
    print("\nBaselines, same engine and windows:")
    print(pd.DataFrame([live, sleeves]).to_string(index=False))
    print("\nMedian of ALL dial configs (what a random pick from this grid gets):")
    print(df.drop(columns="config").median().round(2).to_string())
    df.to_csv("data/dial_optimization_fair.csv", index=False)
    print("\nFull grid saved to data/dial_optimization_fair.csv")


if __name__ == "__main__":
    main()
