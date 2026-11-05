# sleeves_backtest.py
"""
Additive-sleeve TQQQ/SQQQ strategy (alternative to the vote-averaging dial).

Each sleeve holds its own signed weight (+ = TQQQ, - = SQQQ). The net position is
the sum of the sleeves, clipped to [-1, 1], scaled by a volatility overlay and
moved toward the target by at most MAX_STEP per day.

Timing: a signal at the close of day t sets the position at that close (Mallik
trades in the last 10 minutes); the position earns day t+1's return.
--fill next delays that by one more day (approximation of a next-open fill).

Usage:
  python sleeves_backtest.py --project-dir C:\\path\\to\\your\\project
  python sleeves_backtest.py --project-dir ... --fill next --no-vol --short-scale 1.0
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime

import numpy as np
import pandas as pd

# Mallik's stated calendar-year results (from the interview); None = not stated
MALLIK_YEARLY = {2000: 200, 2001: 152, 2008: 83, 2009: 155, 2020: 469, 2021: 81}


@dataclass
class Params:
    # Sleeve capital shares (sum of positive trend shares = 1.0 -> 100% TQQQ in a clean bull)
    w_trend_regime: float = 0.80   # +1 above both MAs, -1 below both, +0.5 in between
    w_trend_250: float = 0.20      # sign of price vs 250d MA
    w_decel: float = 0.15          # trim when momentum is decelerating in a bull
    w_overext: float = 0.15        # trim when far above the 50d MA
    w_bounce: float = 0.15         # add on an oversold dip while above the 250d MA
    short_scale: float = 0.50      # scale applied to negative (SQQQ) positions
    # Sleeve thresholds
    decel_drop: float = 0.03       # fall in 10d ROC vs 5 days ago
    overext_dist: float = 0.10     # distance above 50d MA
    bounce_dist: float = -0.06     # distance below 50d MA
    # Overlay / execution
    use_vol: bool = True
    target_vol: float = 0.60       # annualized vol target for the 3x position
    max_step: float = 0.10         # max position change per day
    band: float = 0.0              # price must clear the 50d MA by this fraction to flip regime
    cost_bps: float = 5.0          # per unit of turnover
    enabled: tuple = ("trend_regime", "trend_250", "decel", "overext", "bounce")


SLEEVES = ("trend_regime", "trend_250", "decel", "overext", "bounce")


# ------------------------------------------------------------------ data
def load_data(start: str, end: str, use_real: bool):
    from data_provider import get_ticker
    from synthetic import synthetic_levered_series

    ndx = get_ticker("^NDX").history(start=start, end=end)
    ndx.index = pd.to_datetime(ndx.index).tz_localize(None)
    close = ndx["Close"].dropna()
    ret = close.pct_change().fillna(0.0)

    tq = synthetic_levered_series(ret, leverage=+3.0, expense_ratio=0.0095, financing_spread=0.02).pct_change().fillna(0.0)
    sq = synthetic_levered_series(ret, leverage=-3.0, expense_ratio=0.0095, financing_spread=0.02).pct_change().fillna(0.0)

    if use_real:  # splice actual ETF returns over the synthetic ones where available
        for tk, series in (("TQQQ", tq), ("SQQQ", sq)):
            df = get_ticker(tk).history(start=start, end=end)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            real = df["Close"].dropna().pct_change().dropna()
            series.loc[real.index.intersection(series.index)] = real
    return close, tq, sq


# ------------------------------------------------------------------ signals
def compute_signals(close: pd.Series) -> pd.DataFrame:
    s50 = close.rolling(50).mean()
    s250 = close.rolling(250).mean()
    roc10 = close.pct_change(10)
    return pd.DataFrame({
        "close": close,
        "s50": s50,
        "s250": s250,
        "dist50": close / s50 - 1.0,
        "accel": roc10 - roc10.shift(5),
        "rv": close.pct_change().rolling(20).std() * np.sqrt(252),
    })


def sleeve_weights(sig: pd.DataFrame, p: Params) -> pd.DataFrame:
    px, s50, s250 = sig["close"], sig["s50"], sig["s250"]
    bull = (px > s50 * (1 + p.band)) & (s50 > s250)
    bear = (px < s50 * (1 - p.band)) & (s50 < s250)

    regime = pd.Series(0.5, index=sig.index)
    regime[bull] = 1.0
    regime[bear] = -1.0
    above250 = np.where(px > s250, 1.0, -1.0)

    raw = pd.DataFrame({
        "trend_regime": p.w_trend_regime * regime,
        "trend_250": p.w_trend_250 * pd.Series(above250, index=sig.index),
        "decel": -p.w_decel * (bull & (sig["accel"] < -p.decel_drop)).astype(float),
        "overext": -p.w_overext * (sig["dist50"] > p.overext_dist).astype(float),
        "bounce": p.w_bounce * ((px > s250) & (sig["dist50"] < p.bounce_dist)).astype(float),
    })
    return raw.where(sig["s250"].notna())  # NaN until the 250d MA exists


def target_position(sig: pd.DataFrame, p: Params) -> pd.Series:
    w = sleeve_weights(sig, p)
    net = w[list(p.enabled)].sum(axis=1, min_count=1)
    net = net.where(net >= 0, net * p.short_scale).clip(-1.0, 1.0)
    if p.use_vol:
        scale = (p.target_vol / (sig["rv"] * 3.0)).clip(upper=1.0)
        net = net * scale.fillna(1.0)
    return net


def smooth(target: pd.Series, max_step: float) -> pd.Series:
    out = np.zeros(len(target))
    t = target.fillna(0.0).to_numpy()
    for i in range(1, len(t)):
        out[i] = out[i - 1] + np.clip(t[i] - out[i - 1], -max_step, max_step)
    return pd.Series(out, index=target.index)


# ------------------------------------------------------------------ backtest
def run(sig, tq, sq, p: Params, fill: str = "close") -> dict:
    target = target_position(sig, p)
    pos = smooth(target, p.max_step)
    pos[target.isna()] = 0.0
    lag = 1 if fill == "close" else 2
    held = pos.shift(lag).fillna(0.0)

    gross = np.where(held >= 0, held * tq.reindex(held.index), -held * sq.reindex(held.index))
    turnover = held.diff().abs().fillna(0.0)
    ret = pd.Series(gross, index=held.index).fillna(0.0) - turnover * p.cost_bps / 1e4
    ret = ret[target.first_valid_index():]
    return {"ret": ret, "pos": held.loc[ret.index], "turnover": turnover.loc[ret.index]}


def stats(ret: pd.Series, turnover: pd.Series | None = None) -> dict:
    if len(ret) < 20:
        return {}
    eq = (1 + ret).cumprod()
    years = (ret.index[-1] - ret.index[0]).days / 365.25
    vol = ret.std() * np.sqrt(252)
    out = {
        "Total%": (eq.iloc[-1] - 1) * 100,
        "CAGR%": (eq.iloc[-1] ** (1 / years) - 1) * 100,
        "MaxDD%": (eq / eq.cummax() - 1).min() * 100,
        "Vol%": vol * 100,
        "Sharpe": ret.mean() * 252 / vol if vol > 0 else 0.0,
    }
    if turnover is not None:
        out["Trades/wk"] = (turnover > 0.02).sum() / (years * 52)
    return out


def fmt(label: str, s: dict) -> str:
    if not s:
        return f"  {label:<26} (insufficient data)"
    tw = f" | trades/wk {s['Trades/wk']:.1f}" if "Trades/wk" in s else ""
    return (f"  {label:<26} Total {s['Total%']:>+14,.0f}% | CAGR {s['CAGR%']:>+7.1f}% | MaxDD {s['MaxDD%']:>7.1f}% | "
            f"Vol {s['Vol%']:>5.1f}% | Sharpe {s['Sharpe']:>5.2f}{tw}")


# ------------------------------------------------------------------ reports
def yearly_table(ret: pd.Series, bh3x: pd.Series, ndx: pd.Series) -> None:
    ys = (1 + ret).groupby(ret.index.year).prod() - 1
    b3 = (1 + bh3x).groupby(bh3x.index.year).prod() - 1
    bn = ndx.pct_change().fillna(0.0).pipe(lambda r: (1 + r).groupby(r.index.year).prod() - 1)
    print(f"\n{'Year':<6}{'Strategy':>11}{'3x B&H':>11}{'NDX':>9}{'Mallik':>9}")
    for y, v in ys.items():
        m = f"{MALLIK_YEARLY[y]:>+8d}%" if y in MALLIK_YEARLY else f"{'':>9}"
        print(f"{y:<6}{v*100:>+10.1f}%{b3.get(y, np.nan)*100:>+10.1f}%{bn.get(y, np.nan)*100:>+8.1f}%{m}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=None, help="folder containing data_provider.py and synthetic.py")
    ap.add_argument("--start", default="1985-01-01")
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--split", default=None, help="train ends here, test starts after (default: midpoint of the window)")
    ap.add_argument("--years", type=int, default=None, help="only evaluate the last N years (signals still warm up on full history)")
    ap.add_argument("--fill", choices=["close", "next"], default="close")
    ap.add_argument("--real", action="store_true", help="use actual TQQQ/SQQQ returns where available")
    ap.add_argument("--no-vol", action="store_true")
    ap.add_argument("--short-scale", type=float, default=0.5)
    ap.add_argument("--max-step", type=float, default=0.10)
    ap.add_argument("--target-vol", type=float, default=0.60)
    ap.add_argument("--band", type=float, default=0.0, help="e.g. 0.02 = price must be 2%% past the 50d MA")
    ap.add_argument("--sleeves", default=",".join(SLEEVES), help="comma list, e.g. trend_regime,trend_250")
    args = ap.parse_args()

    if args.project_dir:
        sys.path.insert(0, args.project_dir)

    close, tq, sq = load_data(args.start, args.end, args.real)
    sig = compute_signals(close)
    base = Params(use_vol=not args.no_vol, short_scale=args.short_scale, max_step=args.max_step,
                  target_vol=args.target_vol, band=args.band,
                  enabled=tuple(args.sleeves.split(",")))

    res = run(sig, tq, sq, base, args.fill)
    ret, turn = res["ret"], res["turnover"]
    if args.years:
        win_start = ret.index[-1] - pd.DateOffset(years=args.years)
        ret, turn = ret[ret.index > win_start], turn[turn.index > win_start]
    else:
        win_start = ret.index[0]
    split = pd.Timestamp(args.split) if args.split else ret.index[len(ret) // 2]

    print(f"\nSLEEVES  fill={args.fill}  vol_overlay={base.use_vol}  {ret.index[0].date()} -> {ret.index[-1].date()}")
    print(fmt("Full period", stats(ret, turn)))
    print(fmt(f"TRAIN (<= {split.date()})", stats(ret[ret.index <= split], turn[turn.index <= split])))
    print(fmt(f"TEST  (>  {split.date()})  <- trust this", stats(ret[ret.index > split], turn[turn.index > split])))
    print(fmt("3x B&H (same window)", stats(tq.loc[ret.index])))
    print(fmt("NDX B&H (same window)", stats(close.pct_change().loc[ret.index])))
    print("  Mallik reference: ~40% live CAGR / ~33% DD; Collective2 +20.7% / -37.6%; synthetic claim ~80%/yr")

    print("\nSLEEVE ATTRIBUTION (TRAIN period only; keep a sleeve only if it helps, then confirm on TEST)")
    test = lambda r: r[(r.index > win_start) & (r.index <= split)]
    for name in SLEEVES:
        alone = run(sig, tq, sq, replace(base, enabled=(name,)), args.fill)["ret"]
        drop = run(sig, tq, sq, replace(base, enabled=tuple(s for s in SLEEVES if s != name)), args.fill)["ret"]
        print(fmt(f"{name} alone", stats(test(alone))))
        print(fmt(f"all minus {name}", stats(test(drop))))

    yearly_table(ret, tq.loc[ret.index], close.loc[ret.index])


if __name__ == "__main__":
    main()
