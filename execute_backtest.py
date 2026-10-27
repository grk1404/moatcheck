# execute_backtest.py
"""
Execution engine for the 3-way {TQQQ, SQQQ, CASH} strategy.

Key properties:
  - Decision at close of day t; execution at open of day t+1.
  - Positions marked to close each day.
  - Realistic cost per side (default 5 bps).
  - Optional dividend/distribution ignored (TQQQ/SQQQ don't pay meaningfully).
  - Tracks TQQQ weight, SQQQ weight, cash weight over time for diagnostic use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StrategyBacktestResult:
    equity: pd.Series
    tqqq_weight: pd.Series
    sqqq_weight: pd.Series
    cash_weight: pd.Series
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict = field(default_factory=dict)


def execute(
    targets: pd.DataFrame,        # columns: tqqq, sqqq, cash (fractions); index = date
    tqqq_px: pd.DataFrame,        # must have 'Open' and 'Close'
    sqqq_px: pd.DataFrame,
    starting_cash: float = 10_000.0,
    cost_bps_per_side: float = 5.0,
    min_trade_pct: float = 0.02,
) -> StrategyBacktestResult:
    """Run the strategy and return daily equity + weight diagnostics."""
    idx = targets.index.intersection(tqqq_px.index).intersection(sqqq_px.index)
    targets = targets.loc[idx]
    tqqq_px = tqqq_px.loc[idx]
    sqqq_px = sqqq_px.loc[idx]

    n = len(idx)
    cash = float(starting_cash)
    tq_sh = 0.0
    sq_sh = 0.0

    equity = np.zeros(n)
    w_tq = np.zeros(n)
    w_sq = np.zeros(n)
    w_cash = np.zeros(n)
    trade_rows: list[dict] = []

    cost_frac = cost_bps_per_side / 10_000.0

    for i in range(n):
        d = idx[i]
        # --- Morning: execute yesterday's target at today's open ---
        if i > 0:
            prev = targets.iloc[i - 1]
            tq_open = float(tqqq_px["Open"].iloc[i])
            sq_open = float(sqqq_px["Open"].iloc[i])

            # Mark to open to get NAV used for sizing
            tq_val = tq_sh * tq_open
            sq_val = sq_sh * sq_open
            nav = tq_val + sq_val + cash
            if nav > 0 and tq_open > 0 and sq_open > 0:
                tgt_tq_val = prev["tqqq"] * nav
                tgt_sq_val = prev["sqqq"] * nav
                dq = tgt_tq_val - tq_val
                ds = tgt_sq_val - sq_val

                if abs(dq) / nav >= min_trade_pct or abs(ds) / nav >= min_trade_pct:
                    if abs(dq) / nav >= min_trade_pct:
                        sh = dq / tq_open
                        cost = abs(dq) * cost_frac
                        tq_sh += sh
                        cash -= dq + cost
                        trade_rows.append({
                            "date": d, "ticker": "TQQQ", "shares": sh,
                            "price": tq_open, "dollar": dq, "cost": cost,
                        })
                    if abs(ds) / nav >= min_trade_pct:
                        sh = ds / sq_open
                        cost = abs(ds) * cost_frac
                        sq_sh += sh
                        cash -= ds + cost
                        trade_rows.append({
                            "date": d, "ticker": "SQQQ", "shares": sh,
                            "price": sq_open, "dollar": ds, "cost": cost,
                        })

        # --- End of day: mark to close ---
        tq_close = float(tqqq_px["Close"].iloc[i])
        sq_close = float(sqqq_px["Close"].iloc[i])
        tq_val = tq_sh * tq_close
        sq_val = sq_sh * sq_close
        nav = tq_val + sq_val + cash

        equity[i] = nav
        if nav > 0:
            w_tq[i] = tq_val / nav
            w_sq[i] = sq_val / nav
            w_cash[i] = cash / nav

    equity_s = pd.Series(equity, index=idx, name="equity")
    trades_df = pd.DataFrame(trade_rows)

    diagnostics = _diagnostics(equity_s, trades_df, starting_cash, pd.Series(w_tq, index=idx), pd.Series(w_sq, index=idx), pd.Series(w_cash, index=idx))
    return StrategyBacktestResult(
        equity=equity_s,
        tqqq_weight=pd.Series(w_tq, index=idx),
        sqqq_weight=pd.Series(w_sq, index=idx),
        cash_weight=pd.Series(w_cash, index=idx),
        trades=trades_df,
        diagnostics=diagnostics,
    )


def _diagnostics(equity, trades, starting_cash, w_tq, w_sq, w_cash) -> dict:
    if len(equity) < 2:
        return {}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / starting_cash) ** (1 / years) - 1 if years > 0 else 0.0

    rets = equity.pct_change().dropna()
    vol = float(rets.std() * np.sqrt(252))
    sharpe = float(rets.mean() * 252 / vol) if vol > 0 else 0.0

    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    max_dd = float(dd.min())

    # Round-trip win rate (FIFO matching)
    rt_wins = rt_losses = 0
    open_pos = {"TQQQ": [], "SQQQ": []}
    durations = []
    for _, t in trades.iterrows():
        side = "BUY" if t["dollar"] > 0 else "SELL"
        tk = t["ticker"]
        if side == "BUY":
            open_pos[tk].append(t)
        else:
            remaining = -t["shares"]
            while remaining > 1e-9 and open_pos[tk]:
                b = open_pos[tk].pop(0)
                matched = min(b["shares"], remaining)
                pnl = matched * (t["price"] - b["price"])
                if pnl > 0: rt_wins += 1
                elif pnl < 0: rt_losses += 1
                durations.append((t["date"] - b["date"]).days)
                remaining -= matched

    total_rt = rt_wins + rt_losses
    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "num_trades": len(trades),
        "num_round_trips": total_rt,
        "win_rate": rt_wins / total_rt if total_rt else 0.0,
        "avg_duration_days": float(np.mean(durations)) if durations else 0.0,
        "avg_tqqq_weight": float(w_tq.mean()),
        "avg_sqqq_weight": float(w_sq.mean()),
        "avg_cash_weight": float(w_cash.mean()),
        "pct_days_mostly_cash": float((w_cash > 0.5).mean()),
        "pct_days_short_heavy": float((w_sq > 0.5).mean()),
        "total_cost_paid": float(trades["cost"].sum()) if not trades.empty else 0.0,
    }