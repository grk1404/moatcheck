"""
Backtest engine for MoatCheck.

Simulates a long-or-flat TQQQ strategy with next-day open execution:
  - Signal computed from prices up to and including day N's close
  - Trade executes at day N+1's open
  - Equity marks to market at each day's close
  - No lookahead, no same-day fills

Supports two price series:
  - Mode A (default): real TQQQ closes from yfinance — the expense
    ratio is already baked into the historical NAV.
  - Mode B (comparison): synthetic "no-fee TQQQ" reconstructed from
    QQQ daily returns × 3, with no expense deduction. The difference
    between A and B shows the fee drag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Everything the UI needs to display a backtest run."""
    ticker: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    starting_cash: float

    # Daily series
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_tqqq: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_qqq: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    position: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))

    # Trade log
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Summary metrics (all percentages are fractions, e.g. 0.15 = 15%)
    total_return: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    time_in_market: float = 0.0

    # Benchmark comparisons
    tqqq_buy_hold_return: float = 0.0
    qqq_buy_hold_return: float = 0.0

    # Optional: synthetic no-fee comparison (Mode B)
    no_fee_total_return: float | None = None
    fee_drag: float | None = None


def load_backtest_data(start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLC for TQQQ and QQQ.

    Returns a DataFrame with columns:
        TQQQ_open, TQQQ_close, QQQ_close
    indexed by date.

    Raises RuntimeError if either ticker has no data in the window.
    """
    from data_provider import get_ticker

    frames = {}
    for sym in ("TQQQ", "QQQ"):
        try:
            hist = get_ticker(sym).history(start=start, end=end)
        except Exception as e:
            raise RuntimeError(f"Could not fetch {sym}: {e}") from e
        if hist is None or hist.empty:
            raise RuntimeError(f"No data returned for {sym} in {start}..{end}")

        hist = hist.copy()
        hist.index = pd.to_datetime(hist.index).tz_localize(None)

        if sym == "TQQQ":
            frames["TQQQ_open"] = hist["Open"]
            frames["TQQQ_close"] = hist["Close"]
        else:
            frames["QQQ_close"] = hist["Close"]

    df = pd.DataFrame(frames)
    # Align on common trading days only
    df = df.dropna()
    return df


def _compute_max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a positive fraction (0.25 = -25%)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(-drawdown.min())


def _compute_sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sharpe from daily returns. Risk-free rate assumed 0."""
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - risk_free / 252.0
    return float(np.sqrt(252.0) * excess.mean() / returns.std())


def _compute_cagr(equity: pd.Series) -> float:
    """Compound annual growth rate from the equity curve."""
    if equity.empty or len(equity) < 2:
        return 0.0
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    if start_val <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((end_val / start_val) ** (1.0 / years) - 1.0)


def run_backtest(prices: pd.DataFrame, signal: pd.Series, starting_cash: float = 10000.0, warmup_days: int = 30, progress_callback=None) -> BacktestResult:
    """Run the long-or-flat backtest.

    Args:
        prices: DataFrame with TQQQ_open, TQQQ_close, QQQ_close columns.
        signal: Series of 0 (cash) or 1 (long TQQQ) indexed the same as prices.
                signal[N] is the position to hold on day N+1.
        starting_cash: initial portfolio value.
        warmup_days: number of trading days to skip at the start. Gives
                     indicators like SMA 200 time to compute before the
                     strategy is allowed to trade.
        progress_callback: optional callable(fraction_done: float).

    Returns:
        BacktestResult with equity curve, trades, and metrics.
    """
    if prices.empty:
        raise ValueError("No price data supplied.")

    # --- Trim the warm-up window ---
    if warmup_days > 0 and len(prices) > warmup_days:
        prices = prices.iloc[warmup_days:].copy()

    if prices.empty:
        raise ValueError(f"After {warmup_days} warm-up days, no price data remains. Use a longer date range.")

    signal = signal.reindex(prices.index).fillna(0).astype(int)
    position = signal.shift(1).fillna(0).astype(int)

    n = len(prices)
    cash = float(starting_cash)
    shares = 0.0
    equity_values = np.zeros(n)
    trade_rows = []
    open_trade = None

    for i in range(n):
        date = prices.index[i]
        px_open = float(prices["TQQQ_open"].iloc[i])
        px_close = float(prices["TQQQ_close"].iloc[i])
        want_long = position.iloc[i] == 1
        is_long = shares > 0

        if want_long and not is_long and px_open > 0:
            shares = cash / px_open
            cash = 0.0
            open_trade = {"entry_date": date, "entry_price": px_open, "shares": shares}

        elif not want_long and is_long and px_open > 0:
            proceeds = shares * px_open
            cash = proceeds
            if open_trade is not None:
                pnl = (px_open - open_trade["entry_price"]) * open_trade["shares"]
                ret_pct = px_open / open_trade["entry_price"] - 1.0
                trade_rows.append({"entry_date": open_trade["entry_date"], "exit_date": date, "entry_price": open_trade["entry_price"], "exit_price": px_open, "shares": open_trade["shares"], "pnl": pnl, "return_pct": ret_pct, "days_held": (date - open_trade["entry_date"]).days})
            shares = 0.0
            open_trade = None

        equity_values[i] = cash + shares * px_close

        if progress_callback and i % max(1, n // 50) == 0:
            progress_callback((i + 1) / n)

    if open_trade is not None:
        final_close = float(prices["TQQQ_close"].iloc[-1])
        pnl = (final_close - open_trade["entry_price"]) * open_trade["shares"]
        ret_pct = final_close / open_trade["entry_price"] - 1.0
        trade_rows.append({"entry_date": open_trade["entry_date"], "exit_date": prices.index[-1], "entry_price": open_trade["entry_price"], "exit_price": final_close, "shares": open_trade["shares"], "pnl": pnl, "return_pct": ret_pct, "days_held": (prices.index[-1] - open_trade["entry_date"]).days, "open_at_end": True})

    equity = pd.Series(equity_values, index=prices.index)
    trades_df = pd.DataFrame(trade_rows)

    daily_returns = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / starting_cash - 1.0) if len(equity) else 0.0
    cagr = _compute_cagr(equity)
    vol = float(daily_returns.std() * np.sqrt(252.0)) if not daily_returns.empty else 0.0
    sharpe = _compute_sharpe(daily_returns)
    max_dd = _compute_max_drawdown(equity)
    num_trades = len(trades_df)

    if num_trades > 0:
        wins = trades_df[trades_df["return_pct"] > 0]
        losses = trades_df[trades_df["return_pct"] <= 0]
        win_rate = float(len(wins) / num_trades)
        avg_win = float(wins["return_pct"].mean()) if not wins.empty else 0.0
        avg_loss = float(losses["return_pct"].mean()) if not losses.empty else 0.0
    else:
        win_rate = avg_win = avg_loss = 0.0

    time_in_market = float(position.mean()) if len(position) else 0.0

    if (position == 1).any():
        first_trade_date = position[position == 1].index[0]
    else:
        first_trade_date = prices.index[0]

    first_idx = prices.index.get_loc(first_trade_date)
    tqqq_entry_price = float(prices["TQQQ_open"].iloc[first_idx])
    qqq_entry_price = float(prices["QQQ_close"].iloc[first_idx])

    if tqqq_entry_price <= 0 or qqq_entry_price <= 0:
        tqqq_entry_price = float(prices["TQQQ_close"].iloc[0])
        qqq_entry_price = float(prices["QQQ_close"].iloc[0])

    tqqq_vals = prices["TQQQ_close"] / tqqq_entry_price * starting_cash
    qqq_vals = prices["QQQ_close"] / qqq_entry_price * starting_cash
    tqqq_bh = float(tqqq_vals.iloc[-1] / starting_cash - 1.0)
    qqq_bh = float(qqq_vals.iloc[-1] / starting_cash - 1.0)

    return BacktestResult(ticker="TQQQ", start_date=prices.index[0], end_date=prices.index[-1], starting_cash=starting_cash, equity_curve=equity, benchmark_tqqq=tqqq_vals, benchmark_qqq=qqq_vals, position=position, trades=trades_df, total_return=total_return, cagr=cagr, volatility=vol, sharpe=sharpe, max_drawdown=max_dd, num_trades=num_trades, win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss, time_in_market=time_in_market, tqqq_buy_hold_return=tqqq_bh, qqq_buy_hold_return=qqq_bh)