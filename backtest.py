"""
Backtest engine for MoatCheck's Back Test page.

Simulates a long-or-cash position in ONE ticker, sized by a target weight
in [0, 1] (0/1 for the on/off strategies, fractional for Sleeves and the dial):
  - Signal computed from prices up to and including day N's close
  - Target weight executes at day N+1's open (daily rebalance to target)
  - Optional same-day-close exit for rules that say "sell at the close"
  - Equity marks to market at each day's close
  - No lookahead; optional per-side trading cost in basis points

Price frame columns (see load_backtest_data):
  open, close          the entered ticker (what is actually traded)
  signal_close         the series trend-regime strategies read (ticker or QQQ)
  qqq_open, qqq_close  Nasdaq-100 ETF, used only as a benchmark
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

BENCH_SYMBOL = "QQQ"
# 250-day SMA (live regime filter) needs ~250 trading bars = ~360 calendar
# days before the window; 450 leaves margin for holidays and data gaps.
WARMUP_CALENDAR_DAYS = 450


@dataclass
class BacktestResult:
    """Everything the UI needs to display a backtest run."""
    ticker: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    starting_cash: float

    # Daily series
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_ticker: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_qqq: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    position: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # weight held at close

    # Round-trip trade log (flat -> invested -> flat)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Summary metrics (percentages are fractions, e.g. 0.15 = 15%)
    total_return: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    num_rebalances: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    time_in_market: float = 0.0
    avg_exposure: float = 0.0

    # Buy & hold benchmarks, entered on the same bar as the strategies
    ticker_buy_hold_return: float = 0.0
    qqq_buy_hold_return: float = 0.0


def load_backtest_data(ticker: str, start: str, end: str, signal_symbol: str | None = None) -> pd.DataFrame:
    """Fetch daily OHLC for the traded ticker, the signal source and QQQ.

    Loads WARMUP_CALENDAR_DAYS of history before `start` so indicators are
    defined on the first day of the window. Rows are inner-joined across
    symbols; df.attrs["dropped_ticker_bars"] counts ticker bars lost to the
    join so the UI can warn when another symbol's calendar clipped the data.
    """
    from data_provider import get_ticker

    start_dt = pd.to_datetime(start)
    extended_start = (start_dt - timedelta(days=WARMUP_CALENDAR_DAYS)).strftime("%Y-%m-%d")
    signal_symbol = signal_symbol or ticker

    hist_by_sym: dict[str, pd.DataFrame] = {}
    for sym in dict.fromkeys([ticker, BENCH_SYMBOL, signal_symbol]):
        try:
            hist = get_ticker(sym).history(start=extended_start, end=end)
        except Exception as e:
            raise RuntimeError(f"Could not fetch {sym}: {e}") from e
        if hist is None or hist.empty:
            raise RuntimeError(f"No data returned for {sym} in {extended_start}..{end}")
        hist = hist.copy()
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist_by_sym[sym] = hist

    tk, qq, sg = hist_by_sym[ticker], hist_by_sym[BENCH_SYMBOL], hist_by_sym[signal_symbol]
    df = pd.DataFrame({
        "open": tk["Open"],
        "close": tk["Close"],
        "signal_close": sg["Close"],
        "qqq_open": qq["Open"],
        "qqq_close": qq["Close"],
    })
    ticker_bars = int(tk[["Open", "Close"]].dropna().shape[0])
    df = df.dropna()
    df.attrs["dropped_ticker_bars"] = ticker_bars - len(df)
    df.attrs["ticker"] = ticker
    df.attrs["signal_symbol"] = signal_symbol
    return df


def _compute_max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a positive fraction (0.25 = -25%)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(-drawdown.min())


def _compute_sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sharpe = sqrt(252) x mean(daily excess return) / std(daily excess return).

    `risk_free` is an annual rate (default 0). This is NOT CAGR / volatility.
    """
    if returns.empty:
        return 0.0
    excess = returns - risk_free / 252.0
    sd = excess.std()
    if not np.isfinite(sd) or sd < 1e-12:  # constant series gives ~1e-18 float noise, not exactly 0
        return 0.0
    return float(np.sqrt(252.0) * excess.mean() / sd)


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


def _buy_hold(px_open: pd.Series, px_close: pd.Series, entry_i: int, cash: float) -> pd.Series:
    """Buy at bar `entry_i`'s open with all the cash, hold to the end; cash before."""
    vals = px_close / float(px_open.iloc[entry_i]) * cash
    vals.iloc[:entry_i] = cash
    return vals


def run_backtest(
    prices: pd.DataFrame,
    signal: pd.Series,
    starting_cash: float = 10000.0,
    warmup_days: int = 0,
    progress_callback=None,
    cost_bps: float = 0.0,
    exit_at_close: pd.Series | None = None,
    ticker: str = "",
) -> BacktestResult:
    """Run the long-or-cash backtest.

    Args:
        prices: frame with open, close, qqq_open, qqq_close columns.
        signal: target weight in [0, 1] (bool/int/float) indexed like prices.
                signal[N] is the weight to hold from day N+1's open.
        starting_cash: initial portfolio value.
        warmup_days: leading bars to skip (0 = none; the page slices its own window).
        cost_bps: cost per side, in basis points of traded value (default 0).
        exit_at_close: optional bool Series; on True days any open position is
                sold at that day's close instead of waiting for the next open.
        ticker: label for the result.
    """
    if prices.empty:
        raise ValueError("No price data supplied.")
    if warmup_days > 0 and len(prices) > warmup_days:
        prices = prices.iloc[warmup_days:].copy()
    if prices.empty:
        raise ValueError(f"After {warmup_days} warm-up days, no price data remains. Use a longer date range.")

    idx = prices.index
    n = len(prices)
    weight = signal.reindex(idx).fillna(0.0).astype(float).clip(0.0, 1.0)
    target = weight.shift(1).fillna(0.0).to_numpy()  # weight to hold from today's open
    if exit_at_close is not None:
        close_exit = exit_at_close.reindex(idx).fillna(False).astype(bool).to_numpy()
    else:
        close_exit = np.zeros(n, dtype=bool)

    opens = prices["open"].to_numpy(dtype=float)
    closes = prices["close"].to_numpy(dtype=float)
    cf = cost_bps / 10_000.0

    cash = float(starting_cash)
    shares = 0.0
    equity = np.zeros(n)
    held_w = np.zeros(n)
    trip: dict | None = None
    trade_rows: list[dict] = []
    rebalances = 0

    def _finish(date, open_at_end=False):
        nonlocal trip
        pnl = trip["sell_val"] - trip["buy_val"] - trip["cost"]
        row = {
            "entry_date": trip["entry_date"], "exit_date": date,
            "entry_price": trip["buy_val"] / trip["buy_sh"],
            "exit_price": trip["sell_val"] / trip["sell_sh"],
            "shares": trip["buy_sh"], "pnl": pnl,
            "return_pct": pnl / trip["buy_val"],
            "days_held": (date - trip["entry_date"]).days,
        }
        if open_at_end:
            row["open_at_end"] = True
        trade_rows.append(row)
        trip = None

    def _sell(value, shares_sold, price_shares_unused=None):
        nonlocal cash
        cost = value * cf
        cash += value - cost
        trip["sell_val"] += value
        trip["sell_sh"] += shares_sold
        trip["cost"] += cost

    for i in range(n):
        date = idx[i]
        px_o, px_c = opens[i], closes[i]

        if px_o > 0:
            cur = shares * px_o
            nav = cash + cur
            desired = target[i] * nav
            diff = desired - cur
            tol = 1e-9 * max(nav, 1.0)
            if shares > 0 and target[i] <= 0.0:
                _sell(cur, shares)
                shares = 0.0
                rebalances += 1
                _finish(date)
            elif diff > tol:
                buy_val = min(diff, cash / (1.0 + cf))
                if buy_val > 0:
                    if shares == 0.0:
                        trip = {"entry_date": date, "buy_val": 0.0, "buy_sh": 0.0, "sell_val": 0.0, "sell_sh": 0.0, "cost": 0.0}
                    sh = buy_val / px_o
                    shares += sh
                    cash -= buy_val * (1.0 + cf)
                    trip["buy_val"] += buy_val
                    trip["buy_sh"] += sh
                    trip["cost"] += buy_val * cf
                    rebalances += 1
            elif diff < -tol and shares > 0:
                sell_val = -diff
                sh = sell_val / px_o
                shares -= sh
                _sell(sell_val, sh)
                rebalances += 1

        if close_exit[i] and shares > 0 and px_c > 0:
            _sell(shares * px_c, shares)
            shares = 0.0
            rebalances += 1
            _finish(date)

        equity[i] = cash + shares * px_c
        held_w[i] = (shares * px_c / equity[i]) if equity[i] > 0 else 0.0

        if progress_callback and i % max(1, n // 50) == 0:
            progress_callback((i + 1) / n)

    if trip is not None and shares > 0:  # still invested at the end: mark at the last close
        trip["sell_val"] += shares * closes[-1]
        trip["sell_sh"] += shares
        _finish(idx[-1], open_at_end=True)

    equity_s = pd.Series(equity, index=idx)
    trades_df = pd.DataFrame(trade_rows)
    position = pd.Series(held_w, index=idx)

    daily_returns = equity_s.pct_change().dropna()
    total_return = float(equity_s.iloc[-1] / starting_cash - 1.0)
    vol = float(daily_returns.std() * np.sqrt(252.0)) if not daily_returns.empty else 0.0
    num_trades = len(trades_df)
    if num_trades:
        wins = trades_df[trades_df["return_pct"] > 0]
        losses = trades_df[trades_df["return_pct"] <= 0]
        win_rate = float(len(wins) / num_trades)
        avg_win = float(wins["return_pct"].mean()) if not wins.empty else 0.0
        avg_loss = float(losses["return_pct"].mean()) if not losses.empty else 0.0
    else:
        win_rate = avg_win = avg_loss = 0.0

    # Benchmarks enter on the first executable bar (bar 1's open), exactly like Buy and Hold.
    entry_i = 1 if n > 1 else 0
    bench_t = _buy_hold(prices["open"], prices["close"], entry_i, starting_cash)
    bench_q = _buy_hold(prices["qqq_open"], prices["qqq_close"], entry_i, starting_cash)

    return BacktestResult(
        ticker=ticker, start_date=idx[0], end_date=idx[-1], starting_cash=starting_cash,
        equity_curve=equity_s, benchmark_ticker=bench_t, benchmark_qqq=bench_q, position=position,
        trades=trades_df, total_return=total_return, cagr=_compute_cagr(equity_s), volatility=vol,
        sharpe=_compute_sharpe(daily_returns), max_drawdown=_compute_max_drawdown(equity_s),
        num_trades=num_trades, num_rebalances=rebalances, win_rate=win_rate, avg_win=avg_win,
        avg_loss=avg_loss, time_in_market=float((position > 1e-9).mean()), avg_exposure=float(position.mean()),
        ticker_buy_hold_return=float(bench_t.iloc[-1] / starting_cash - 1.0),
        qqq_buy_hold_return=float(bench_q.iloc[-1] / starting_cash - 1.0),
    )
