from backtest import load_backtest_data, run_backtest
import backtest_strategies as bs
import pandas as pd

# Load data
prices = load_backtest_data("2020-01-01", "2024-01-01")
qqq_close = prices["QQQ_close"]
tqqq_close = prices["TQQQ_close"]

print(f"Loaded {len(prices)} days: {prices.index[0].date()} to {prices.index[-1].date()}")
print()

for name, fn in bs.STRATEGIES.items():
    print(f"--- {name} ---")
    signal = fn(qqq_close, tqqq_close)
    result = run_backtest(prices, signal, starting_cash=10000.0, warmup_days=30)
    print(f"  Total return:     {result.total_return*100:+.2f}%")
    print(f"  CAGR:             {result.cagr*100:+.2f}%")
    print(f"  Max drawdown:     {result.max_drawdown*100:.2f}%")
    print(f"  Sharpe:           {result.sharpe:.2f}")
    print(f"  Number of trades: {result.num_trades}")
    print(f"  Time in market:   {result.time_in_market*100:.0f}%")
    print(f"  Win rate:         {result.win_rate*100:.0f}%")
    print()
