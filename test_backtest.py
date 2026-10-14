from backtest import load_backtest_data, run_backtest
import pandas as pd

prices = load_backtest_data("2020-01-01", "2024-01-01")
print(f"Loaded {len(prices)} trading days from {prices.index[0].date()} to {prices.index[-1].date()}")

# Always long — should match TQQQ buy-and-hold
signal = pd.Series(1, index=prices.index)
result = run_backtest(prices, signal, starting_cash=10000.0)

print()
print(f"Strategy total return:  {result.total_return*100:+.2f}%")
print(f"TQQQ buy-and-hold:      {result.tqqq_buy_hold_return*100:+.2f}%")
print(f"Difference:             {(result.total_return - result.tqqq_buy_hold_return)*100:+.4f}%")
print()
print(f"CAGR:                   {result.cagr*100:+.2f}%")
print(f"Max drawdown:           {result.max_drawdown*100:.2f}%")
print(f"Sharpe:                 {result.sharpe:.2f}")
print(f"Number of trades:       {result.num_trades}")
