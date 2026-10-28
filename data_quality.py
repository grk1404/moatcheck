import pandas as pd

def check_data_quality(prices: pd.DataFrame) -> dict:
    """Perform sanity checks on the price data.
    
    Returns a dictionary of quality metrics.
    """
    if prices.empty or "Close" not in prices.columns:
        return {"error": "Data is empty or missing 'Close' column."}

    # 1. Date Range & Length
    start_date = prices.index[0].strftime("%Y-%m-%d")
    end_date = prices.index[-1].strftime("%Y-%m-%d")
    total_rows = len(prices)
    
    # 2. Missing Trading Days
    expected_years = (prices.index[-1] - prices.index[0]).days / 365.25
    bdays = pd.bdate_range(start=prices.index[0], end=prices.index[-1])
    missing_days = len(bdays) - total_rows
    expected_missing = expected_years * 9 # ~9 market holidays per year
    
    # 3. Zero or Negative Prices
    bad_prices = prices[prices["Close"] <= 0]
    bad_price_count = len(bad_prices)
    
    # 4. Massive Jumps (>50% in one day implies unadjusted split)
    if total_rows > 1:
        daily_returns = prices["Close"].pct_change()
        jumps = daily_returns[daily_returns.abs() > 0.50]
        jump_count = len(jumps)
    else:
        jump_count = 0

    return {
        "start_date": start_date,
        "end_date": end_date,
        "total_rows": total_rows,
        "missing_days": missing_days,
        "expected_missing": expected_missing,
        "bad_price_count": bad_price_count,
        "jump_count": jump_count,
        "is_healthy": bad_price_count == 0 and jump_count == 0 and missing_days <= expected_missing + 20
    }