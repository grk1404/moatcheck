from data_provider import get_ticker

tickers = ["FTNT", "FXAIX", "ABBV", "SPY", "ATKR", "ESINQ"]

for sym in tickers:
    try:
        info = get_ticker(sym).info or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        high = info.get("fiftyTwoWeekHigh")
        low = info.get("fiftyTwoWeekLow")
        print("{}: price={} high52={} low52={}".format(sym, price, high, low))
    except Exception as e:
        print("{}: ERROR {}: {}".format(sym, type(e).__name__, e))
