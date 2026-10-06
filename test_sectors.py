from technical_indicators import TechnicalIndicatorAnalyzer

tickers = [
    # Tech
    "AAPL", "MSFT", "NVDA", "META",
    # Utilities (slow, dividend-heavy, capital-intensive)
    "NEE", "DUK", "SO", "AEP", "CEG", "VST",
    # Financials (rate-sensitive, cyclical)
    "JPM", "BAC", "WFC", "GS", "PGR",
    # Cyclical / industrial for contrast
    "BA", "CAT", "DE",
]

header = "{:<8} {:>10} {:<12} {:>6} {:<8} {:>6} {:>6} {:>6} {:>6} {:>6}".format(
    "Ticker", "Composite", "Verdict", "Conf", "Regime",
    "SMA", "MACD", "RSI", "Stoch", "BB"
)
print(header)
print("-" * 100)

for t in tickers:
    try:
        a = TechnicalIndicatorAnalyzer(t, "1y")
        if not a.fetch_data():
            print("{:<8} fetch failed".format(t))
            continue
        a.calculate_all_indicators()
        s = a.get_trading_signals()
        sc = s["indicator_scores"]
        print("{:<8} {:>+10.3f} {:<12} {:>5.1f}% {:<8} {:>+6.2f} {:>+6.2f} {:>+6.2f} {:>+6.2f} {:>+6.2f}".format(
            t,
            s["composite"],
            s["recommendation"],
            s["confidence"],
            s["regime"],
            sc["sma"], sc["macd"], sc["rsi"], sc["stoch"], sc["bollinger"],
        ))
    except Exception as e:
        print("{:<8} error: {}".format(t, e))
