from technical_indicators import TechnicalIndicatorAnalyzer

tickers = [
    # Tech (large cap, momentum-driven)
    "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "TSLA", "AVGO",
    # Utilities (slow, dividend-heavy, capital-intensive)
    "NEE", "DUK", "SO", "AEP", "CEG", "VST", "EXC", "D", "XEL",
    # Financials (rate-sensitive, cyclical)
    "JPM", "BAC", "WFC", "GS", "PGR", "MS", "BLK",
    # REITs (income-heavy, rate-sensitive)
    "O", "SPG", "PLD", "AMT",
    # Cyclicals / industrials
    "BA", "CAT", "DE", "HON",
]

header = "{:<8} {:>10} {:<12} {:>6} {:<8} {:>6} {:>6} {:>6} {:>6} {:>6}".format(
    "Ticker", "Composite", "Verdict", "Conf", "Regime",
    "SMA", "MACD", "RSI", "Stoch", "BB"
)
print(header)
print("-" * 100)

results = []
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
        results.append(s)
    except Exception as e:
        print("{:<8} error: {}".format(t, e))

# ---- Sector-level summary ----
print()
print("=" * 100)
print("SECTOR SUMMARY")
print("=" * 100)

sectors = {
    "Tech":      ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "TSLA", "AVGO"],
    "Utilities": ["NEE", "DUK", "SO", "AEP", "CEG", "VST", "EXC", "D", "XEL"],
    "Financials":["JPM", "BAC", "WFC", "GS", "PGR", "MS", "BLK"],
    "REITs":     ["O", "SPG", "PLD", "AMT"],
    "Cyclicals": ["BA", "CAT", "DE", "HON"],
}

for sector, syms in sectors.items():
    sector_results = []
    for t, s in zip(tickers, results):
        # Only include if the ticker matches this sector and we have a result
        if t in syms:
            sector_results.append(s)
    if not sector_results:
        continue
    avg_comp = sum(r["composite"] for r in sector_results) / len(sector_results)
    buys = sum(1 for r in sector_results if "BUY" in r["recommendation"])
    sells = sum(1 for r in sector_results if "SELL" in r["recommendation"])
    waits = len(sector_results) - buys - sells
    print("{:<12} n={:<3} avg_composite={:>+7.3f}  buys={}  sells={}  wait={}".format(
        sector, len(sector_results), avg_comp, buys, sells, waits
    ))
