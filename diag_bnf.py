from technical_indicators import TechnicalIndicatorAnalyzer

tickers = ["AAPL", "MSFT", "NVDA", "BBWI", "SMCI", "SWK", "ZM", "ULTA", "KNSL", "LCID"]

header = "{:<8} {:>10} {:>10} {:>12} {:>8} {:>10} {:>10}".format(
    "Ticker", "Price", "MA25", "Dislocation", "RSI", "BB_Lower", "Below BB?"
)
print(header)
print("-" * 85)

for t in tickers:
    try:
        a = TechnicalIndicatorAnalyzer(t, "1y")
        if not a.fetch_data():
            print("{:<8} fetch failed".format(t))
            continue
        a.calculate_all_indicators()
        res = a.check_bnf_reversal_setup()
        if "error" in res and res.get("error"):
            print("{:<8} error: {}".format(t, res["error"]))
            continue

        price = res["current_price"]
        ma25 = res["ma_25"]
        disp = res["dislocation_pct"]
        rsi = res["rsi"]

        # Get bb_lower directly
        bb_lower = float(a.stock_data["BB_Lower"].iloc[-1]) if "BB_Lower" in a.stock_data.columns else None
        below_bb = "YES" if bb_lower and price < bb_lower else "no"

        print("{:<8} {:>10.2f} {:>10.2f} {:>11.2f}% {:>8.1f} {:>10.2f} {:>10}".format(
            t, price, ma25, disp, rsi, bb_lower if bb_lower else 0, below_bb
        ))
    except Exception as e:
        print("{:<8} error: {}".format(t, e))
