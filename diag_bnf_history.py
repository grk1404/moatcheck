from technical_indicators import TechnicalIndicatorAnalyzer

# CCL (Carnival Cruise) crashed ~80% in 2020
# BA (Boeing) crashed ~70%
# XOM (Exxon) crashed ~50%
for t in ["CCL", "BA", "XOM"]:
    a = TechnicalIndicatorAnalyzer(t, "max")
    if not a.fetch_data():
        print(t, "fetch failed")
        continue
    a.calculate_all_indicators()
    df = a.stock_data.copy()
    df["MA25"] = df["Close"].rolling(25).mean()
    df["Disloc"] = (df["Close"] - df["MA25"]) / df["MA25"] * 100

    # Find days where dislocation <= -20% AND (RSI < 30 OR close < BB_Lower)
    hits = df[(df["Disloc"] <= -20) & ((df["RSI"] < 30) | (df["Close"] < df["BB_Lower"]))]
    print(f"{t}: {len(hits)} BNF trigger days found")
    if len(hits) > 0:
        print(f"   first: {hits.index[0].date()}   last: {hits.index[-1].date()}")
        print(f"   worst dislocation: {hits['Disloc'].min():.1f}%")
