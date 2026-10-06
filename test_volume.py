from technical_indicators import TechnicalIndicatorAnalyzer

for t in ["MSFT", "NVDA", "CEG", "HON", "JPM", "CAT", "SO"]:
    a = TechnicalIndicatorAnalyzer(t, "1y")
    a.fetch_data()
    a.calculate_all_indicators()
    vm = a.volume_metrics
    print("{:<6} ratio={:.2f}  severity={:<10}  weight={:.2f}".format(
        t, vm["volume_ratio"], vm["severity"], a.get_volume_signal_weight()
    ))
