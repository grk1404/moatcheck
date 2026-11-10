import sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd, yfinance as yf
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

def show(title, obj):
    print("\n=== " + title + " ===")
    if obj is None:
        print("None")
    elif isinstance(obj, pd.DataFrame):
        print(obj if not obj.empty else "(empty DataFrame)")
    elif isinstance(obj, pd.Series):
        print(obj if not obj.empty else "(empty Series)")
    else:
        print(obj)

for ticker in sys.argv[1:]:
    print("\n" + "#" * 70)
    print("# " + ticker)
    print("#" * 70)
    try:
        t = yf.Ticker(ticker)
        try: show("analyst_price_targets", t.analyst_price_targets)
        except Exception as e: print("analyst_price_targets error: " + repr(e))
        try: show("growth_estimates", t.growth_estimates)
        except Exception as e: print("growth_estimates error: " + repr(e))
        try: show("earnings_estimate", t.earnings_estimate)
        except Exception as e: print("earnings_estimate error: " + repr(e))
        try: show("revenue_estimate", t.revenue_estimate)
        except Exception as e: print("revenue_estimate error: " + repr(e))
        try:
            info = t.info or {}
            print("\n=== info (analyst subset) ===")
            for k in ("earningsGrowth","revenueGrowth","earningsQuarterlyGrowth",
                      "trailingPE","forwardPE","recommendationKey","recommendationMean",
                      "numberOfAnalystOpinions","targetMeanPrice","targetHighPrice",
                      "targetLowPrice","targetMedianPrice"):
                print("  " + k.ljust(28) + " = " + repr(info.get(k)))
        except Exception as e:
            print("info error: " + repr(e))
    except Exception as e:
        print("!! " + ticker + ": " + repr(e))
