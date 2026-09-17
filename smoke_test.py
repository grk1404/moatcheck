from data_provider import get_ticker
import time

print("--- First call (fetches + caches) ---")
t0 = time.time()
t = get_ticker("AAPL")
price = t.history(period="1d")["Close"].iloc[-1]
print(f"AAPL price: {price:.2f}  time: {time.time()-t0:.2f}s")

print("--- Second call (should hit cache) ---")
t0 = time.time()
t2 = get_ticker("AAPL")
_ = t2.info
print(f"Elapsed: {time.time()-t0:.2f}s")

print()
print("--- Previously-broken tickers ---")
for sym in ["AON", "AER", "AA", "AG"]:
    try:
        tx = get_ticker(sym)
        h = tx.history(period="1d")
        if not h.empty:
            last_close = h["Close"].iloc[-1]
            print(f"{sym:6} price={last_close:.2f}  info_keys={len(tx.info)}")
        else:
            print(f"{sym:6} no history")
    except Exception as e:
        print(f"{sym:6} FAILED — {type(e).__name__}: {e}")