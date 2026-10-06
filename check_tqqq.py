from data_provider import get_ticker

t = get_ticker("TQQQ")
s = get_ticker("SQQQ")

ti = t.fast_info
si = s.fast_info

t_fast = ti.get("last_price") or ti.get("lastPrice")
s_fast = si.get("last_price") or si.get("lastPrice")

hist_t = t.history(period="5d")
hist_s = s.history(period="5d")
t_close = float(hist_t["Close"].iloc[-1])
s_close = float(hist_s["Close"].iloc[-1])

print("TQQQ fast_info last_price:", t_fast)
print("SQQQ fast_info last_price:", s_fast)
print("TQQQ last close:         ", t_close)
print("SQQQ last close:         ", s_close)
print()

qty_t = 21.362
qty_s = 38.218

app_value_fast = qty_t * float(t_fast) + qty_s * float(s_fast)
app_value_close = qty_t * t_close + qty_s * s_close

print("Sleeve value using fast_info:")
print("  {:.2f}".format(app_value_fast))
print()
print("Sleeve value using last close:")
print("  {:.2f}".format(app_value_close))
print()

fid_t = 82.9223
fid_s = 32.345
fid_value = qty_t * fid_t + qty_s * fid_s
print("Sleeve value using Fidelity CSV prices:")
print("  {:.2f}".format(fid_value))
print()

committed = 3009.70
print("Committed capital:", committed)
print("P&L vs fast_info:  {:+.2f}".format(app_value_fast - committed))
print("P&L vs last close: {:+.2f}".format(app_value_close - committed))
print("P&L vs Fidelity:   {:+.2f}".format(fid_value - committed))
