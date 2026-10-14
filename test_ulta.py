from moatcheck import fetch, value_price, compute_big5

fin = fetch("ULTA")
big5 = compute_big5(fin)
eps_windows = big5.eps.values
stable = sorted([v for w, v in eps_windows.items() if w in (5, 3, 10) and v is not None])
growth = stable[len(stable)//2] if len(stable) % 2 else (stable[len(stable)//2-1] + stable[len(stable)//2]) / 2

val = value_price(
    current_eps=float(fin.eps.iloc[-1]),
    big5_eps_growth=None,
    analyst_growth=None,
    historical_pe=fin.pe_ratio_ttm,
    custom_growth=0.15,
)

print(f"ULTA current price:  ${fin.current_price:.2f}")
print(f"value_price:         ${val.value_price:.2f}")
print(f"val.mos_price:       ${val.mos_price:.2f}")
print(f"mos multiplier:      {val.mos_price / val.value_price:.2f}x")
print(f"price <= mos_price?  {fin.current_price <= val.mos_price}")
