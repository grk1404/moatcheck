from moatcheck import fetch, value_price, compute_big5

for sym in ("ZM", "KNSL"):
    fin = fetch(sym)
    big5 = compute_big5(fin)
    eps_windows = big5.eps.values
    stable = sorted([v for w, v in eps_windows.items() if w in (5, 3, 10) and v is not None])
    if stable:
        growth = stable[len(stable)//2] if len(stable) % 2 else (stable[len(stable)//2-1] + stable[len(stable)//2]) / 2
    else:
        growth = eps_windows.get(1)

    # Sanity guard, same as app
    if growth is not None:
        if growth > 0.30: growth = 0.30
        if growth < 0.02: growth = 0.02

    eps_ttm = float(fin.eps.iloc[-1]) if not fin.eps.empty else None
    current_eps = eps_ttm
    if current_eps is None or current_eps <= 0:
        if not fin.eps.empty:
            positive = fin.eps[fin.eps > 0]
            if not positive.empty:
                current_eps = float(positive.tail(3).mean())

    val = value_price(
        current_eps=current_eps,
        big5_eps_growth=None,
        analyst_growth=None,
        historical_pe=fin.pe_ratio_ttm,
        custom_growth=0.15,
    )

    print(f"=== {sym} ===")
    print(f"  current price:  ${fin.current_price:.2f}")
    print(f"  EPS TTM:        ${eps_ttm:.2f}" if eps_ttm else "  EPS TTM:        n/a")
    print(f"  growth used:    {growth*100:.1f}%" if growth else "  growth used:    n/a")
    if val:
        print(f"  value_price:    ${val.value_price:.2f}")
        print(f"  val.mos_price:  ${val.mos_price:.2f}")
        print(f"  price <= MOS?   {fin.current_price <= val.mos_price}")
    else:
        print("  value_price:    None (cannot compute)")
    print()
