from moatcheck import fetch, compute_big5, dcf_two_stage, peter_lynch_fair, graham_number, graham_formula, peg_fair_value
import statistics

for sym in ("SWK", "ZM", "AAPL", "META"):
    try:
        fin = fetch(sym)
        big5 = compute_big5(fin)
        eps_w = big5.eps.values
        growth = eps_w.get(5) or eps_w.get(3)
        eps_ttm = float(fin.eps.iloc[-1])
        fcf_ttm = float(fin.free_cash_flow.iloc[-1]) if not fin.free_cash_flow.empty else None

        vals = []
        dcf = dcf_two_stage(fcf_ttm=fcf_ttm, shares_out=fin.shares_outstanding, current_price=fin.current_price, growth_rate=growth, discount_rate=0.10, terminal_growth=0.025)
        if dcf and dcf.fair_value: vals.append(dcf.fair_value)
        lynch = peter_lynch_fair(current_eps=eps_ttm, growth_rate=growth, dividend_yield=fin.dividend_yield, current_price=fin.current_price, mos=0.25)
        if lynch and lynch.fair_value: vals.append(lynch.fair_value)
        gn = graham_number(current_eps=eps_ttm, book_value_per_share=fin.book_value_per_share, current_price=fin.current_price, mos=0.25)
        if gn and gn.fair_value: vals.append(gn.fair_value)
        gf = graham_formula(current_eps=eps_ttm, growth_rate=growth, current_price=fin.current_price, aaa_bond_yield=0.045, mos=0.25)
        if gf and gf.fair_value: vals.append(gf.fair_value)
        peg = peg_fair_value(current_eps=eps_ttm, growth_rate=growth, current_price=fin.current_price, mos=0.25)
        if peg and peg.fair_value: vals.append(peg.fair_value)

        med = statistics.median(vals) if len(vals) >= 2 else (vals[0] if vals else None)
        vs = (fin.current_price - med) / med * 100 if med else None
        print("{:5s} price=${:7.2f} intrinsic={} vs={}".format(
            sym, fin.current_price,
            "${:.2f}".format(med) if med else "—",
            "{:+.0f}%".format(vs) if vs is not None else "—"
        ))
    except Exception as e:
        print("{}: {}".format(sym, e))
