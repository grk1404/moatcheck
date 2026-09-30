from investments import _find_csv, _parse_fidelity_csv, _derive, _portfolio_returns

df = _derive(_parse_fidelity_csv(_find_csv()))
r = _portfolio_returns(df)

for w in (10, 5, 3):
    info = r.get(w, {})
    cagr = info.get("cagr")
    n_in = info.get("n_included", 0)
    n_tot = info.get("n_total", 0)
    cov = info.get("coverage_basis", 0.0)
    tot = info.get("total_basis", 0.0)
    if cagr is None:
        print("{}Y: n/a   included={}/{}  basis_cov={:,.0f}/{:.0f}".format(w, n_in, n_tot, cov, tot))
    else:
        print("{}Y: cagr={:+.2f}%  included={}/{}  basis_cov={:,.0f}/{:.0f}".format(w, cagr * 100, n_in, n_tot, cov, tot))
