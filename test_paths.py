from moatcheck.fetcher import fetch as yf_fetch
from moatcheck.combined import fetch as combined_fetch

for sym in ("RACE", "SHEL"):
    print(f"=== {sym} ===")
    try:
        yf_fin = yf_fetch(sym)
        print(f"  yf_fetch:        sector={yf_fin.company_sector!r} country={yf_fin.company_country!r} summary_len={len(yf_fin.company_summary)}")
    except Exception as e:
        print(f"  yf_fetch ERROR: {e}")
    try:
        c_fin = combined_fetch(sym)
        print(f"  combined_fetch:  sector={c_fin.company_sector!r} country={c_fin.company_country!r} summary_len={len(c_fin.company_summary)}")
    except Exception as e:
        print(f"  combined_fetch ERROR: {e}")
