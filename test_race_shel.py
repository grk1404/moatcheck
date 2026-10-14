from moatcheck.combined import fetch
for t in ("RACE", "SHEL", "NVDA"):
    fin = fetch(t)
    print(f"--- {t} ---")
    print(f"  price:    {fin.current_price}")
    print(f"  sector:   {fin.company_sector!r}")
    print(f"  industry: {fin.company_industry!r}")
    print(f"  country:  {fin.company_country!r}")
    print(f"  summary:  {len(fin.company_summary)} chars")
    print(f"  52w:      {fin.week52_low} - {fin.week52_high}")
