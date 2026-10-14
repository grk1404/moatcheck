from moatcheck.combined import fetch
for t in ("NVDA", "PARR", "RACE", "SHEL"):
    fin = fetch(t)
    print(f"--- {t} ---")
    print(f"  industry: {fin.company_industry!r}")
    print(f"  sector:   {fin.company_sector!r}")
    print(f"  country:  {fin.company_country!r}")
    print(f"  summary:  {len(fin.company_summary)} chars")
