from moatcheck.combined import fetch
fin = fetch("NVDA")
print("company_sector:", repr(fin.company_sector))
print("company_summary (first 200 chars):")
print(fin.company_summary[:200] if fin.company_summary else "(empty)")
