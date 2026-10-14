import time
from data_provider import get_ticker

for sym in ("RACE", "SHEL"):
    print(f"=== {sym} ===")
    info = get_ticker(sym).info or {}
    print(f"  keys count:  {len(info)}")
    print(f"  sector:      {info.get('sector')!r}")
    print(f"  industry:    {info.get('industry')!r}")
    print(f"  country:     {info.get('country')!r}")
    print(f"  summary_len: {len(info.get('longBusinessSummary') or '')}")
    print(f"  currentPrice: {info.get('currentPrice')!r}")
    print(f"  marketCap:    {info.get('marketCap')!r}")
    print(f"  52wHigh:      {info.get('fiftyTwoWeekHigh')!r}")
    print()
    time.sleep(3)  # rate-limit protection
