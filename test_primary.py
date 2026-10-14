from data_provider import get_ticker

for sym in ("RACE", "RACE.MI", "SHEL", "SHEL.L", "HSBC", "MUFG"):
    try:
        info = get_ticker(sym).info or {}
        print(f"{sym:10s} sector={info.get('sector')!r:20s} country={info.get('country')!r:20s} summary_len={len(info.get('longBusinessSummary') or '')}")
    except Exception as e:
        print(f"{sym:10s} ERROR: {e}")
