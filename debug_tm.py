from moatcheck.edgar import _load_facts, _resolve_ticker
cik, _ = _resolve_ticker("TM")
f = _load_facts(cik)["facts"]["ifrs-full"]
for tag in ("NumberOfSharesOutstanding", "AdjustedWeightedAverageShares",
            "WeightedAverageNumberOfOrdinarySharesOutstanding",
            "BasicEarningsLossPerShare"):
    if tag in f:
        units = f[tag]["units"]
        uk = next(iter(units.keys()))
        annual = [e for e in units[uk] if str(e.get("form","")).startswith("20-F")]
        print("==", tag, "==", uk)
        for e in annual[-4:]:
            print("   ", e.get("end"), "->", "{:,.0f}".format(e.get("val", 0)))
    else:
        print("==", tag, "== NOT PRESENT")
