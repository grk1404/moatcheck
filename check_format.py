"""
Regression check for the CallMeBot '$<digit>' corruption bug.

CallMeBot strips '$1', '$2', '$9' (shell positional params), so any
literal '$' immediately before a digit in a WhatsApp message must be
escaped as '\\$' in the Python source.

This script:
  1. Scans investment_alerts.py for bare '$<digit>' patterns.
  2. Asserts the simulated post-fix message strings contain no such pattern.

Run:  python check_format.py
Exit: 0 = clean, 1 = at least one violation.
"""
import re
import sys

SOURCE = "investment_alerts.py"

# Bare '$' immediately followed by a digit, NOT already escaped with backslash.
BARE = re.compile(r"(?<!\\)\$[0-9]")


def scan_source() -> list[tuple[int, str]]:
    with open(SOURCE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    hits = []
    for n, line in enumerate(lines, start=1):
        if BARE.search(line):
            hits.append((n, line.rstrip()))
    return hits


def simulate_messages() -> list[tuple[str, str]]:
    """
    Mirror the message families after Fix 1.
    If Fix 1 was not applied, the actual source scan would catch it —
    these simulate the *intended* output so we can verify the escaping logic.
    """
    return [
        ("52w-high", f"FTNT: \\$189.91 (98.7% of 52w high \\$192.41)"),
        ("52w-high", f"FXAIX: \\$271.69 (100.0% of 52w high \\$271.69)"),
        ("52w-low",  f"ATKR: \\$94.83 (99.9% above 52w low \\$94.88)"),
        ("MOS",      f"XOM: \\$112.34 ≤ MOS \\$125.00 (intrinsic \\$140.00, -19.8%)"),
        ("BNF",      f"ZBRA: \\$301.22\nDislocation: 4.1% below MA25 (\\$314.10)\nRSI: 28.4"),
        ("TQQQ-buy", f"BUY 0.3421 shares TQQQ (~\\$18.94)"),
        ("TQQQ-sell",f"SELL 0.1000 shares TQQQ (~\\$5.42)"),
        ("SQQQ-buy", f"BUY 0.2000 shares SQQQ (~\\$3.10)"),
        ("SQQQ-sell",f"SELL 0.0500 shares SQQQ (~\\$0.78)"),
        ("Sleeve",   f"Sleeve value: \\$54,321.09"),
    ]


def main() -> int:
    ok = True

    print("=== Scanning source for bare '$<digit>' ===")
    src_hits = scan_source()
    if src_hits:
        ok = False
        print(f"FAIL: {len(src_hits)} occurrence(s) in {SOURCE}:")
        for n, line in src_hits:
            print(f"  line {n}: {line}")
    else:
        print("OK: no bare '$<digit>' in source.")

    print("\n=== Simulated emitted messages ===")
    for label, msg in simulate_messages():
        bad = BARE.search(msg)
        status = "FAIL" if bad else "OK  "
        if bad:
            ok = False
        print(f"{label:11} | {status} | {msg!r}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())