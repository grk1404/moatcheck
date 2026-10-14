import csv
from pathlib import Path

path = Path("data/fidelity_positions.csv")
rows = []
with path.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        if not row or len(row) < 15:
            continue
        sym = (row[2] or "").strip()
        if not sym or "*" in sym:
            continue
        if not sym.replace(".", "").replace("-", "").isalpha():
            continue
        if len(sym) > 6:
            continue
        try:
            qty = float((row[4] or "0").replace(",", "").strip())
        except Exception:
            qty = None
        try:
            avg = float((row[14] or "0").replace("$", "").replace(",", "").strip())
        except Exception:
            avg = None
        rows.append((sym, qty, avg))

print(f"Parsed {len(rows)} equity positions:")
for sym, qty, avg in rows[:10]:
    q = f"{qty:>10,.3f}" if qty is not None else "n/a"
    a = f"${avg:>8,.2f}" if avg is not None else "n/a"
    print(f"  {sym:8s} qty={q}  avg={a}")
