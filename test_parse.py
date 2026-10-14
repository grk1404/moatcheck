import csv
from pathlib import Path

path = Path("data/fidelity_positions.csv")
rows = []
with path.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        if not row or len(row) < 20:
            continue
        sym = (row[2] or "").strip()
        if not sym or not sym.replace(".", "").isalpha() or len(sym) > 5:
            continue
        try:
            qty = float((row[18] or "0").replace(",", "").strip())
            avg = float((row[19] or "0").replace("$", "").replace(",", "").strip())
        except Exception:
            continue
        rows.append((sym, qty, avg))

print(f"Parsed {len(rows)} positions:")
for r in rows[:10]:
    print(f"  {r[0]:8s} qty={r[1]:>12,.3f}  avg_cost=${r[2]:>8,.2f}")
