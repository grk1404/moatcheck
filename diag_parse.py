import csv
from pathlib import Path

path = Path("data/fidelity_positions.csv")
print(f"File exists: {path.exists()}")
if path.exists():
    print(f"File size:   {path.stat().st_size} bytes")
    print()

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i > 3:
                break
            print(f"--- Row {i} (len={len(row)}) ---")
            for j in range(min(25, len(row))):
                v = row[j]
                if v and str(v).strip():
                    print(f"  col {j:2d}: {repr(v)[:60]}")
            print()
