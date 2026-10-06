import csv

path = "data/sleeve_positions.csv"

with open(path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i > 4:
            break
        print(f"--- Row {i} (len={len(row)}) ---")
        for j, v in enumerate(row[:22]):
            if v is not None and str(v).strip():
                print(f"  col {j:2d}: {repr(v)[:70]}")
        print()
