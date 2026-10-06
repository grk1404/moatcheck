import pandas as pd

df = pd.read_csv("data/sleeve_positions.csv", dtype=str, skipinitialspace=True, usecols=range(21))
df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
df = df[df["Symbol"].astype(str).str.match(r"^[A-Z]{1,5}$", na=False)]

print("Rows after filter:", len(df))
for sym in ("TQQQ", "SQQQ"):
    row = df[df["Symbol"] == sym]
    if not row.empty:
        r = row.iloc[0]
        print(sym, "qty:", r.get("Quantity"), "avg:", r.get("Average cost basis"))
    else:
        print(sym, "NOT FOUND")
