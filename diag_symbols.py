import pandas as pd
from pathlib import Path

df = pd.read_csv("data/sleeve_positions.csv", dtype=str, skipinitialspace=True)
df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]

print("Total rows:", len(df))
print()
print("All Symbol values (raw, showing repr so whitespace/quotes are visible):")
for v in df["Symbol"].tolist():
    print("  ", repr(v))
