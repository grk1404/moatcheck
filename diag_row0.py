import pandas as pd

df = pd.read_csv("data/sleeve_positions.csv", dtype=str, skipinitialspace=True)
df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]

print("Shape:", df.shape)
print()
print("Row 0 values by column (first 25 columns):")
for i, c in enumerate(df.columns[:25]):
    v = df.iloc[0][c]
    if v is None:
        continue
    print(f"  {i:2d} {c[:35]:35s} = {repr(v)[:60]}")
