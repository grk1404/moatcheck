import pandas as pd

# Read with header=None so we see the physical layout exactly
df = pd.read_csv("data/sleeve_positions.csv", dtype=str, skipinitialspace=True, header=None)

print("Shape:", df.shape)
print()
for i in range(min(3, len(df))):
    row = df.iloc[i]
    print(f"--- Row {i} ---")
    for j in range(min(22, df.shape[1])):
        v = row[j]
        if v is not None and str(v).strip() and str(v) != "nan":
            print(f"  col {j:2d}: {repr(v)[:60]}")
    print()
