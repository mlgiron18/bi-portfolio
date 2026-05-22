import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "openpyxl", "-q"])

import pandas as pd

df = pd.read_excel("Typeform Responses 5.21.xlsx")
print("COLUMNS:")
for i, col in enumerate(df.columns):
    print(f"  [{i}] {col}")
print(f"\nTotal rows: {len(df)}")
print("\nFirst row preview:")
print(df.iloc[0])
