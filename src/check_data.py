import platform
import pandas as pd
from config import RAW

print("Python", platform.python_version(), "| pandas", pd.__version__)
part_b = RAW / "part_b"
folders = sorted(p for p in part_b.iterdir() if p.is_dir())
print(f"{len(folders)} Part B folders found")

f = next((part_b / "Medicare_Physician_Other_Practitioners_by_Provider_2023").glob("*.csv"))
df = pd.read_csv(f, nrows=1000, low_memory=False)
print(f.name, "->", df.shape)
print(df.columns.tolist()[:10])
