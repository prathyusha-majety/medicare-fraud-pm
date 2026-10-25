import os
from pathlib import Path
DATA_DIR = Path(os.environ.get("DATA_DIR", "/scratch/sm3664/medicare-fraud"))
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"
RESULTS = DATA_DIR / "results"
LOGS = DATA_DIR / "logs"
