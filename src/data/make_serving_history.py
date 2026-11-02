from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "train.csv"
OUT_FILE = ROOT / "data" / "serving" / "history.csv"

KEEP_DAYS = 400  # lag_365 needs 365; extra days are a safety margin

df = pd.read_csv(RAW_FILE, parse_dates=["date"])
cutoff = df["date"].max() - pd.Timedelta(days=KEEP_DAYS - 1)
out = df[df["date"] >= cutoff].sort_values(["store", "item", "date"])

assert out.groupby(["store", "item"]).size().eq(KEEP_DAYS).all()

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_FILE, index=False)
print(f"Saved {len(out):,} rows to {OUT_FILE}")
