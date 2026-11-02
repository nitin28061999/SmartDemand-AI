
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
train = pd.read_csv(RAW_DATA / "train.csv", parse_dates=["date"])
test = pd.read_csv(RAW_DATA / "test.csv", parse_dates=["date"])

print("\n========== DATASET OVERVIEW ==========")
print("Training shape:", train.shape)
print("Testing shape:", test.shape)

print("\nTraining columns:")
print(train.columns.tolist())

print("\nTesting columns:")
print(test.columns.tolist())

# Data types and missing values
print("\n========== DATA TYPES ==========")
print(train.dtypes)

print("\n========== MISSING VALUES ==========")
print("Training:\n", train.isnull().sum())
print("\nTesting:\n", test.isnull().sum())

# Duplicate checks
print("\n========== DUPLICATE CHECK ==========")
print("Duplicate training rows:", train.duplicated().sum())
print("Duplicate test rows:", test.duplicated().sum())

# Date coverage
print("\n========== DATE ANALYSIS ==========")
print("Start date:", train["date"].min())
print("End date:", train["date"].max())
print("Unique dates:", train["date"].nunique())

# Store and item coverage
print("\n========== STORE AND ITEM ANALYSIS ==========")
print("Unique stores:", train["store"].nunique())
print("Unique items:", train["item"].nunique())

print("\nSales by store:")
print(train.groupby("store")["sales"].agg(["count", "mean", "sum"]))

print("\nTop 10 items by total sales:")
print(
    train.groupby("item")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# Sales statistics
print("\n========== SALES DISTRIBUTION ==========")
print(train["sales"].describe())

# Date-based sales trends
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day_of_week"] = train["date"].dt.day_of_week

monthly_sales = (
    train.groupby(["year", "month"])["sales"]
    .sum()
    .reset_index()
)

weekly_sales = (
    train.groupby("day_of_week")["sales"]
    .mean()
    .reset_index()
)

# Save initial analysis outputs
monthly_sales.to_csv(OUTPUT_DIR / "monthly_sales.csv", index=False)
weekly_sales.to_csv(OUTPUT_DIR / "weekly_sales.csv", index=False)

print("\n========== EDA COMPLETE ==========")
print("Monthly sales saved to:", OUTPUT_DIR / "monthly_sales.csv")
print("Weekly sales saved to:", OUTPUT_DIR / "weekly_sales.csv")
