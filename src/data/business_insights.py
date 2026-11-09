from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
OUTPUT_PATH = ROOT / "reports" / "eda" / "business_insights.txt"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

lines = []


def add(text=""):
    lines.append(str(text))


add("SMARTDEMAND AI - BUSINESS INSIGHTS REPORT")
add("=" * 55)

# Overall performance
add("\n1. OVERALL SALES")
add(f"Total units sold: {df['sales'].sum():,}")
add(f"Average daily store-item demand: {df['sales'].mean():.2f}")
add(f"Median daily store-item demand: {df['sales'].median():.2f}")

# Store performance
store_sales = (
    df.groupby("store")["sales"]
    .agg(["mean", "sum"])
    .sort_values("sum", ascending=False)
)

add("\n2. STORE PERFORMANCE")
add(store_sales.to_string())

add(f"\nHighest-selling store: Store {store_sales['sum'].idxmax()}")
add(f"Lowest-selling store: Store {store_sales['sum'].idxmin()}")

# Item performance
item_sales = df.groupby("item")["sales"].sum().sort_values(ascending=False)

add("\n3. PRODUCT PERFORMANCE")
add(f"Top-selling item: Item {item_sales.idxmax()}")
add(f"Top item units sold: {item_sales.max():,}")
add(f"Lowest-selling item: Item {item_sales.idxmin()}")

# Monthly performance
df["month"] = df["date"].dt.month
monthly = df.groupby("month")["sales"].mean()

add("\n4. MONTHLY SEASONALITY")
add(monthly.to_string())
add(f"Highest average-sales month: {monthly.idxmax()}")
add(f"Lowest average-sales month: {monthly.idxmin()}")

# Weekday performance
df["weekday"] = df["date"].dt.day_of_week
weekday = df.groupby("weekday")["sales"].mean()

weekday_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

add("\n5. WEEKDAY PERFORMANCE")
for day, value in weekday.items():
    add(f"{weekday_names[day]}: {value:.2f}")  # pyright: ignore[reportCallIssue, reportArgumentType]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print(f"\nReport saved to: {OUTPUT_PATH}")
