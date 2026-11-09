from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])

# Create date features
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day_of_week"] = train["date"].dt.day_of_week

# 1. Monthly sales trend
monthly_sales = (
    train.groupby(pd.Grouper(key="date", freq="MS"))["sales"].sum().reset_index()
)

plt.figure(figsize=(14, 5))
sns.lineplot(data=monthly_sales, x="date", y="sales")
plt.title("Monthly Retail Sales Trend (2013–2017)")
plt.xlabel("Date")
plt.ylabel("Total Units Sold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_sales_trend.png", dpi=150)
plt.show()

# 2. Average sales by weekday
weekday_sales = train.groupby("day_of_week")["sales"].mean().reset_index()

weekday_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

weekday_sales["weekday"] = weekday_sales["day_of_week"].map(
    dict(enumerate(weekday_names))
)

plt.figure(figsize=(10, 5))
sns.barplot(data=weekday_sales, x="weekday", y="sales")
plt.title("Average Sales by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average Units Sold")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "weekday_sales.png", dpi=150)
plt.show()

# 3. Average sales by store
store_sales = (
    train.groupby("store")["sales"]
    .mean()
    .reset_index()
    .sort_values("sales", ascending=False)
)

plt.figure(figsize=(10, 5))
sns.barplot(data=store_sales, x="store", y="sales")
plt.title("Average Daily Sales by Store")
plt.xlabel("Store")
plt.ylabel("Average Units Sold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "store_sales.png", dpi=150)
plt.show()

# 4. Top 10 products by total sales
item_sales = train.groupby("item")["sales"].sum().nlargest(10).reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(data=item_sales, x="item", y="sales")
plt.title("Top 10 Products by Total Sales")
plt.xlabel("Item")
plt.ylabel("Total Units Sold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_items.png", dpi=150)
plt.show()

# 5. Sales distribution
plt.figure(figsize=(10, 5))
sns.histplot(data=train, x="sales", bins=50, kde=True)
plt.title("Distribution of Daily Sales")
plt.xlabel("Units Sold")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sales_distribution.png", dpi=150)
plt.show()

# 6. Monthly seasonality across years
monthly_year_sales = train.groupby(["year", "month"])["sales"].mean().unstack()

plt.figure(figsize=(12, 5))
sns.heatmap(monthly_year_sales, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Average Daily Sales by Year and Month")
plt.xlabel("Month")
plt.ylabel("Year")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_seasonality.png", dpi=150)
plt.show()

# Additional checks
print("\n========== ADDITIONAL EDA ==========")
print(
    "Percentage of zero-sales records:",
    round((train["sales"].eq(0).mean()) * 100, 2),
    "%",
)

print("\nTest date range:")
test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
print(test["date"].min(), "to", test["date"].max())

print("\nVisualization files saved in:", OUTPUT_DIR)
