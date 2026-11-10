from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "synthetic" / "daily_sales.csv"
OUTPUT_DIR = ROOT / "data" / "synthetic" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "features.csv"

FEATURE_COLUMNS = [
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_365",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe features for synthetic daily demand."""

    df = df.sort_values(
        ["store", "item", "date"]
    ).reset_index(drop=True)

    # Calendar features.
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.day_of_week
    df["day_of_year"] = df["date"].dt.day_of_year
    df["week_of_year"] = (
        df["date"].dt.isocalendar().week.astype(int)
    )
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Historical demand features, calculated separately
    # for each store-product combination.
    grouped = df.groupby(
        ["store", "item"],
        sort=False,
    )["sales"]

    for lag in [7, 14, 28, 365]:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # Shift by one day before calculating rolling averages.
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = grouped.transform(
            lambda series: (
                series.shift(1)
                .rolling(window=window, min_periods=window)
                .mean()
            )
        )

    # The first 365 days cannot have a complete lag_365.
    df = df.dropna(
        subset=FEATURE_COLUMNS
    ).reset_index(drop=True)

    return df


def main():
    print("Loading synthetic daily sales data...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    print("Input rows:", f"{len(df):,}")

    print("\nBuilding calendar and historical features...")

    features = build_features(df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n===== SYNTHETIC FEATURE ENGINEERING =====")
    print("Output:", OUTPUT_FILE)
    print("Rows before feature engineering:", f"{len(df):,}")
    print("Rows after feature engineering:", f"{len(features):,}")
    print("Features created:", len(FEATURE_COLUMNS))
    print("Stores:", features["store"].nunique())
    print("Items:", features["item"].nunique())
    print("Date range:", features["date"].min().date(),
          "to", features["date"].max().date())
    print("Missing feature values:",
          features[FEATURE_COLUMNS].isna().sum().sum())

    print("\nFeature columns:")
    for column in FEATURE_COLUMNS:
        print(" -", column)

    assert features[FEATURE_COLUMNS].isna().sum().sum() == 0
    assert len(features) > 0

    print("\nSYNTHETIC FEATURE ENGINEERING COMPLETED")


if __name__ == "__main__":
    main()
