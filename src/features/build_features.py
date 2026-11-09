from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe historical demand features."""

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

    # Calendar features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.day_of_week
    df["day_of_year"] = df["date"].dt.day_of_year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Historical demand features
    group = df.groupby(["store", "item"])["sales"]

    for lag in [7, 14, 28, 365]:
        df[f"lag_{lag}"] = group.shift(lag)

    # Shift before rolling to prevent current-day target leakage
    shifted_sales = df.groupby(["store", "item"])["sales"].shift(1)

    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = (
            shifted_sales.groupby([df["store"], df["item"]])
            .rolling(window)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

    return df


if __name__ == "__main__":
    print("Loading training data...")

    train = pd.read_csv(RAW_DIR / "train.csv")

    print("Original shape:", train.shape)

    print("Generating features...")

    featured = build_features(train)

    print("Feature generation completed.")
    print("Shape before removing missing values:", featured.shape)

    # Initial historical training dataset:
    # Remove rows without the complete required lag/rolling history.
    featured = featured.dropna().reset_index(drop=True)

    print("Shape after removing missing values:", featured.shape)

    output_file = PROCESSED_DIR / "train_features.csv"

    featured.to_csv(output_file, index=False)

    print("\nGenerated features:")
    print(featured.columns.tolist())

    print("\nMissing values:")
    print(featured.isnull().sum())

    print("\nSaved feature dataset to:", output_file)
