from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "train_features.csv"

VALIDATION_DAYS = 90


def main():
    print("Loading feature dataset...")

    df = pd.read_csv(DATA_FILE, parse_dates=["date"])

    print("Dataset shape:", df.shape)

    # Identify the final 90 calendar days
    validation_start = df["date"].max() - pd.Timedelta(days=VALIDATION_DAYS - 1)

    training = df[df["date"] < validation_start].copy()
    validation = df[df["date"] >= validation_start].copy()

    print("\nCHRONOLOGICAL SPLIT")
    print("=" * 45)

    print("Training start:", training["date"].min().date())
    print("Training end:", training["date"].max().date())

    print("Validation start:", validation["date"].min().date())
    print("Validation end:", validation["date"].max().date())

    print("\nTRAINING DATA")
    print("Rows:", len(training))
    print("Unique dates:", training["date"].nunique())

    print("\nVALIDATION DATA")
    print("Rows:", len(validation))
    print("Unique dates:", validation["date"].nunique())

    print("\nSTORE-ITEM VALIDATION CHECK")

    expected_combinations = df[["store", "item"]].drop_duplicates().shape[0]

    combinations_per_day = validation.groupby("date")[["store", "item"]].apply(
        lambda x: len(x.drop_duplicates())
    )

    print("Expected store-item combinations:", expected_combinations)
    print("Minimum combinations per day:", combinations_per_day.min())
    print("Maximum combinations per day:", combinations_per_day.max())

    assert validation["date"].nunique() == VALIDATION_DAYS
    assert len(validation) == VALIDATION_DAYS * expected_combinations
    assert combinations_per_day.eq(expected_combinations).all()

    print("\nValidation checks passed.")


if __name__ == "__main__":
    main()
