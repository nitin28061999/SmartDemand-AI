import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "synthetic"
OUTPUT_FILE = OUTPUT_DIR / "daily_sales.csv"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

NUMBER_OF_STORES = 10
NUMBER_OF_ITEMS = 50


def generate_synthetic_sales(years: int = 3, seed: int = 42):
    """Generate reproducible simulated daily retail demand."""

    if years < 2:
        raise ValueError("At least two years of history are required.")

    rng = np.random.default_rng(seed)

    end_date = pd.Timestamp(date.today()) - pd.Timedelta(days=1)
    start_date = end_date - pd.DateOffset(years=years) + pd.Timedelta(days=1)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    day_of_week = dates.dayofweek.to_numpy()
    day_of_year = dates.dayofyear.to_numpy()
    elapsed_days = np.arange(len(dates))

    # Shared calendar patterns.
    weekly_pattern = np.array(
        [0.94, 0.97, 1.00, 1.02, 1.08, 1.18, 1.12]
    )

    weekly_effect = weekly_pattern[day_of_week]

    annual_effect = (
        1.0
        + 0.12
        * np.sin(2 * np.pi * (day_of_year - 30) / 365.25)
    )

    growth_effect = 1.0 + 0.08 * (
        elapsed_days / max(len(dates) - 1, 1)
    )

    # Simulated high-demand seasonal periods.
    seasonal_event = np.isin(
        dates.month.to_numpy(),
        [11, 12],
    ).astype(float)

    seasonal_event_effect = 1.0 + 0.10 * seasonal_event

    store_multipliers = np.linspace(0.80, 1.25, NUMBER_OF_STORES)
    item_multipliers = rng.lognormal(
        mean=0.0,
        sigma=0.35,
        size=NUMBER_OF_ITEMS,
    )

    records = []

    for store in range(1, NUMBER_OF_STORES + 1):
        for item in range(1, NUMBER_OF_ITEMS + 1):
            base_demand = (
                45.0
                * store_multipliers[store - 1]
                * item_multipliers[item - 1]
            )

            # Some products have stronger weekend demand than others.
            weekend_sensitivity = rng.uniform(0.85, 1.25)
            store_weekly_effect = weekly_effect.copy()
            store_weekly_effect[5:] *= weekend_sensitivity

            # Random promotional events.
            promotion_days = rng.random(len(dates)) < 0.06
            promotion_effect = np.where(
                promotion_days,
                rng.uniform(1.15, 1.45, len(dates)),
                1.0,
            )

            # Natural demand variation.
            noise = rng.normal(
                loc=1.0,
                scale=0.12,
                size=len(dates),
            )
            noise = np.clip(noise, 0.50, 1.60)

            demand = (
                base_demand
                * store_weekly_effect
                * annual_effect
                * growth_effect
                * seasonal_event_effect
                * promotion_effect
                * noise
            )

            sales = np.maximum(np.rint(demand), 0).astype(int)

            records.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "store": store,
                        "item": item,
                        "sales": sales,
                    }
                )
            )

    result = pd.concat(records, ignore_index=True)
    result = result.sort_values(
        ["date", "store", "item"]
    ).reset_index(drop=True)

    return result, start_date, end_date


def main():
    parser = argparse.ArgumentParser(
        description="Generate clearly labelled synthetic retail demand data."
    )
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating simulated retail demand...")
    print("Years:", args.years)
    print("Random seed:", args.seed)

    df, start_date, end_date = generate_synthetic_sales(
        years=args.years,
        seed=args.seed,
    )

    df.to_csv(OUTPUT_FILE, index=False)

    metadata = {
        "data_source": "synthetic",
        "data_type": "simulated_daily_retail_sales",
        "disclosure": (
            "All sales observations are artificially generated for "
            "a portfolio demonstration. They are not real retailer "
            "transactions or actual customer demand."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "number_of_stores": NUMBER_OF_STORES,
        "number_of_items": NUMBER_OF_ITEMS,
        "records": int(len(df)),
        "frequency": "daily",
        "random_seed": args.seed,
        "simulation_features": [
            "store_and_item_demand_variation",
            "weekday_and_weekend_patterns",
            "annual_seasonality",
            "gradual_demand_growth",
            "simulated_promotional_spikes",
            "random_daily_variation",
        ],
    }

    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nSYNTHETIC DATA GENERATION COMPLETED")
    print("=" * 45)
    print("Output:", OUTPUT_FILE)
    print("Metadata:", METADATA_FILE)
    print("Rows:", f"{len(df):,}")
    print("Stores:", df["store"].nunique())
    print("Items:", df["item"].nunique())
    print("Start date:", df["date"].min().date())
    print("End date:", df["date"].max().date())
    print("Average daily sales:", round(df["sales"].mean(), 2))
    print("Minimum sales:", df["sales"].min())
    print("Maximum sales:", df["sales"].max())
    print("\nData source: SYNTHETIC — NOT REAL SALES")


if __name__ == "__main__":
    main()
