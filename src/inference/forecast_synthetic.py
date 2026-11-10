import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = ROOT / "models" / "synthetic" / "xgboost_synthetic_model.joblib"
HISTORY_FILE = ROOT / "data" / "synthetic" / "daily_sales.csv"
OUTPUT_DIR = ROOT / "reports" / "synthetic" / "forecasts"

FEATURES = [
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


def create_features(history, forecast_date):
    """Build features using only historical or previously predicted sales."""

    sales = history["sales"]

    row = {
        "year": forecast_date.year,
        "month": forecast_date.month,
        "day": forecast_date.day,
        "day_of_week": forecast_date.day_of_week,
        "day_of_year": forecast_date.day_of_year,
        "week_of_year": int(forecast_date.isocalendar().week),
        "is_weekend": int(forecast_date.day_of_week >= 5),
        "lag_7": sales.iloc[-7],
        "lag_14": sales.iloc[-14],
        "lag_28": sales.iloc[-28],
        "lag_365": sales.iloc[-365],
        "rolling_mean_7": sales.iloc[-7:].mean(),
        "rolling_mean_14": sales.iloc[-14:].mean(),
        "rolling_mean_30": sales.iloc[-30:].mean(),
    }

    return pd.DataFrame([row], columns=FEATURES)


def forecast_demand(store, item, horizon):
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

    if not HISTORY_FILE.exists():
        raise FileNotFoundError(f"History not found: {HISTORY_FILE}")

    model = joblib.load(MODEL_FILE)

    df = pd.read_csv(
        HISTORY_FILE,
        parse_dates=["date"],
    )

    history = df[
        (df["store"] == store)
        & (df["item"] == item)
    ][["date", "sales"]].copy()

    history = history.sort_values("date").reset_index(drop=True)

    if len(history) < 365:
        raise ValueError("At least 365 days of history are required.")

    if history["date"].duplicated().any():
        raise ValueError("Duplicate dates found in selected history.")

    expected_dates = pd.date_range(
        start=history["date"].min(),
        end=history["date"].max(),
        freq="D",
    )

    if not history["date"].reset_index(drop=True).equals(
        pd.Series(expected_dates, name="date")
    ):
        raise ValueError("History must contain continuous daily observations.")

    last_history_date = history["date"].max()
    predictions = []

    for day in range(1, horizon + 1):
        forecast_date = last_history_date + pd.Timedelta(days=day)

        X = create_features(history, forecast_date)

        prediction = float(model.predict(X)[0])
        prediction = max(0.0, prediction)

        predictions.append({
            "date": forecast_date,
            "store": store,
            "item": item,
            "predicted_sales": round(prediction, 2),
            "data_source": "synthetic",
        })

        # Append this prediction so the next forecast uses it.
        history = pd.concat(
            [
                history,
                pd.DataFrame({
                    "date": [forecast_date],
                    "sales": [prediction],
                }),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(predictions), last_history_date


def main():
    parser = argparse.ArgumentParser(
        description="Recursive forecasting using synthetic retail demand data."
    )
    parser.add_argument("--store", type=int, default=1)
    parser.add_argument("--item", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=7)

    args = parser.parse_args()

    if not 1 <= args.store <= 10:
        parser.error("Store must be between 1 and 10.")

    if not 1 <= args.item <= 50:
        parser.error("Item must be between 1 and 50.")

    if not 1 <= args.horizon <= 90:
        parser.error("Horizon must be between 1 and 90 days.")

    print("Loading synthetic model and history...")

    forecasts, last_history_date = forecast_demand(
        store=args.store,
        item=args.item,
        horizon=args.horizon,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = (
        OUTPUT_DIR
        / f"forecast_store_{args.store}_item_{args.item}.csv"
    )

    forecasts.to_csv(output_file, index=False)

    print("\n===== RECURSIVE DEMAND FORECAST =====")
    print("Data source: SYNTHETIC — NOT REAL SALES")
    print("Store:", args.store)
    print("Item:", args.item)
    print("Last historical date:", last_history_date.date())
    print("Forecast horizon:", args.horizon, "days")
    print("First forecast date:", forecasts["date"].min().date())
    print("Last forecast date:", forecasts["date"].max().date())

    print("\nForecast:")
    print(
        forecasts[["date", "predicted_sales"]].to_string(
            index=False
        )
    )

    print("\nTotal predicted demand:",
          round(forecasts["predicted_sales"].sum(), 2))

    print("Saved forecast:", output_file)
    print("\nRECURSIVE FORECAST COMPLETED")


if __name__ == "__main__":
    main()
