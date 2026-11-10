from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = ROOT / "models/synthetic/xgboost_synthetic_model.joblib"
HISTORY_FILE = ROOT / "data/synthetic/daily_sales.csv"
OUTPUT_DIR = ROOT / "reports/synthetic/forecasts"

HORIZON = 7

FEATURES = [
    "year", "month", "day", "day_of_week",
    "day_of_year", "week_of_year", "is_weekend",
    "lag_7", "lag_14", "lag_28", "lag_365",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
]


def create_features(history, forecast_date):
    sales = history["sales"]

    values = {
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

    return pd.DataFrame([values], columns=FEATURES)


def main():
    print("Loading model and synthetic history...")

    model = joblib.load(MODEL_FILE)

    df = pd.read_csv(
        HISTORY_FILE,
        parse_dates=["date"],
    )

    records = []
    last_history_date = df["date"].max()

    grouped = df.groupby(["store", "item"], sort=True)

    print("Store-product combinations:", len(grouped))
    print("Forecast horizon:", HORIZON, "days")

    for (store, item), group in grouped:
        history = (
            group[["date", "sales"]]
            .sort_values("date")
            .reset_index(drop=True)
            .copy()
        )
        history["sales"] = history["sales"].astype("float64")

        if len(history) < 365:
            raise ValueError(
                f"Insufficient history for store {store}, item {item}"
            )

        for day in range(1, HORIZON + 1):
            forecast_date = (
                last_history_date + pd.Timedelta(days=day)
            )

            X = create_features(history, forecast_date)

            prediction = max(
                0.0,
                float(model.predict(X)[0]),
            )

            records.append({
                "date": forecast_date,
                "store": store,
                "item": item,
                "predicted_sales": round(prediction, 2),
                "data_source": "synthetic",
            })

            history.loc[len(history)] = [
                forecast_date,
                prediction,
            ]

    forecasts = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_file = OUTPUT_DIR / "all_stores_items_7day.csv"
    summary_file = OUTPUT_DIR / "daily_demand_summary.csv"

    forecasts.to_csv(detail_file, index=False)

    summary = (
        forecasts.groupby("date", as_index=False)
        .agg(
            total_predicted_demand=("predicted_sales", "sum"),
            store_count=("store", "nunique"),
            item_count=("item", "nunique"),
            forecast_records=("predicted_sales", "size"),
        )
    )

    summary["data_source"] = "synthetic"
    summary.to_csv(summary_file, index=False)

    print("\n===== ALL-STORE SYNTHETIC FORECAST =====")
    print("Data source: SYNTHETIC — NOT REAL SALES")
    print("Forecast records:", f"{len(forecasts):,}")
    print("Unique stores:", forecasts["store"].nunique())
    print("Unique items:", forecasts["item"].nunique())
    print("Forecast start:", forecasts["date"].min().date())
    print("Forecast end:", forecasts["date"].max().date())

    print("\n===== DAILY AGGREGATE DEMAND =====")
    print(summary.to_string(index=False))

    print(
        "\nTotal predicted demand:",
        round(forecasts["predicted_sales"].sum(), 2),
    )

    print("\nSaved detailed forecast:", detail_file)
    print("Saved daily summary:", summary_file)
    print("\nALL-STORE RECURSIVE FORECAST COMPLETED")


if __name__ == "__main__":
    main()
