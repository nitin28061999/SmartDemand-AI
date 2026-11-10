import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = ROOT / "models/synthetic/xgboost_synthetic_model.joblib"
HISTORY_FILE = ROOT / "data/synthetic/daily_sales.csv"
OUTPUT_DIR = ROOT / "reports/synthetic/recursive_backtest"

VALIDATION_START = pd.Timestamp("2026-08-12")
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


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = predicted - actual

    wape_denominator = np.abs(actual).sum()

    smape_denominator = np.abs(actual) + np.abs(predicted)
    smape_values = np.divide(
        2 * np.abs(errors),
        smape_denominator,
        out=np.zeros_like(errors),
        where=smape_denominator != 0,
    )

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "WAPE_percent": float(
            np.abs(errors).sum() / wape_denominator * 100
        ) if wape_denominator else None,
        "sMAPE_percent": float(np.mean(smape_values) * 100),
    }


def main():  # sourcery skip: list-comprehension, low-code-quality
    print("Loading synthetic model and historical data...")

    model = joblib.load(MODEL_FILE)

    df = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

    if df.duplicated(["store", "item", "date"]).any():
        raise ValueError("Duplicate store-item-date observations found.")

    latest_date = df["date"].max()
    combinations = df.groupby(["store", "item"]).ngroups

    if combinations != 500:
        raise ValueError(
            f"Expected 500 store-item combinations, found {combinations}."
        )

    origins = pd.date_range(
        start=VALIDATION_START,
        end=latest_date - pd.Timedelta(days=6),
        freq="7D",
    )

    if len(origins) == 0:
        raise ValueError("No complete seven-day validation windows available.")

    print("Data source: SYNTHETIC — NOT REAL SALES")
    print("Store-item combinations:", combinations)
    print("Backtest origins:", len(origins))
    print("Forecast horizon:", HORIZON, "days")

    records = []

    grouped = {
        key: group[["date", "sales"]].reset_index(drop=True)
        for key, group in df.groupby(["store", "item"], sort=True)
    }

    for origin_number, origin in enumerate(origins, start=1):
        forecast_dates = pd.date_range(origin, periods=HORIZON)

        print(
            f"\nBacktest {origin_number}/{len(origins)}:",
            origin.date(),
            "to",
            forecast_dates[-1].date(),
        )

        for (store, item), group in grouped.items():
            history = (
                group[group["date"] < origin]
                .copy()
                .reset_index(drop=True)
            )
            history["sales"] = history["sales"].astype("float64")

            actual_window = group[
                group["date"].isin(forecast_dates)
            ]

            if len(history) < 365:
                raise ValueError(
                    f"Insufficient history for store {store}, item {item}"
                )

            if len(actual_window) != HORIZON:
                raise ValueError(
                    f"Incomplete validation window for store {store}, "
                    f"item {item}, origin {origin.date()}"
                )

            actual_lookup = actual_window.set_index("date")["sales"]

            for horizon_day, forecast_date in enumerate(
                forecast_dates, start=1
            ):
                X = create_features(history, forecast_date)

                prediction = max(
                    0.0,
                    float(model.predict(X)[0]),
                )

                actual = float(actual_lookup.loc[forecast_date])

                records.append({
                    "forecast_origin": origin,
                    "date": forecast_date,
                    "horizon_day": horizon_day,
                    "store": store,
                    "item": item,
                    "actual_sales": actual,
                    "predicted_sales": prediction,
                    "data_source": "synthetic",
                })

                # Recursive step: use the prediction, not actual sales.
                history.loc[len(history)] = [
                    forecast_date,
                    prediction,
                ]

    forecasts = pd.DataFrame(records)

    expected_rows = len(origins) * HORIZON * combinations

    if len(forecasts) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, got {len(forecasts)}."
        )

    forecasts["absolute_error"] = (
        forecasts["predicted_sales"] - forecasts["actual_sales"]
    ).abs()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_file = OUTPUT_DIR / "recursive_backtest_predictions.csv"
    origin_file = OUTPUT_DIR / "metrics_by_origin.csv"
    horizon_file = OUTPUT_DIR / "metrics_by_horizon.csv"
    metrics_file = OUTPUT_DIR / "recursive_backtest_metrics.json"

    forecasts.to_csv(detail_file, index=False)

    origin_metrics = [
        {
            "forecast_origin": str(
                origin.date() # pyright: ignore[reportAttributeAccessIssue]
            ),  # pyright: ignore[reportAttributeAccessIssue]
            "forecast_end": str(group["date"].max().date()),
            "records": len(group),
            **calculate_metrics(
                group["actual_sales"],
                group["predicted_sales"],
            ),
        }
        for origin, group in forecasts.groupby("forecast_origin")
    ]
    pd.DataFrame(origin_metrics).to_csv(origin_file, index=False)

    horizon_metrics = [
        {
            "horizon_day": int(horizon_day), # pyright: ignore[reportArgumentType]
            "records": len(group),
            **calculate_metrics(
                group["actual_sales"],
                group["predicted_sales"],
            ),
        }
        for horizon_day, group in forecasts.groupby("horizon_day")
    ]

    pd.DataFrame(horizon_metrics).to_csv(horizon_file, index=False)

    overall_metrics = calculate_metrics(
        forecasts["actual_sales"],
        forecasts["predicted_sales"],
    )

    results = {
        "data_source": "synthetic",
        "disclosure": (
            "All observations are artificially generated for a portfolio "
            "demonstration. These results do not represent real retailer "
            "transactions or real-world forecasting performance."
        ),
        "evaluation_method": (
            "Fixed-model, rolling-origin, recursive seven-day backtest. "
            "Predictions are fed back into history during each window."
        ),
        "model_file": str(MODEL_FILE.relative_to(ROOT)),
        "validation_start": str(VALIDATION_START.date()),
        "latest_actual_date": str(latest_date.date()),
        "forecast_horizon_days": HORIZON,
        "backtest_origins": len(origins),
        "store_item_combinations": combinations,
        "total_predictions": len(forecasts),
        "overall_metrics": overall_metrics,
    }

    with metrics_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n===== RECURSIVE BACKTEST RESULTS =====")
    print("Data source: SYNTHETIC — NOT REAL SALES")
    print("Backtest windows:", len(origins))
    print("Total predictions:", f"{len(forecasts):,}")

    print("\nOverall metrics:")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nSaved files:")
    print(detail_file)
    print(origin_file)
    print(horizon_file)
    print(metrics_file)

    print("\nRECURSIVE SYNTHETIC BACKTEST COMPLETED")


if __name__ == "__main__":
    main()
