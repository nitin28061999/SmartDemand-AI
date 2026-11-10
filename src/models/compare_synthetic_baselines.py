
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

HISTORY_FILE = ROOT / "data/synthetic/daily_sales.csv"
XGBOOST_FILE = (
    ROOT
    / "reports/synthetic/recursive_backtest/"
    / "recursive_backtest_predictions.csv"
)
OUTPUT_DIR = ROOT / "reports/synthetic/baseline_comparison"

HORIZON = 7
EXPECTED_COMBINATIONS = 500
EXPECTED_XGB_ROWS = 42_000

KEYS = [
    "forecast_origin",
    "date",
    "horizon_day",
    "store",
    "item",
]


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = predicted - actual
    denominator = np.abs(actual).sum()

    smape_denominator = np.abs(actual) + np.abs(predicted)
    smape_values = np.divide(
        2 * np.abs(errors),
        smape_denominator,
        out=np.zeros_like(errors),
        where=smape_denominator != 0,
    )

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "WAPE_percent": (
            float(np.abs(errors).sum() / denominator * 100)
            if denominator
            else None
        ),
        "sMAPE_percent": float(np.mean(smape_values) * 100),
    }


def main():  # sourcery skip: low-code-quality
    print("Loading synthetic history and XGBoost backtest...")

    history_df = pd.read_csv(
        HISTORY_FILE,
        parse_dates=["date"],
    )

    xgb_df = pd.read_csv(
        XGBOOST_FILE,
        parse_dates=["forecast_origin", "date"],
    )

    if len(xgb_df) != EXPECTED_XGB_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_XGB_ROWS} XGBoost predictions, "
            f"found {len(xgb_df)}."
        )

    if xgb_df.duplicated(KEYS).any():
        raise ValueError("Duplicate XGBoost backtest keys found.")

    if history_df.duplicated(
        ["store", "item", "date"]
    ).any():
        raise ValueError("Duplicate historical observations found.")

    combinations = history_df.groupby(
        ["store", "item"]
    ).ngroups

    if combinations != EXPECTED_COMBINATIONS:
        raise ValueError(
            f"Expected {EXPECTED_COMBINATIONS} combinations, "
            f"found {combinations}."
        )

    origins = sorted(xgb_df["forecast_origin"].unique())

    # Store each combination as NumPy arrays for efficient processing.
    grouped = {}

    for key, group in history_df.groupby(
        ["store", "item"],
        sort=True,
    ):
        group = group.sort_values("date")

        grouped[key] = {
            "dates": group["date"].to_numpy(),
            "sales": group["sales"].to_numpy(dtype=float),
        }

    print("Data source: SYNTHETIC — NOT REAL SALES")
    print("Backtest origins:", len(origins))
    print("Store-item combinations:", combinations)
    print("Forecast horizon:", HORIZON)

    records = []

    for origin_number, origin in enumerate(origins, start=1):
        origin = pd.Timestamp(origin)
        forecast_dates = pd.date_range(
            origin,
            periods=HORIZON,
        )

        print(
            f"Comparison window {origin_number}/{len(origins)}:",
            origin.date(),
            "to",
            forecast_dates[-1].date(),
        )

        for (store, item), group in grouped.items():
            dates = group["dates"]
            sales = group["sales"]

            history_mask = dates < np.datetime64(origin)

            # Each baseline gets its own independent history.
            seasonal_history = sales[history_mask].tolist()
            moving_history = sales[history_mask].tolist()

            actual_mask = np.isin(
                dates,
                forecast_dates.to_numpy(dtype="datetime64[ns]"),
            )

            actual_dates = dates[actual_mask]
            actual_values = sales[actual_mask]

            if len(seasonal_history) < 365:
                raise ValueError(
                    f"Insufficient history for store {store}, "
                    f"item {item}."
                )

            if len(actual_values) != HORIZON:
                raise ValueError(
                    f"Incomplete validation window for store {store}, "
                    f"item {item}, origin {origin.date()}."
                )

            actual_lookup = dict(zip(actual_dates, actual_values))

            for horizon_day, forecast_date in enumerate(
                forecast_dates,
                start=1,
            ):
                actual = float(
                    actual_lookup[np.datetime64(forecast_date)]
                )

                # Baseline 1: same weekday from previous week.
                seasonal_prediction = max(
                    0.0,
                    float(seasonal_history[-7]),
                )

                # Baseline 2: trailing seven-day moving average.
                moving_prediction = max(
                    0.0,
                    float(np.mean(moving_history[-7:])),
                )

                common = {
                    "forecast_origin": origin,
                    "date": forecast_date,
                    "horizon_day": horizon_day,
                    "store": store,
                    "item": item,
                    "actual_sales": actual,
                    "data_source": "synthetic",
                }

                records.extend(
                    (
                        {
                            **common,
                            "model": "seasonal_naive_lag_7",
                            "predicted_sales": seasonal_prediction,
                        },
                        {
                            **common,
                            "model": "moving_average_7",
                            "predicted_sales": moving_prediction,
                        },
                    )
                )
                # Feed each model's prediction into its own history.
                seasonal_history.append(seasonal_prediction)
                moving_history.append(moving_prediction)

    baseline_df = pd.DataFrame(records)

    # Verify that baseline actual values match the XGBoost backtest.
    actual_check = baseline_df.merge(
        xgb_df[KEYS + ["actual_sales"]],
        on=KEYS,
        how="left",
        suffixes=("", "_xgb"),
        validate="many_to_one",
    )

    if actual_check["actual_sales_xgb"].isna().any():
        raise ValueError("Some baseline observations did not match XGBoost.")

    if not np.allclose(
        actual_check["actual_sales"],
        actual_check["actual_sales_xgb"],
    ):
        raise ValueError("Actual values differ between model evaluations.")

    # Reuse existing XGBoost predictions. Do not retrain.
    xgb_long = xgb_df[
        KEYS + ["actual_sales", "predicted_sales"]
    ].copy()

    xgb_long["model"] = "xgboost"
    xgb_long["data_source"] = "synthetic"

    comparison = pd.concat(
        [baseline_df, xgb_long],
        ignore_index=True,
    )

    expected_rows = EXPECTED_XGB_ROWS * 3

    if len(comparison) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} comparison rows, "
            f"found {len(comparison)}."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_file = OUTPUT_DIR / "baseline_comparison_predictions.csv"
    overall_file = OUTPUT_DIR / "metrics_overall.csv"
    horizon_file = OUTPUT_DIR / "metrics_by_horizon.csv"
    origin_file = OUTPUT_DIR / "metrics_by_origin.csv"
    summary_file = OUTPUT_DIR / "baseline_comparison_summary.json"

    comparison.to_csv(detail_file, index=False)

    overall_metrics = [
        {
            "model": model_name,
            "records": len(group),
            **calculate_metrics(
                group["actual_sales"],
                group["predicted_sales"],
            ),
        }
        for model_name, group in comparison.groupby("model")
    ]
    overall_df = pd.DataFrame(overall_metrics).sort_values(
        "WAPE_percent"
    )
    overall_df.to_csv(overall_file, index=False)

    horizon_metrics = [
        {
            "model": model_name,
            "horizon_day": int(horizon),  # pyright: ignore[reportArgumentType]
            "records": len(group),
            **calculate_metrics(
                group["actual_sales"],
                group["predicted_sales"],
            ),
        }
        for (model_name, horizon), group in comparison.groupby(
            ["model", "horizon_day"]
        )
    ]
    pd.DataFrame(horizon_metrics).to_csv(
        horizon_file,
        index=False,
    )

    origin_metrics = [
        {
            "model": model_name,
            "forecast_origin": origin.date().isoformat(), # pyright: ignore[reportAttributeAccessIssue]
            "records": len(group),
            **calculate_metrics(
                group["actual_sales"],
                group["predicted_sales"],
            ),
        }
        for (model_name, origin), group in comparison.groupby(
            ["model", "forecast_origin"]
        )
    ]
    pd.DataFrame(origin_metrics).to_csv(
        origin_file,
        index=False,
    )

    summary = {
        "data_source": "synthetic",
        "disclosure": (
            "All observations are artificially generated for a "
            "portfolio demonstration. Results do not represent "
            "real retailer transactions or real-world performance."
        ),
        "evaluation_method": (
            "Same rolling origins, seven-day recursive horizon, "
            "store-item combinations, and actual observations."
        ),
        "backtest_origins": len(origins),
        "store_item_combinations": combinations,
        "forecast_horizon_days": HORIZON,
        "predictions_per_model": EXPECTED_XGB_ROWS,
        "models": [
            "xgboost",
            "seasonal_naive_lag_7",
            "moving_average_7",
        ],
        "overall_metrics": overall_metrics,
    }

    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n===== BASELINE COMPARISON RESULTS =====")
    print(overall_df.to_string(index=False))

    print("\nSaved files:")
    print(detail_file)
    print(overall_file)
    print(horizon_file)
    print(origin_file)
    print(summary_file)

    print("\nDATA SOURCE: SYNTHETIC — NOT REAL SALES")
    print("BASELINE COMPARISON COMPLETED")
    print("BASELINE COMPARISON COMPLETED")


if __name__ == "__main__":
    main()
