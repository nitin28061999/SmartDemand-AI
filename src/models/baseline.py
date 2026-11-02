
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)

ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = ROOT / "data" / "raw" / "train.csv"
OUTPUT_DIR = ROOT / "reports" / "baseline"

VALIDATION_DAYS = 90


def calculate_metrics(actual, predicted):
    """Calculate forecasting evaluation metrics."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "RMSLE": np.sqrt(
            mean_squared_log_error(actual, predicted)
        ),
    }


def run_recursive_baseline(history, validation, method):
    """Generate recursive forecasts without using actual future sales."""

    history = deepcopy(history)

    predictions = []

    validation_dates = sorted(validation["date"].unique())
    store_item_pairs = sorted(history.keys())

    for current_date in validation_dates:

        daily_predictions = []

        for store, item in store_item_pairs:

            key = (store, item)
            past_sales = history[key]

            if method == "lag_7":
                prediction = past_sales[-7]

            elif method == "lag_365":
                prediction = past_sales[-365]

            elif method == "rolling_7":
                prediction = np.mean(past_sales[-7:])

            else:
                raise ValueError(f"Unknown forecasting method: {method}")

            prediction = max(0, float(prediction))

            daily_predictions.append({
                "date": current_date,
                "store": store,
                "item": item,
                "prediction": prediction,
            })

            # Add the prediction, not actual sales, to future history.
            history[key].append(prediction)

        predictions.extend(daily_predictions)

    return pd.DataFrame(predictions)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading original training data...")

    df = pd.read_csv(RAW_FILE, parse_dates=["date"])

    print("Original dataset shape:", df.shape)

    # Identify the final 90 days.
    validation_start = (
        df["date"].max()
        - pd.Timedelta(days=VALIDATION_DAYS - 1)
    )

    # Split actual historical data from validation data.
    training = df[df["date"] < validation_start].copy()

    validation = df[df["date"] >= validation_start].copy()

    print("\nVALIDATION PERIOD")
    print("=" * 45)
    print("Training ends:", training["date"].max().date())
    print("Validation starts:", validation_start.date())
    print("Validation ends:", validation["date"].max().date())
    print("Validation rows:", len(validation))

    # Build initial history using only data before validation.
    history = {}

    for (store, item), group in training.groupby(["store", "item"]):

        history[(store, item)] = (
            group.sort_values("date")["sales"]
            .astype(float)
            .tolist()
        )

    print("\nStore-item combinations:", len(history))

    # Verify enough history exists for both lag methods.
    assert all(len(values) >= 365 for values in history.values())

    # Actual validation sales are used only for evaluation.
    (
        validation
        .set_index(["date", "store", "item"])["sales"]
        .to_dict()
    )

    methods = ["lag_7", "lag_365", "rolling_7"]

    prediction_results = {}

    print("\nRUNNING RECURSIVE BASELINES")
    print("=" * 45)

    for method in methods:

        print(f"Running {method}...")

        predictions = run_recursive_baseline(
            history=history,
            validation=validation,
            method=method,
        )

        prediction_results[method] = predictions

    # Create a common evaluation dataset.
    results = validation[
        ["date", "store", "item", "sales"]
    ].copy()

    results = results.rename(columns={"sales": "actual"})

    for method in methods:

        prediction_df = prediction_results[method]

        prediction_df = prediction_df.rename(
            columns={"prediction": f"{method}_prediction"}
        )

        results = results.merge(
            prediction_df,
            on=["date", "store", "item"],
            how="left",
            validate="one_to_one",
        )

    # Calculate evaluation metrics.
    metric_rows = []

    print("\nBASELINE MODEL PERFORMANCE")
    print("=" * 60)

    for method in methods:

        prediction_column = f"{method}_prediction"

        metrics = calculate_metrics(
            results["actual"],
            results[prediction_column],
        )

        metric_rows.append({
            "Model": method,
            **metrics,
        })

        print(f"\n{method.upper()}")
        print(f"MAE:   {metrics['MAE']:.4f}")
        print(f"RMSE:  {metrics['RMSE']:.4f}")
        print(f"RMSLE: {metrics['RMSLE']:.4f}")

    metrics_df = pd.DataFrame(metric_rows)

    # Save predictions and metrics.
    predictions_file = OUTPUT_DIR / "validation_predictions.csv"
    metrics_file = OUTPUT_DIR / "baseline_metrics.csv"

    results.to_csv(predictions_file, index=False)
    metrics_df.to_csv(metrics_file, index=False)

    print("\nFILES SAVED")
    print("=" * 45)
    print("Predictions:", predictions_file)
    print("Metrics:", metrics_file)

    print("\nBaseline evaluation completed successfully.")


if __name__ == "__main__":
    main()
