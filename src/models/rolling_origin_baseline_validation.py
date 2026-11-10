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
XGBOOST_DIR = ROOT / "reports" / "rolling_origin"
OUTPUT_DIR = ROOT / "reports" / "rolling_origin_baselines"

VALIDATION_DAYS = 30
EXPECTED_PAIRS = 500

FOLDS = [
    ("fold_1", "2016-04-02"),
    ("fold_2", "2016-07-01"),
    ("fold_3", "2016-10-03"),
]

METHODS = ["lag_7", "lag_365", "rolling_7"]


def calculate_metrics(actual, predicted):
    """Calculate forecasting metrics."""

    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "RMSLE": np.sqrt(
            mean_squared_log_error(actual, predicted)
        ),
    }


def run_recursive_baseline(history, validation_dates, method):
    """Generate recursive predictions without using future actual sales."""

    history = deepcopy(history)
    predictions = []

    for current_date in validation_dates:
        current_date = pd.Timestamp(current_date)

        daily_predictions = []

        for key in sorted(history.keys()):
            past_sales = history[key]

            if method == "lag_7":
                prediction = past_sales[-7]

            elif method == "lag_365":
                prediction = past_sales[-365]

            elif method == "rolling_7":
                prediction = float(np.mean(past_sales[-7:]))

            else:
                raise ValueError(f"Unknown method: {method}")

            daily_predictions.append(
                {
                    "date": current_date,
                    "store": key[0],
                    "item": key[1],
                    "prediction": max(0.0, float(prediction)),
                }
            )

        # Append predictions after all pairs have been predicted.
        for row in daily_predictions:
            key = (row["store"], row["item"])
            history[key].append(row["prediction"])

        predictions.extend(daily_predictions)

    return pd.DataFrame(predictions)


def run_fold(fold_name, validation_start, raw_df):
    """Evaluate all recursive baselines on one validation fold."""

    validation_start = pd.Timestamp(validation_start)
    validation_end = (
        validation_start + pd.Timedelta(days=VALIDATION_DAYS - 1)
    )

    print("\n" + "=" * 60)
    print(f"STARTING BASELINE {fold_name.upper()}")
    print("=" * 60)

    history_data = raw_df[
        raw_df["date"] < validation_start
    ].copy()

    validation = raw_df[
        (raw_df["date"] >= validation_start)
        & (raw_df["date"] <= validation_end)
    ].copy()

    history = {
        (int(store), int(item)): (
            group.sort_values("date")["sales"]
            .astype(float)
            .tolist()
        )
        for (store, item), group in history_data.groupby(
            ["store", "item"]
        )
    }

    if len(history) != EXPECTED_PAIRS:
        raise ValueError(
            f"{fold_name}: expected {EXPECTED_PAIRS} pairs, "
            f"found {len(history)}."
        )

    if any(len(values) < 365 for values in history.values()):
        raise ValueError(
            f"{fold_name}: insufficient history for a baseline."
        )

    validation_dates = sorted(validation["date"].unique())

    if len(validation_dates) != VALIDATION_DAYS:
        raise ValueError(
            f"{fold_name}: expected {VALIDATION_DAYS} dates, "
            f"found {len(validation_dates)}."
        )

    expected_rows = VALIDATION_DAYS * EXPECTED_PAIRS

    if len(validation) != expected_rows:
        raise ValueError(
            f"{fold_name}: expected {expected_rows} actual rows, "
            f"found {len(validation)}."
        )

    results = validation[
        ["date", "store", "item", "sales"]
    ].rename(columns={"sales": "actual"})

    metric_rows = []

    for method in METHODS:
        print(f"Running {method}...")

        prediction_df = run_recursive_baseline(
            history=history,
            validation_dates=validation_dates,
            method=method,
        )

        results = results.merge(
            prediction_df.rename(
                columns={"prediction": f"{method}_prediction"}
            ),
            on=["date", "store", "item"],
            how="left",
            validate="one_to_one",
        )

        prediction_column = f"{method}_prediction"

        if results[prediction_column].isna().any():
            raise ValueError(
                f"{fold_name}: missing predictions for {method}."
            )

        metrics = calculate_metrics(
            results["actual"],
            results[prediction_column],
        )

        metric_rows.append(
            {
                "fold": fold_name,
                "model": method,
                **metrics,
            }
        )

        print(
            f"{method}: "
            f"MAE={metrics['MAE']:.6f}, "
            f"RMSE={metrics['RMSE']:.6f}, "
            f"RMSLE={metrics['RMSLE']:.6f}"
        )

    results.to_csv(
        OUTPUT_DIR / f"{fold_name}_predictions.csv",
        index=False,
    )

    return results, metric_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading original historical data...")

    raw_df = pd.read_csv(
        RAW_FILE,
        parse_dates=["date"],
    )

    if raw_df.duplicated(["date", "store", "item"]).any():
        raise ValueError("Duplicate keys found in raw dataset.")

    all_results = []
    all_metric_rows = []

    for fold_name, start_date in FOLDS:
        results, metric_rows = run_fold(
            fold_name,
            start_date,
            raw_df,
        )

        all_results.append(
            (fold_name, results)
        )
        all_metric_rows.extend(metric_rows)

    # Save all baseline predictions in one combined file.
    combined_baseline = pd.concat(
        [
            results.assign(fold=fold_name)
            for fold_name, results in all_results
        ],
        ignore_index=True,
    )

    combined_baseline.to_csv(
        OUTPUT_DIR / "all_fold_predictions.csv",
        index=False,
    )

    baseline_metrics = pd.DataFrame(all_metric_rows)

    baseline_metrics.to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False,
    )

    # Compare baselines against the already evaluated XGBoost folds.
    comparison_rows = []

    for fold_name, baseline_results in all_results:
        xgb_path = (
            XGBOOST_DIR / f"{fold_name}_predictions.csv"
        )

        xgb_results = pd.read_csv(
            xgb_path,
            parse_dates=["date"],
        )

        keys = ["date", "store", "item"]

        aligned = baseline_results.merge(
            xgb_results[keys + ["actual", "prediction"]],
            on=keys,
            how="inner",
            validate="one_to_one",
            suffixes=("_baseline", "_xgb"),
        )

        if len(aligned) != VALIDATION_DAYS * EXPECTED_PAIRS:
            raise ValueError(
                f"{fold_name}: XGBoost and baseline rows do not align."
            )

        if not np.allclose(
            aligned["actual_baseline"],
            aligned["actual_xgb"],
        ):
            raise ValueError(
                f"{fold_name}: actual values differ between models."
            )

        for method in METHODS:
            metrics = calculate_metrics(
                aligned["actual_baseline"],
                aligned[f"{method}_prediction"],
            )

            comparison_rows.append(
                {
                    "fold": fold_name,
                    "model": method,
                    **metrics,
                }
            )

        xgb_metrics = calculate_metrics(
            aligned["actual_xgb"],
            aligned["prediction"],
        )

        comparison_rows.append(
            {
                "fold": fold_name,
                "model": "XGBoost",
                **xgb_metrics,
            }
        )

    comparison_by_fold = pd.DataFrame(comparison_rows)

    comparison_by_fold.to_csv(
        OUTPUT_DIR / "model_comparison_by_fold.csv",
        index=False,
    )

    # Calculate pooled metrics from all 45,000 aligned observations.
    pooled_rows = []

    for model_name, prediction_column in [
        ("lag_7", "lag_7_prediction"),
        ("lag_365", "lag_365_prediction"),
        ("rolling_7", "rolling_7_prediction"),
    ]:
        metrics = calculate_metrics(
            combined_baseline["actual"],
            combined_baseline[prediction_column],
        )

        pooled_rows.append(
            {
                "model": model_name,
                "validation_rows": len(combined_baseline),
                **metrics,
            }
        )

    xgb_all = pd.concat(
        [
            pd.read_csv(
                XGBOOST_DIR / f"{fold_name}_predictions.csv",
                parse_dates=["date"],
            )
            for fold_name, _ in all_results
        ],
        ignore_index=True,
    )

    if len(xgb_all) != len(combined_baseline):
        raise ValueError("Pooled XGBoost row count does not match.")

    if not np.allclose(
        combined_baseline["actual"].to_numpy(),
        xgb_all["actual"].to_numpy(),
    ):
        raise ValueError("Pooled actual values do not align.")

    xgb_metrics = calculate_metrics(
        xgb_all["actual"],
        xgb_all["prediction"],
    )

    pooled_rows.append(
        {
            "model": "XGBoost",
            "validation_rows": len(xgb_all),
            **xgb_metrics,
        }
    )

    pooled_comparison = pd.DataFrame(pooled_rows).sort_values(
        "MAE"
    )

    pooled_comparison.to_csv(
        OUTPUT_DIR / "pooled_model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("ROLLING-ORIGIN BASELINE EVALUATION COMPLETED")
    print("=" * 60)

    print("\nComparison by fold:")
    print(
        comparison_by_fold.to_string(index=False)
    )

    print("\nPooled model comparison:")
    print(
        pooled_comparison.to_string(index=False)
    )

    print("\nSaved reports:")
    for filename in [
        "fold_1_predictions.csv",
        "fold_2_predictions.csv",
        "fold_3_predictions.csv",
        "all_fold_predictions.csv",
        "fold_metrics.csv",
        "model_comparison_by_fold.csv",
        "pooled_model_comparison.csv",
    ]:
        print(OUTPUT_DIR / filename)


if __name__ == "__main__":
    main()
