from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = ROOT / "data" / "processed" / "train_features.csv"
RAW_FILE = ROOT / "data" / "raw" / "train.csv"

REPORT_DIR = ROOT / "reports" / "rolling_origin"

VALIDATION_DAYS = 30
EXPECTED_PAIRS = 500

FOLDS = [
    ("fold_1", "2016-04-02"),
    ("fold_2", "2016-07-01"),
    ("fold_3", "2016-10-03"),
]

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


def create_features(current_date, history):
    """Create recursive forecasting features from historical sales."""

    if len(history) < 365:
        raise ValueError(
            f"Insufficient history: expected 365 values, got {len(history)}"
        )

    return {
        "year": current_date.year,
        "month": current_date.month,
        "day": current_date.day,
        "day_of_week": current_date.day_of_week,
        "day_of_year": current_date.day_of_year,
        "week_of_year": int(current_date.isocalendar().week),
        "is_weekend": int(current_date.day_of_week >= 5),
        "lag_7": history[-7],
        "lag_14": history[-14],
        "lag_28": history[-28],
        "lag_365": history[-365],
        "rolling_mean_7": float(np.mean(history[-7:])),
        "rolling_mean_14": float(np.mean(history[-14:])),
        "rolling_mean_30": float(np.mean(history[-30:])),
    }


def calculate_metrics(actual, predicted):
    """Calculate regression metrics with nonnegative predictions."""

    predicted = np.maximum(np.asarray(predicted), 0)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "RMSLE": np.sqrt(
            mean_squared_log_error(actual, predicted)
        ),
    }


def build_model():
    """Build XGBoost with the existing project configuration."""

    return XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def run_fold(
    fold_name,
    validation_start,
    features_df,
    raw_df,
):
    """Train and recursively evaluate one chronological fold."""

    validation_start = pd.Timestamp(validation_start)
    validation_end = (
        validation_start + pd.Timedelta(days=VALIDATION_DAYS - 1)
    )

    print("\n" + "=" * 65)
    print(f"STARTING {fold_name.upper()}")
    print("=" * 65)
    print("Validation start:", validation_start.date())
    print("Validation end:", validation_end.date())

    # Only engineered observations strictly before the fold are used.
    training = features_df[
        features_df["date"] < validation_start
    ].copy()

    if training.empty:
        raise ValueError(f"{fold_name}: training dataset is empty.")

    if training["date"].max() >= validation_start:
        raise ValueError(f"{fold_name}: training boundary violation.")

    print("Training rows:", len(training))
    print("Training end:", training["date"].max().date())

    X_train = training[FEATURES]
    y_train = training["sales"]

    model = build_model()

    print("Training XGBoost...")
    model.fit(X_train, y_train)
    print("Training completed.")

    # Initialize recursive history using observations before validation.
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

    short_histories = [
        key for key, values in history.items()
        if len(values) < 365
    ]

    if short_histories:
        raise ValueError(
            f"{fold_name}: {len(short_histories)} pairs "
            "have insufficient history."
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
            f"{fold_name}: expected {expected_rows} validation rows, "
            f"found {len(validation)}."
        )

    predictions = []

    print("Starting recursive forecasting...")

    for day_number, current_date in enumerate(validation_dates, 1):
        current_date = pd.Timestamp(current_date)

        daily_rows = []
        daily_keys = []

        for key in sorted(history.keys()):
            row = create_features(
                current_date,
                history[key],
            )

            daily_rows.append(row)
            daily_keys.append(key)

        X_day = pd.DataFrame(
            daily_rows,
            columns=FEATURES,
        )

        daily_predictions = np.maximum(
            model.predict(X_day),
            0,
        )

        if len(daily_predictions) != EXPECTED_PAIRS:
            raise ValueError(
                f"{fold_name}: incorrect prediction count "
                f"on {current_date.date()}."
            )

        # Update histories only after every pair has been predicted.
        for key, prediction in zip(
            daily_keys,
            daily_predictions,
        ):
            history[key].append(float(prediction))

            predictions.append(
                {
                    "date": current_date,
                    "store": key[0],
                    "item": key[1],
                    "prediction": float(prediction),
                }
            )

        if day_number % 10 == 0:
            print(
                f"Completed {day_number}/{VALIDATION_DAYS} days"
            )

    prediction_df = pd.DataFrame(predictions)

    actuals = validation[
        ["date", "store", "item", "sales"]
    ].rename(columns={"sales": "actual"})

    results = actuals.merge(
        prediction_df,
        on=["date", "store", "item"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not (results["_merge"] == "both").all():
        raise ValueError(
            f"{fold_name}: missing or unmatched predictions."
        )

    results = results.drop(columns=["_merge"])

    if len(results) != expected_rows:
        raise ValueError(
            f"{fold_name}: expected {expected_rows} result rows, "
            f"found {len(results)}."
        )

    if not np.isfinite(results["prediction"]).all():
        raise ValueError(
            f"{fold_name}: non-finite predictions detected."
        )

    if (results["prediction"] < 0).any():
        raise ValueError(
            f"{fold_name}: negative predictions detected."
        )

    metrics = calculate_metrics(
        results["actual"],
        results["prediction"],
    )

    print(f"\n{fold_name.upper()} RESULTS")
    print("-" * 40)

    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    print("Prediction rows:", len(results))

    results.to_csv(
        REPORT_DIR / f"{fold_name}_predictions.csv",
        index=False,
    )

    return {
        "fold": fold_name,
        "validation_start": validation_start.date(),
        "validation_end": validation_end.date(),
        "training_rows": len(training),
        "validation_rows": len(results),
        **metrics,
    }, results


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading engineered features...")
    features_df = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["date"],
    )

    print("Loading original historical sales...")
    raw_df = pd.read_csv(
        RAW_FILE,
        parse_dates=["date"],
    )

    if features_df[FEATURES + ["sales"]].isna().any().any():
        raise ValueError("Feature dataset contains missing values.")

    if raw_df.duplicated(["date", "store", "item"]).any():
        raise ValueError("Raw dataset contains duplicate keys.")

    fold_metrics = []
    all_predictions = []

    for fold_name, start_date in FOLDS:
        metrics, predictions = run_fold(
            fold_name,
            start_date,
            features_df,
            raw_df,
        )

        fold_metrics.append(metrics)
        all_predictions.append(predictions)

    metrics_df = pd.DataFrame(fold_metrics)

    combined_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    overall_metrics = calculate_metrics(
        combined_predictions["actual"],
        combined_predictions["prediction"],
    )

    summary = pd.DataFrame(
        [
            {
                "evaluation": "Pooled_3_Folds",
                "folds": len(FOLDS),
                "total_validation_rows": len(combined_predictions),
                **overall_metrics,
            }
        ]
    )

    metrics_df.to_csv(
        REPORT_DIR / "fold_metrics.csv",
        index=False,
    )

    summary.to_csv(
        REPORT_DIR / "summary_metrics.csv",
        index=False,
    )

    combined_predictions.to_csv(
        REPORT_DIR / "all_fold_predictions.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("ROLLING-ORIGIN VALIDATION COMPLETED")
    print("=" * 65)

    print("\nMetrics by fold:")
    print(metrics_df.to_string(index=False))

    print("\nPooled metrics across all folds:")
    print(summary.to_string(index=False))

    print("\nSaved reports:")
    print("Folder:", REPORT_DIR)
    print("Fold metrics:", REPORT_DIR / "fold_metrics.csv")
    print("Summary metrics:", REPORT_DIR / "summary_metrics.csv")
    print("All predictions:", REPORT_DIR / "all_fold_predictions.csv")


if __name__ == "__main__":
    main()
