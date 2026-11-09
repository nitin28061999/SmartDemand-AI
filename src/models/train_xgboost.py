from pathlib import Path

import joblib
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

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "xgboost"

VALIDATION_DAYS = 90

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
    """Create leakage-safe features for one store-item combination."""

    return {
        "year": current_date.year,
        "month": current_date.month,
        "day": current_date.day,
        "day_of_week": current_date.day_of_week,
        "day_of_year": current_date.day_of_year,
        "is_weekend": int(current_date.day_of_week >= 5),
        "week_of_year": int(current_date.isocalendar().week),
        "lag_7": history[-7],
        "lag_14": history[-14],
        "lag_28": history[-28],
        "lag_365": history[-365],
        "rolling_mean_7": float(np.mean(history[-7:])),
        "rolling_mean_14": float(np.mean(history[-14:])),
        "rolling_mean_30": float(np.mean(history[-30:])),
    }


def calculate_metrics(actual, predicted):
    predicted = np.maximum(np.asarray(predicted), 0)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "RMSLE": np.sqrt(mean_squared_log_error(actual, predicted)),
    }


def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading engineered training data...")

    features_df = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["date"],
    )

    validation_start = features_df["date"].max() - pd.Timedelta(
        days=VALIDATION_DAYS - 1
    )

    # Only pre-validation rows are used for model fitting.
    training = features_df[features_df["date"] < validation_start].copy()

    print("Training rows:", len(training))
    print("Training end:", training["date"].max().date())

    X_train = training[FEATURES]
    y_train = training["sales"]

    print("\nTraining XGBoost model...")

    model = XGBRegressor(
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

    model.fit(X_train, y_train)

    print("Model training completed.")

    # Load original observations to initialize recursive history.
    raw_df = pd.read_csv(RAW_FILE, parse_dates=["date"])

    history_data = raw_df[raw_df["date"] < validation_start].copy()

    validation = raw_df[raw_df["date"] >= validation_start].copy()

    history = {
        (store, item): (
            group.sort_values("date")["sales"].astype(float).tolist()
        )
        for (store, item), group in history_data.groupby(["store", "item"])
    }
    predictions = []

    validation_dates = sorted(validation["date"].unique())

    print("\nStarting recursive 90-day forecasting...")

    for day_number, current_date in enumerate(validation_dates, 1):
        daily_rows = []
        daily_keys = []

        current_date = pd.Timestamp(current_date)

        for store, item in sorted(history.keys()):
            key = (store, item)

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

        # Append predictions only after all combinations
        # for the current day have been predicted.
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

        if day_number % 15 == 0:
            print(f"Completed {day_number}/{VALIDATION_DAYS} days")

    prediction_df = pd.DataFrame(predictions)

    results = validation[["date", "store", "item", "sales"]].rename(
        columns={"sales": "actual"}
    )

    results = results.merge(
        prediction_df,
        on=["date", "store", "item"],
        how="left",
        validate="one_to_one",
    )

    if results["prediction"].isna().any():
        raise ValueError("Missing predictions found in validation results.")

    expected_rows = len(validation_dates) * len(history)

    if len(results) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} validation rows, "
            f"but got {len(results)}."
        )

    metrics = calculate_metrics(
        results["actual"],
        results["prediction"],
    )

    print("\nXGBOOST VALIDATION PERFORMANCE")
    print("=" * 45)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save model, predictions and metrics.
    model_path = MODEL_DIR / "xgboost_demand_model.joblib"
    prediction_path = REPORT_DIR / "validation_predictions.csv"
    metrics_path = REPORT_DIR / "metrics.csv"

    joblib.dump(model, model_path)
    results.to_csv(prediction_path, index=False)

    pd.DataFrame([metrics]).to_csv(
        metrics_path,
        index=False,
    )

    # Save feature importance.
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,  # pyright: ignore[reportAttributeAccessIssue]
        }
    ).sort_values("importance", ascending=False)

    importance.to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False,
    )

    print("\nFILES SAVED")
    print("Model:", model_path)
    print("Predictions:", prediction_path)
    print("Metrics:", metrics_path)
    print("Feature importance:", REPORT_DIR / "feature_importance.csv")

    print("\nXGBoost evaluation completed successfully.")


if __name__ == "__main__":
    main()
