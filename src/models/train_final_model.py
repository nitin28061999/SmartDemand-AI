from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = ROOT / "data" / "processed" / "train_features.csv"
RAW_FILE = ROOT / "data" / "raw" / "train.csv"
TEST_FILE = ROOT / "data" / "raw" / "test.csv"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "final_forecast"

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
    """Generate leakage-safe features from historical sales."""

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


def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading complete engineered training dataset...")

    features_df = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["date"],
    )

    print("Training dataset shape:", features_df.shape)
    print(
        "Training date range:",
        features_df["date"].min().date(),
        "to",
        features_df["date"].max().date(),
    )

    X_train = features_df[FEATURES]
    y_train = features_df["sales"]

    print("\nTraining final XGBoost model...")

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

    print("Final model training completed.")

    # Save final model separately from validation model.
    model_path = MODEL_DIR / "xgboost_final_model.joblib"
    joblib.dump(model, model_path)

    print("Final model saved:", model_path)

    # Load complete historical sales for recursive forecasting.
    print("\nLoading historical sales...")

    raw_df = pd.read_csv(
        RAW_FILE,
        parse_dates=["date"],
    )

    history = {}

    for (store, item), group in raw_df.groupby(["store", "item"]):
        history[(int(store), int(item))] = (  # pyright: ignore[reportArgumentType]
            group.sort_values("date")["sales"].astype(float).tolist()
        )

    # Load Kaggle test data.
    test_df = pd.read_csv(
        TEST_FILE,
        parse_dates=["date"],
    )

    test_dates = sorted(test_df["date"].unique())

    print("Test dataset shape:", test_df.shape)
    print(
        "Forecast date range:",
        test_df["date"].min().date(),
        "to",
        test_df["date"].max().date(),
    )

    predictions = []

    print("\nStarting recursive test forecasting...")

    for day_number, current_date in enumerate(test_dates, 1):
        current_date = pd.Timestamp(current_date)

        daily_rows = []
        daily_keys = []

        for store, item in sorted(history.keys()):
            row = create_features(
                current_date,
                history[(store, item)],
            )

            daily_rows.append(row)
            daily_keys.append((store, item))

        X_day = pd.DataFrame(
            daily_rows,
            columns=FEATURES,
        )

        daily_predictions = np.maximum(
            model.predict(X_day),
            0,
        )

        # Update history only after all daily predictions.
        for key, prediction in zip(
            daily_keys,
            daily_predictions,
        ):
            history[key].append(float(prediction))

        # Store predictions only for requested test combinations.
        test_day = test_df[test_df["date"] == current_date]

        prediction_lookup = dict(zip(daily_keys, daily_predictions))

        for row in test_day.itertuples(index=False):
            key = (int(row.store), int(row.item))  # pyright: ignore[reportArgumentType]

            predictions.append(
                {
                    "id": int(row.id),  # pyright: ignore[reportArgumentType]
                    "sales": float(prediction_lookup[key]),
                }
            )

        if day_number % 15 == 0:
            print(f"Completed {day_number}/{len(test_dates)} days")

    # Build final submission.
    prediction_df = pd.DataFrame(predictions)

    submission = test_df[["id"]].merge(
        prediction_df,
        on="id",
        how="left",
        validate="one_to_one",
    )

    assert len(submission) == len(test_df)
    assert submission["sales"].notna().all()
    assert submission["id"].equals(test_df["id"])
    assert (submission["sales"] >= 0).all()

    submission_path = REPORT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)

    print("\nFINAL FORECAST COMPLETED")
    print("=" * 45)
    print("Submission rows:", len(submission))
    print("Missing predictions:", submission["sales"].isna().sum())
    print("Minimum prediction:", submission["sales"].min())
    print("Maximum prediction:", submission["sales"].max())
    print("Average prediction:", submission["sales"].mean())

    print("\nSaved submission:", submission_path)
    print("Final model:", model_path)


if __name__ == "__main__":
    main()
