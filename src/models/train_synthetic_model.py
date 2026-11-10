import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "synthetic" / "processed" / "features.csv"
MODEL_DIR = ROOT / "models" / "synthetic"
REPORT_DIR = ROOT / "reports" / "synthetic"

MODEL_FILE = MODEL_DIR / "xgboost_synthetic_model.joblib"
METRICS_FILE = REPORT_DIR / "training_metrics.json"

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

TARGET = "sales"
VALIDATION_DAYS = 90


def main():
    print("Loading synthetic features...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    if df["sales"].isna().any():
        raise ValueError("Target column contains missing values.")

    latest_date = df["date"].max()
    validation_start = (
        latest_date - pd.Timedelta(days=VALIDATION_DAYS - 1)
    )

    train_df = df[df["date"] < validation_start].copy()
    valid_df = df[df["date"] >= validation_start].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("Training or validation dataset is empty.")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_valid = valid_df[FEATURES]
    y_valid = valid_df[TARGET]

    print("\n===== CHRONOLOGICAL DATA SPLIT =====")
    print("Training rows:", f"{len(train_df):,}")
    print("Validation rows:", f"{len(valid_df):,}")
    print("Training end:", train_df["date"].max().date())
    print("Validation start:", validation_start.date())
    print("Validation end:", latest_date.date())

    print("\nTraining XGBoost model...")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("Generating validation predictions...")

    predictions = model.predict(X_valid)
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_valid, predictions)
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))

    wape = (
        np.abs(y_valid.to_numpy() - predictions).sum()
        / y_valid.sum()
    ) * 100

    smape = (
        np.mean(
            2 * np.abs(y_valid.to_numpy() - predictions)
            / (
                np.abs(y_valid.to_numpy())
                + np.abs(predictions)
                + 1e-8
            )
        )
        * 100
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    importance = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    metrics = {
        "data_source": "synthetic",
        "disclosure": (
            "Model trained on artificially generated retail demand. "
            "Metrics do not represent real-world retailer performance."
        ),
        "model": "XGBoost Regressor",
        "validation_method": (
            "Chronological 90-day one-step-ahead validation "
            "using observed preceding-day sales."
        ),
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "training_start": train_df["date"].min().date().isoformat(),
        "training_end": train_df["date"].max().date().isoformat(),
        "validation_start": validation_start.date().isoformat(),
        "validation_end": latest_date.date().isoformat(),
        "features": FEATURES,
        "mae": float(mae),
        "rmse": float(rmse),
        "wape_percent": float(wape),
        "smape_percent": float(smape),
        "model_parameters": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
        "top_features": [
            {
                "feature": name,
                "importance": float(score),
            }
            for name, score in importance[:10]
        ],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_FILE.relative_to(ROOT)),
    }

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n===== SYNTHETIC MODEL EVALUATION =====")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("WAPE:", f"{wape:.2f}%")
    print("sMAPE:", f"{smape:.2f}%")

    print("\nTop 10 important features:")
    for name, score in importance[:10]:
        print(f"{name:20s} {score:.5f}")

    print("\n===== SAVED ARTIFACTS =====")
    print("Model:", MODEL_FILE)
    print("Metrics:", METRICS_FILE)
    print("\nDATA SOURCE: SYNTHETIC — NOT REAL SALES")
    print("SYNTHETIC MODEL TRAINING COMPLETED")


if __name__ == "__main__":
    main()
