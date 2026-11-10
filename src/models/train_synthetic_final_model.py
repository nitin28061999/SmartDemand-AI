
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "synthetic" / "processed" / "features.csv"
)
MODEL_DIR = ROOT / "models" / "synthetic"
REPORT_DIR = ROOT / "reports" / "synthetic"

MODEL_FILE = MODEL_DIR / "xgboost_synthetic_final_model.joblib"
METADATA_FILE = REPORT_DIR / "final_model_metadata.json"

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

PARAMETERS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
}


def main():
    print("Loading synthetic feature dataset...")

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

    required_columns = FEATURES + ["sales", "date"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[required_columns].isna().any().any():
        raise ValueError("Missing values found in training data.")

    if not df[FEATURES + ["sales"]].apply(
        lambda column: pd.to_numeric(
            column, errors="coerce"
        ).notna().all()
    ).all(): # pyright: ignore[reportGeneralTypeIssues]
        raise ValueError("Non-numeric values found in model data.")

    df = df.sort_values(["date", "store", "item"]).copy()

    X = df[FEATURES]
    y = df["sales"]

    print("\n===== FINAL TRAINING DATA =====")
    print("Training rows:", f"{len(df):,}")
    print("Training start:", df["date"].min().date())
    print("Training end:", df["date"].max().date())
    print("Features:", len(FEATURES))

    print("\nTraining final XGBoost model on all available data...")

    model = XGBRegressor(**PARAMETERS)
    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    metadata = {
        "data_source": "synthetic",
        "disclosure": "All sales observations are artificially generated "
        "for a portfolio demonstration. They are not real "
        "retailer transactions or actual customer demand.",
        "model": "XGBoost Regressor",
        "training_rows": len(df),
        "training_start": df["date"].min().date().isoformat(),
        "training_end": df["date"].max().date().isoformat(),
        "features": FEATURES,
        "model_parameters": PARAMETERS,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "model_path": str(MODEL_FILE.relative_to(ROOT)),
        "purpose": "Final model for synthetic demonstration forecasts. "
        "Not used to calculate validation or backtest metrics.",
    }

    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\n===== FINAL MODEL SAVED =====")
    print("Model:", MODEL_FILE)
    print("Metadata:", METADATA_FILE)
    print("\nDATA SOURCE: SYNTHETIC — NOT REAL SALES")
    print("FINAL SYNTHETIC MODEL TRAINING COMPLETED")


if __name__ == "__main__":
    main()
