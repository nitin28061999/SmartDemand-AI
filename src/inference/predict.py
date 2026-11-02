import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "xgboost_final_model.joblib"))
LOCAL_HISTORY_FILE = Path(os.getenv("HISTORY_FILE", ROOT / "data" / "serving" / "history.csv"))

FEATURES = [
    "year", "month", "day", "day_of_week", "day_of_year",
    "week_of_year", "is_weekend",
    "lag_7", "lag_14", "lag_28", "lag_365",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
]


def resolve_artifacts():
    """Return (model_path, history_path). Downloads from S3 when S3_BUCKET is set."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return LOCAL_MODEL_PATH, LOCAL_HISTORY_FILE

    import boto3  # imported lazily so local runs and tests don't need AWS

    region = os.getenv("AWS_REGION", "ap-southeast-2")
    model_key = os.getenv("MODEL_KEY", "models/xgboost_final_model.joblib")
    history_key = os.getenv("HISTORY_KEY", "serving/history.csv")

    cache = Path(tempfile.gettempdir()) / "smartdemand"
    cache.mkdir(parents=True, exist_ok=True)
    model_path = cache / "model.joblib"
    history_path = cache / "history.csv"

    s3 = boto3.client("s3", region_name=region)
    s3.download_file(bucket, model_key, str(model_path))
    s3.download_file(bucket, history_key, str(history_path))
    return model_path, history_path


def create_features(current_date, history):
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


class DemandForecaster:
    def __init__(self, model_path=None, history_file=None):
        if model_path is None or history_file is None:
            model_path, history_file = resolve_artifacts()

        self.model = joblib.load(model_path)
        df = pd.read_csv(history_file, parse_dates=["date"])
        df = df.sort_values(["store", "item", "date"])
        self.last_date = df["date"].max()
        self.history = {
            key: group["sales"].astype(float).tolist()
            for key, group in df.groupby(["store", "item"])
        }

    def forecast(self, store: int, item: int, horizon_days: int):
        key = (store, item)
        if key not in self.history:
            raise KeyError(f"Unknown store/item: {key}")

        history = list(self.history[key])  # copy, never mutate stored history
        results = []

        for day in range(1, horizon_days + 1):
            date = self.last_date + pd.Timedelta(days=day)
            row = pd.DataFrame([create_features(date, history)], columns=FEATURES)
            prediction = max(float(self.model.predict(row)[0]), 0.0)
            history.append(prediction)
            results.append({
                "date": date.date().isoformat(),
                "predicted_sales": round(prediction, 2),
            })

        return results
