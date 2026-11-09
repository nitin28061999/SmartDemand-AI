import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

LOCAL_MODEL_PATH = Path(
    os.getenv("MODEL_PATH", ROOT / "models" / "xgboost_final_model.joblib")
)
LOCAL_HISTORY_FILE = Path(
    os.getenv("HISTORY_FILE", ROOT / "data" / "serving" / "history.csv")
)

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

REQUIRED_HISTORY_COLUMNS = {"date", "store", "item", "sales"}
MIN_HISTORY_LENGTH = 365


def resolve_artifacts():
    """Resolve local artifacts or download model and history from S3."""
    bucket = os.getenv("S3_BUCKET")

    if not bucket:
        return LOCAL_MODEL_PATH, LOCAL_HISTORY_FILE

    import boto3

    region = os.getenv("AWS_REGION", "ap-southeast-2")
    model_key = os.getenv("MODEL_KEY", "models/xgboost_final_model.joblib")
    history_key = os.getenv("HISTORY_KEY", "serving/history.csv")

    cache = Path(tempfile.gettempdir()) / "smartdemand"
    cache.mkdir(parents=True, exist_ok=True)

    model_path = cache / "model.joblib"
    history_path = cache / "history.csv"

    s3 = boto3.client("s3", region_name=region)

    try:
        s3.download_file(bucket, model_key, str(model_path))
        s3.download_file(bucket, history_key, str(history_path))
    except Exception as error:
        raise RuntimeError(
            f"Unable to download model or history artifacts from S3. Bucket: {bucket}"
        ) from error

    return model_path, history_path


def validate_history(df: pd.DataFrame) -> pd.DataFrame:
    """Validate historical demand data before model initialization."""
    missing_columns = REQUIRED_HISTORY_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"History file is missing required columns: {sorted(missing_columns)}"
        )

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    if df["date"].isna().any():
        raise ValueError("History contains invalid or missing dates.")

    for column in ("store", "item", "sales"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[["store", "item", "sales"]].isna().any().any():
        raise ValueError("History contains missing or non-numeric values.")

    if not np.isfinite(df[["store", "item", "sales"]].to_numpy()).all():
        raise ValueError("History contains non-finite numeric values.")

    if (df["sales"] < 0).any():
        raise ValueError("Historical sales cannot be negative.")

    if (df["store"] < 1).any() or (df["store"] > 10).any():
        raise ValueError("Store IDs must be between 1 and 10.")

    if (df["item"] < 1).any() or (df["item"] > 50).any():
        raise ValueError("Item IDs must be between 1 and 50.")

    if not np.equal(df["store"], np.floor(df["store"])).all():
        raise ValueError("Store IDs must be integers.")

    if not np.equal(df["item"], np.floor(df["item"])).all():
        raise ValueError("Item IDs must be integers.")

    if df.duplicated(["store", "item", "date"]).any():
        raise ValueError("History contains duplicate store/item/date records.")

    df = df.sort_values(["store", "item", "date"])

    for key, group in df.groupby(["store", "item"]):
        if len(group) < MIN_HISTORY_LENGTH:
            raise ValueError(
                f"Insufficient history for store/item {key}: "
                f"found {len(group)}, need at least {MIN_HISTORY_LENGTH}."
            )

        expected_dates = pd.date_range(
            start=group["date"].min(),
            end=group["date"].max(),
            freq="D",
        )

        if len(expected_dates) != len(group):
            raise ValueError(f"Missing daily observations for store/item {key}.")

    return df


def create_features(current_date, history):
    """Create time and lag features for a single forecast date."""
    if len(history) < MIN_HISTORY_LENGTH:
        raise ValueError(f"At least {MIN_HISTORY_LENGTH} observations are required.")

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


class DemandForecaster:
    def __init__(self, model_path=None, history_file=None):
        if model_path is None and history_file is None:
            model_path, history_file = resolve_artifacts()
        elif model_path is None or history_file is None:
            raise ValueError("Provide both model_path and history_file, or neither.")

        model_path = Path(model_path)
        history_file = Path(history_file)

        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not history_file.is_file():
            raise FileNotFoundError(f"History file not found: {history_file}")

        try:
            self.model = joblib.load(model_path)
        except Exception as error:
            raise RuntimeError("Unable to load model artifact.") from error

        try:
            df = pd.read_csv(history_file)
        except Exception as error:
            raise RuntimeError("Unable to read history artifact.") from error

        df = validate_history(df)

        self.last_date = df["date"].max()

        self.history = {
            (int(store), int(item)): group["sales"].astype(float).tolist()  # pyright: ignore[reportArgumentType]
            for (store, item), group in df.groupby(["store", "item"])
        }

        if not self.history:
            raise ValueError("History file contains no store/item records.")

    def forecast(self, store: int, item: int, horizon_days: int):
        if not 1 <= store <= 10:
            raise ValueError("Store must be between 1 and 10.")

        if not 1 <= item <= 50:
            raise ValueError("Item must be between 1 and 50.")

        if not 1 <= horizon_days <= 90:
            raise ValueError("Forecast horizon must be between 1 and 90 days.")

        key = (store, item)

        if key not in self.history:
            raise KeyError(f"Unknown store/item: {key}")

        history = list(self.history[key])
        results = []

        for day in range(1, horizon_days + 1):
            date = self.last_date + pd.Timedelta(days=day)

            row = pd.DataFrame(
                [create_features(date, history)],
                columns=FEATURES,
            )

            prediction = float(self.model.predict(row)[0])

            if not np.isfinite(prediction):
                raise RuntimeError("Model returned a non-finite prediction.")

            prediction = max(prediction, 0.0)
            history.append(prediction)

            results.append(
                {
                    "date": date.date().isoformat(),
                    "predicted_sales": round(prediction, 2),
                }
            )

        return results
