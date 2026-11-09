import joblib  # noqa: F401
import numpy as np
import pandas as pd
import pytest

from src.inference.predict import (
    MIN_HISTORY_LENGTH,
    DemandForecaster,  # noqa: F401, RUF100
    validate_history,
)


@pytest.fixture
def valid_history():
    dates = pd.date_range("2020-01-01", periods=400)

    return pd.DataFrame(
        {
            "date": dates,
            "store": 1,
            "item": 1,
            "sales": np.full(400, 20.0),
        }
    )


def test_valid_history_passes_validation(valid_history):
    result = validate_history(valid_history)

    assert len(result) == 400
    assert list(result.columns) == list(valid_history.columns)


def test_missing_columns_rejected(valid_history):
    invalid_history = valid_history.drop(columns=["sales"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_history(invalid_history)


def test_invalid_dates_rejected(valid_history):
    invalid_history = valid_history.copy()
    invalid_history["date"] = invalid_history["date"].astype(object)
    invalid_history.loc[0, "date"] = "invalid-date"

    with pytest.raises(ValueError, match="invalid or missing dates"):
        validate_history(invalid_history)


def test_negative_sales_rejected(valid_history):
    invalid_history = valid_history.copy()
    invalid_history.loc[0, "sales"] = -10

    with pytest.raises(ValueError, match="cannot be negative"):
        validate_history(invalid_history)


def test_duplicate_dates_rejected(valid_history):
    invalid_history = pd.concat(
        [valid_history, valid_history.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_history(invalid_history)


def test_insufficient_history_rejected(valid_history):
    invalid_history = valid_history.iloc[: MIN_HISTORY_LENGTH - 1]

    with pytest.raises(ValueError, match="Insufficient history"):
        validate_history(invalid_history)


def test_missing_daily_observations_rejected(valid_history):
    invalid_history = valid_history.drop(index=100)

    with pytest.raises(ValueError, match="Missing daily observations"):
        validate_history(invalid_history)


def test_invalid_store_id_rejected(valid_history):
    invalid_history = valid_history.copy()
    invalid_history["store"] = 11

    with pytest.raises(ValueError, match="Store IDs"):
        validate_history(invalid_history)


def test_invalid_item_id_rejected(valid_history):
    invalid_history = valid_history.copy()
    invalid_history["item"] = 51

    with pytest.raises(ValueError, match="Item IDs"):
        validate_history(invalid_history)


def test_non_finite_sales_rejected(valid_history):
    invalid_history = valid_history.copy()
    invalid_history.loc[0, "sales"] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        validate_history(invalid_history)


def test_missing_model_file_rejected(valid_history, tmp_path):
    history_path = tmp_path / "history.csv"
    valid_history.to_csv(history_path, index=False)

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        DemandForecaster(
            model_path=tmp_path / "missing_model.joblib",
            history_file=history_path,
        )


def test_missing_history_file_rejected(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.touch()

    with pytest.raises(FileNotFoundError, match="History file not found"):
        DemandForecaster(
            model_path=model_path,
            history_file=tmp_path / "missing_history.csv",
        )


def test_partial_artifact_paths_rejected(tmp_path):
    with pytest.raises(ValueError, match="Provide both model_path and history_file"):
        DemandForecaster(model_path=tmp_path / "model.joblib")


def test_corrupt_model_file_rejected(valid_history, tmp_path):
    model_path = tmp_path / "corrupt_model.joblib"
    history_path = tmp_path / "history.csv"

    model_path.write_text("not a valid model", encoding="utf-8")
    valid_history.to_csv(history_path, index=False)

    with pytest.raises(RuntimeError, match="Unable to load model artifact"):
        DemandForecaster(
            model_path=model_path,
            history_file=history_path,
        )


class FakeModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, features):
        return np.array([self.prediction])


def test_non_finite_prediction_rejected(monkeypatch):
    forecaster = DemandForecaster()
    monkeypatch.setattr(forecaster, "model", FakeModel(np.nan))

    with pytest.raises(RuntimeError, match="non-finite prediction"):
        forecaster.forecast(store=1, item=1, horizon_days=1)


def test_infinite_prediction_rejected(monkeypatch):
    forecaster = DemandForecaster()
    monkeypatch.setattr(forecaster, "model", FakeModel(np.inf))

    with pytest.raises(RuntimeError, match="non-finite prediction"):
        forecaster.forecast(store=1, item=1, horizon_days=1)


def test_negative_prediction_clipped_to_zero(monkeypatch):
    forecaster = DemandForecaster()
    monkeypatch.setattr(forecaster, "model", FakeModel(-10.0))

    result = forecaster.forecast(store=1, item=1, horizon_days=1)

    assert result[0]["predicted_sales"] == 0.0


def test_forecast_does_not_mutate_stored_history(monkeypatch):
    forecaster = DemandForecaster()
    monkeypatch.setattr(forecaster, "model", FakeModel(25.0))

    original_history = forecaster.history[(1, 1)].copy()

    forecaster.forecast(store=1, item=1, horizon_days=7)

    assert forecaster.history[(1, 1)] == original_history


@pytest.mark.parametrize(
    "store,item,horizon",
    [
        (0, 1, 7),
        (11, 1, 7),
        (1, 0, 7),
        (1, 51, 7),
        (1, 1, 0),
        (1, 1, 91),
    ],
)
def test_invalid_forecast_arguments_rejected(store, item, horizon):
    forecaster = DemandForecaster()

    with pytest.raises(ValueError):
        forecaster.forecast(store, item, horizon)
