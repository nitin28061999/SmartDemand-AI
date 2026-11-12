from pathlib import Path
import pytest

from config.settings import PROJECT_ROOT, Settings


ENVIRONMENT_VARIABLES = [
    "ENVIRONMENT",
    "API_KEY",
    "METRICS_API_KEY",
    "ENABLE_DOCS",
    "MODEL_PATH",
    "HISTORY_FILE",
    "S3_BUCKET",
    "AWS_REGION",
    "MODEL_KEY",
    "HISTORY_KEY",
]


def clear_settings_environment(monkeypatch):
    for name in ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_default_settings(monkeypatch):
    clear_settings_environment(monkeypatch)

    settings = Settings.from_env()

    assert settings.environment == "development"
    assert settings.api_key is None
    assert settings.metrics_api_key is None
    assert settings.docs_enabled is True
    assert settings.model_path == (
        PROJECT_ROOT / "models" / "xgboost_final_model.joblib"
    )
    assert settings.history_file == (
        PROJECT_ROOT / "data" / "serving" / "history.csv"
    )
    assert settings.s3_bucket is None
    assert settings.aws_region == "ap-southeast-2"
    assert settings.model_key == "models/xgboost_final_model.joblib"
    assert settings.history_key == "serving/history.csv"


def test_environment_values_are_loaded_dynamically(monkeypatch):
    clear_settings_environment(monkeypatch)

    monkeypatch.setenv("ENVIRONMENT", "PRODUCTION")
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("METRICS_API_KEY", "test-metrics-key")
    monkeypatch.setenv("MODEL_PATH", "custom/model.joblib")
    monkeypatch.setenv("HISTORY_FILE", "custom/history.csv")
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
    monkeypatch.setenv("MODEL_KEY", "artifacts/model.joblib")
    monkeypatch.setenv("HISTORY_KEY", "artifacts/history.csv")
    monkeypatch.setenv("ENABLE_DOCS", "false")

    settings = Settings.from_env()

    assert settings.environment == "production"
    assert settings.is_production is True
    assert settings.api_key == "test-api-key"
    assert settings.metrics_api_key == "test-metrics-key"
    assert settings.model_path == Path("custom/model.joblib")
    assert settings.history_file == Path("custom/history.csv")
    assert settings.s3_bucket == "test-bucket"
    assert settings.aws_region == "ap-southeast-2"
    assert settings.model_key == "artifacts/model.joblib"
    assert settings.history_key == "artifacts/history.csv"
    assert settings.docs_enabled is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
    ],
)
def test_boolean_environment_values(monkeypatch, value, expected):
    clear_settings_environment(monkeypatch)
    monkeypatch.setenv("ENABLE_DOCS", value)

    assert Settings.from_env().docs_enabled is expected


def test_invalid_boolean_environment_value_is_rejected(monkeypatch):
    clear_settings_environment(monkeypatch)
    monkeypatch.setenv("ENABLE_DOCS", "sometimes")

    with pytest.raises(ValueError, match="ENABLE_DOCS"):
        Settings.from_env()


def test_production_requires_api_key(monkeypatch):
    clear_settings_environment(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")

    settings = Settings.from_env()

    with pytest.raises(
        RuntimeError,
        match="API_KEY must be set when ENVIRONMENT=production",
    ):
        settings.validate_startup()


def test_production_accepts_configured_api_key(monkeypatch):
    clear_settings_environment(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("API_KEY", "production-test-key")

    settings = Settings.from_env()
    settings.validate_startup()

    assert settings.is_production is True


def test_development_does_not_require_api_key(monkeypatch):
    clear_settings_environment(monkeypatch)

    settings = Settings.from_env()
    settings.validate_startup()

    assert settings.is_production is False


def test_empty_values_use_default_paths_and_s3_settings(monkeypatch):
    clear_settings_environment(monkeypatch)

    for name in (
        "MODEL_PATH",
        "HISTORY_FILE",
        "AWS_REGION",
        "MODEL_KEY",
        "HISTORY_KEY",
    ):
        monkeypatch.setenv(name, "")

    settings = Settings.from_env()

    assert settings.model_path == (
        PROJECT_ROOT / "models" / "xgboost_final_model.joblib"
    )
    assert settings.history_file == (
        PROJECT_ROOT / "data" / "serving" / "history.csv"
    )
    assert settings.aws_region == "ap-southeast-2"
    assert settings.model_key == "models/xgboost_final_model.joblib"
    assert settings.history_key == "serving/history.csv"

