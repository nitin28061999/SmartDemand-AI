"""Centralized environment-based configuration for SmartDemand AI."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _optional_env(name: str) -> Optional[str]:
    """Return a configured environment value, or None when empty."""
    value = os.getenv(name)
    return value if value else None


def _read_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with explicit validation."""
    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(
        f"{name} must be one of: true, false, 1, 0, yes, no."
    )


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    environment: str
    api_key: Optional[str]
    metrics_api_key: Optional[str]
    docs_enabled: bool

    model_path: Path
    history_file: Path

    s3_bucket: Optional[str]
    aws_region: str
    model_key: str
    history_key: str

    @property
    def is_production(self) -> bool:
        """Return whether the application is running in production."""
        return self.environment == "production"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the current process environment."""
        environment = (
            os.getenv("ENVIRONMENT", "development").strip().lower()
            or "development"
        )

        return cls(
            environment=environment,
            api_key=_optional_env("API_KEY"),
            metrics_api_key=_optional_env("METRICS_API_KEY"),
            docs_enabled=_read_bool("ENABLE_DOCS", True),
            model_path=Path(
                os.getenv("MODEL_PATH")
                or str(PROJECT_ROOT / "models" / "xgboost_final_model.joblib")
            ),
            history_file=Path(
                os.getenv("HISTORY_FILE")
                or str(PROJECT_ROOT / "data" / "serving" / "history.csv")
            ),
            s3_bucket=_optional_env("S3_BUCKET"),
            aws_region=os.getenv("AWS_REGION") or "ap-southeast-2",
            model_key=(
                os.getenv("MODEL_KEY") or "models/xgboost_final_model.joblib"
            ),
            history_key=os.getenv("HISTORY_KEY") or "serving/history.csv",
        )

    def validate_startup(self) -> None:
        """Reject unsafe production configuration."""
        if self.is_production and not self.api_key:
            raise RuntimeError(
                "API_KEY must be set when ENVIRONMENT=production"
            )
