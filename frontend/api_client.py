"""Client for communicating with the SmartDemand production API."""

import os
from typing import Any

import requests
import streamlit as st

DEFAULT_API_URL = "http://32.236.49.254:18001"


class SmartDemandAPIError(Exception):
    """Raised when the SmartDemand API request fails."""


class SmartDemandAPI:
    """Server-side client for the SmartDemand FastAPI service."""

    def __init__(self) -> None:
        secrets = getattr(st, "secrets", {})

        self.base_url = (
            os.getenv("SMARTDEMAND_API_URL")
            or secrets.get("SMARTDEMAND_API_URL")
            or DEFAULT_API_URL
        ).rstrip("/")

        self.api_key = (
            os.getenv("SMARTDEMAND_API_KEY")
            or secrets.get("SMARTDEMAND_API_KEY")
        )

        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def health_check(self) -> dict[str, Any]:
        """Check whether the production API and model are healthy."""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=10,
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as error:
            raise SmartDemandAPIError(
                "Unable to connect to the production API."
            ) from error

    def forecast(
        self,
        store: int,
        item: int,
        horizon_days: int,
    ) -> dict[str, Any]:
        """Request a demand forecast from the production model."""
        if not self.api_key:
            raise SmartDemandAPIError(
                "API credentials are not configured. "
                "Set SMARTDEMAND_API_KEY in the server environment."
            )

        payload = {
            "store": store,
            "item": item,
            "horizon_days": horizon_days,
        }

        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            response = self.session.post(
                f"{self.base_url}/forecast",
                json=payload,
                headers=headers,
                timeout=30,
            )

            if response.status_code == 401:
                raise SmartDemandAPIError(
                    "API authentication failed. Check the configured credentials."
                )

            if response.status_code == 404:
                raise SmartDemandAPIError(
                    "No forecast is available for this store and item combination."
                )

            response.raise_for_status()
            return response.json()

        except requests.Timeout as error:
            raise SmartDemandAPIError(
                "The forecasting request timed out. Please try again."
            ) from error

        except requests.RequestException as error:
            raise SmartDemandAPIError(
                "The forecasting service encountered a connection error."
            ) from error
