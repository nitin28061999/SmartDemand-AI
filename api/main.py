
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)
from pydantic import BaseModel, Field

from src.inference.predict import DemandForecaster

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("smartdemand")

state = {}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Prometheus application metrics
HTTP_REQUESTS = Counter(
    "smartdemand_http_requests_total",
    "Total number of HTTP requests",
    ["path", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "smartdemand_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["path"],
)

MODEL_LOADED = Gauge(
    "smartdemand_model_loaded",
    "Whether the forecasting model is loaded",
)

FORECAST_HORIZON = Histogram(
    "smartdemand_forecast_horizon_days",
    "Distribution of requested forecast horizons in days",
)

PREDICTED_SALES = Summary(
    "smartdemand_predicted_sales",
    "Distribution of predicted sales values",
)


def log_event(event: str, **fields):
    logger.info(json.dumps({"event": event, **fields}))


def require_api_key(key: str | None = Security(api_key_header)):
    expected = os.getenv("API_KEY")
    if not expected:
        return
    if not key or not hmac.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("ENVIRONMENT") == "production" and not os.getenv("API_KEY"):
        raise RuntimeError("API_KEY must be set when ENVIRONMENT=production")

    MODEL_LOADED.set(0)

    start = time.perf_counter()
    state["forecaster"] = DemandForecaster()
    MODEL_LOADED.set(1)

    log_event("model_loaded", seconds=round(time.perf_counter() - start, 2))

    yield

    MODEL_LOADED.set(0)
    state.clear()


docs_on = os.getenv("ENABLE_DOCS", "true").lower() == "true"

app = FastAPI(
    title="SmartDemand AI",
    version="0.3.0",
    lifespan=lifespan,
    docs_url="/docs" if docs_on else None,
    redoc_url=None,
    openapi_url="/openapi.json" if docs_on else None,
)


def get_metric_path(request: Request) -> str:
    route = request.scope.get("route")

    if route is not None and getattr(route, "path", None):
        return route.path

    known_paths = {"/health", "/forecast"}
    return request.url.path if request.url.path in known_paths else "unmatched"


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    exception_occurred = False
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code

    except Exception as error:  # noqa: BLE001
        exception_occurred = True

        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        log_event(
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=500,
            latency_ms=latency_ms,
            error_type=type(error).__name__,
        )

        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    latency_seconds = time.perf_counter() - start
    latency_ms = round(latency_seconds * 1000, 1)

    # Exclude the metrics scrape endpoint to avoid counting Prometheus scrapes.
    if request.url.path != "/metrics":
        metric_path = get_metric_path(request)
        HTTP_REQUESTS.labels(
            path=metric_path,
            status=str(status_code),
        ).inc()
        HTTP_REQUEST_DURATION.labels(path=metric_path).observe(
            latency_seconds
        )

    response.headers["X-Request-ID"] = request_id

    if not exception_occurred:
        log_event(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=status_code,
            latency_ms=latency_ms,
        )

    return response


class ForecastRequest(BaseModel):
    store: int = Field(ge=1, le=10)
    item: int = Field(ge=1, le=50)
    horizon_days: int = Field(default=7, ge=1, le=90)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "forecaster" in state,
    }


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/forecast", dependencies=[Depends(require_api_key)])
def forecast(request: ForecastRequest):
    try:
        predictions = state["forecaster"].forecast(
            request.store,
            request.item,
            request.horizon_days,
        )

    except KeyError as error:
        log_event(
            "forecast_rejected",
            store=request.store,
            item=request.item,
        )
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        log_event(
            "forecast_failed",
            store=request.store,
            item=request.item,
            error_type=type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Forecast generation failed. Please try again later.",
        ) from error

    values = [p["predicted_sales"] for p in predictions]

    FORECAST_HORIZON.observe(request.horizon_days)

    for value in values:
        PREDICTED_SALES.observe(value)

    log_event(
        "forecast",
        store=request.store,
        item=request.item,
        horizon_days=request.horizon_days,
        pred_mean=round(sum(values) / len(values), 2),
        pred_min=min(values),
        pred_max=max(values),
    )

    return {
        "store": request.store,
        "item": request.item,
        "horizon_days": request.horizon_days,
        "predictions": predictions,
    }
