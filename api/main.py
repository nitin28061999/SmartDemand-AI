from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response
from starlette.requests import Request

from src.inference.predict import DemandForecaster  # pyright: ignore[reportAttributeAccessIssue]

state = {}

HTTP_REQUESTS = Counter(
    "smartdemand_http_requests_total",
    "HTTP requests handled by the API.",
    ("method", "path", "status"),
)
HTTP_REQUEST_DURATION = Histogram(
    "smartdemand_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "path"),
)
MODEL_LOADED = Gauge(
    "smartdemand_model_loaded",
    "Whether the forecasting model and serving history are loaded.",
)
MODEL_LOADED.set(0)
FORECAST_HORIZON = Histogram(
    "smartdemand_forecast_horizon_days",
    "Requested forecast horizon in days.",
    buckets=(1, 7, 14, 30, 60, 90),
)
PREDICTED_SALES = Histogram(
    "smartdemand_predicted_sales",
    "Predicted sales values returned by the API.",
    buckets=(0, 5, 10, 20, 40, 60, 100, 150, 250),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["forecaster"] = DemandForecaster()
    MODEL_LOADED.set(1)
    yield
    MODEL_LOADED.set(0)
    state.clear()


app = FastAPI(title="SmartDemand AI", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def collect_http_metrics(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    start = perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        HTTP_REQUESTS.labels(request.method, path, str(status_code)).inc()
        HTTP_REQUEST_DURATION.labels(request.method, path).observe(
            perf_counter() - start
        )


class ForecastRequest(BaseModel):
    store: int = Field(ge=1, le=10)
    item: int = Field(ge=1, le=50)
    horizon_days: int = Field(default=7, ge=1, le=90)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "forecaster" in state}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        predictions = state["forecaster"].forecast(
            request.store, request.item, request.horizon_days
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error))

    FORECAST_HORIZON.observe(request.horizon_days)
    for prediction in predictions:
        PREDICTED_SALES.observe(prediction["predicted_sales"])

    return {
        "store": request.store,
        "item": request.item,
        "horizon_days": request.horizon_days,
        "predictions": predictions,
    }
