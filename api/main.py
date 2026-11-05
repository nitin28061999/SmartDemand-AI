import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.inference.predict import DemandForecaster

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("smartdemand")

state = {}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def log_event(event: str, **fields):
    logger.info(json.dumps({"event": event, **fields}))


def require_api_key(key: str | None = Security(api_key_header)):
    expected = os.getenv("API_KEY")
    if not expected:  # auth disabled: local development and tests
        return
    if not key or not hmac.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("ENVIRONMENT") == "production" and not os.getenv("API_KEY"):
        raise RuntimeError("API_KEY must be set when ENVIRONMENT=production")
    start = time.perf_counter()
    state["forecaster"] = DemandForecaster()
    log_event("model_loaded", seconds=round(time.perf_counter() - start, 2))
    yield
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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    log_event(
        "request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=latency_ms,
    )
    return response


class ForecastRequest(BaseModel):
    store: int = Field(ge=1, le=10)
    item: int = Field(ge=1, le=50)
    horizon_days: int = Field(default=7, ge=1, le=90)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "forecaster" in state}


@app.post("/forecast", dependencies=[Depends(require_api_key)])
def forecast(request: ForecastRequest):
    try:
        predictions = state["forecaster"].forecast(
            request.store, request.item, request.horizon_days
        )
    except KeyError as error:
        log_event("forecast_rejected", store=request.store, item=request.item)
        raise HTTPException(status_code=404, detail=str(error)) from error

    values = [p["predicted_sales"] for p in predictions]
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
