import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.inference.predict import DemandForecaster

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("smartdemand")

state = {}


def log_event(event: str, **fields):
    logger.info(json.dumps({"event": event, **fields}))


@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.perf_counter()
    state["forecaster"] = DemandForecaster()
    log_event("model_loaded", seconds=round(time.perf_counter() - start, 2))
    yield
    state.clear()


app = FastAPI(title="SmartDemand AI", version="0.2.0", lifespan=lifespan)


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


@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        predictions = state["forecaster"].forecast(
            request.store, request.item, request.horizon_days
        )
    except KeyError as error:
        log_event("forecast_rejected", store=request.store, item=request.item)
        raise HTTPException(status_code=404, detail=str(error))

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
