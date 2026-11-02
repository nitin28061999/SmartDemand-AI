from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.predict import DemandForecaster

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["forecaster"] = DemandForecaster()
    yield
    state.clear()


app = FastAPI(title="SmartDemand AI", version="0.1.0", lifespan=lifespan)


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
        raise HTTPException(status_code=404, detail=str(error))

    return {
        "store": request.store,
        "item": request.item,
        "horizon_days": request.horizon_days,
        "predictions": predictions,
    }
