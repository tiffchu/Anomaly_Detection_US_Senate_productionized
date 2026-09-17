from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict

DEFAULT_TRACKING_URI = "sqlite:////usr/local/airflow/include/data/mlflow.db"
DEFAULT_MODEL_NAME = "congress-trade-anomaly-detector"
DEFAULT_MODEL_ALIAS = "production"


class TradeFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount_low: float | None = None
    amount_high: float | None = None
    disclosure_lag_days: float | None = None
    estimated_trade_price: float | None = None
    recent_price: float | None = None
    performance_pct: float | None = None
    signed_amount_high: float | None = None
    member_avg_amount_high: float | None = None
    member_running_net_amount: float | None = None
    member_trade_count: float | None = None
    ticker_trade_count: float | None = None
    chamber: str | None = None
    state: str | None = None
    transaction_type: str | None = None
    outcome: str | None = None


class ScoreResponse(BaseModel):
    model_name: str
    model_version: int
    model_alias: str
    source_run_id: str
    cluster_label: int
    cluster_anomaly_score: float
    reconstruction_error: float
    anomaly_score: float
    is_anomaly: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    model_name = os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME)
    model_alias = os.getenv("MLFLOW_MODEL_ALIAS", DEFAULT_MODEL_ALIAS)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, model_alias)

    app.state.model = mlflow.pyfunc.load_model(model_version.source)
    app.state.model_name = model_name
    app.state.model_alias = model_alias
    app.state.model_version = int(model_version.version)
    app.state.source_run_id = model_version.run_id
    yield


app = FastAPI(
    title="Congress Trade Anomaly Scoring API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/score", response_model=ScoreResponse)
def score_trade(features: TradeFeatures, request: Request) -> dict[str, Any]:
    model_input = pd.DataFrame([features.model_dump()])
    prediction = request.app.state.model.predict(model_input).iloc[0]
    return {
        "model_name": request.app.state.model_name,
        "model_version": request.app.state.model_version,
        "model_alias": request.app.state.model_alias,
        "source_run_id": request.app.state.source_run_id,
        "cluster_label": int(prediction["cluster_label"]),
        "cluster_anomaly_score": float(prediction["cluster_anomaly_score"]),
        "reconstruction_error": float(prediction["reconstruction_error"]),
        "anomaly_score": float(prediction["anomaly_score"]),
        "is_anomaly": bool(prediction["is_anomaly"]),
    }
