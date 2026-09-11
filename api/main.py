from __future__ import annotations

from fastapi import FastAPI, Query

from config import get_settings
from storage.duckdb_store import DuckDBStore

app = FastAPI(title="Senate Trades Anomaly API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/flagged-trades")
def flagged_trades(limit: int = Query(default=100, ge=1, le=500)) -> list[dict]:
    settings = get_settings()
    store = DuckDBStore(settings.duckdb_path)
    return store.fetch_flagged_trades(limit=limit)
