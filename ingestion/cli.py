from __future__ import annotations

from datetime import date

import typer

from config import get_settings
from ingestion.bargo_client import BargoCongressTradesClient, BargoQuery
from storage.duckdb_store import DuckDBStore

app = typer.Typer(help="Pipeline utilities for Senate trades anomaly detection.")


@app.command("ingest-bargo")
def ingest_bargo(
    from_date: date | None = typer.Option(
        None,
        help="Inclusive transaction/disclosure start date.",
    ),
    to_date: date | None = typer.Option(None, help="Inclusive transaction/disclosure end date."),
    ticker: str | None = typer.Option(None, help="Optional ticker filter."),
    member: str | None = typer.Option(None, help="Optional member substring filter."),
    chamber: str | None = typer.Option(None, help="house or senate."),
    transaction_type: str | None = typer.Option(None, help="purchase, sale, or exchange."),
    max_pages: int | None = typer.Option(None, help="Optional safety cap while developing."),
) -> None:
    settings = get_settings()
    client = BargoCongressTradesClient(settings.bargo_base_url, settings.bargo_api_key)
    query = BargoQuery(
        from_date=from_date.isoformat() if from_date else None,
        to_date=to_date.isoformat() if to_date else None,
        ticker=ticker,
        member=member,
        chamber=chamber,
        transaction_type=transaction_type,
        limit=settings.bargo_page_limit,
    )
    trades = list(client.iter_trades(query, max_pages=max_pages))

    store = DuckDBStore(settings.duckdb_path)
    run_id = store.start_ingestion_run(
        source="bargo_congress_trades",
        query=query.to_params(page=0),
    )
    store.upsert_raw_trades(trades, run_id=run_id)
    store.finish_ingestion_run(run_id=run_id, row_count=len(trades))

    typer.echo(f"Loaded {len(trades)} Bargo trades into {settings.duckdb_path}")


if __name__ == "__main__":
    app()
