from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task


@dag(
    dag_id="bargo_congress_trades_ingestion",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={
        "owner": "data-eng",
        "retries": 3,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["bargo", "congress", "trades"],
)
def bargo_congress_trades_ingestion() -> None:
    @task
    def ingest_latest_trades() -> str:
        from datetime import date, timedelta

        from config import get_settings
        from ingestion.bargo_client import BargoCongressTradesClient, BargoQuery
        from storage.duckdb_store import DuckDBStore

        settings = get_settings()
        client = BargoCongressTradesClient(settings.bargo_base_url, settings.bargo_api_key)

        # Pull a disclosure window larger than one day because STOCK Act filings lag trades.
        to_date = date.today()
        from_date = to_date - timedelta(days=7)
        query = BargoQuery(
            from_date=from_date.isoformat(),
            to_date=to_date.isoformat(),
            limit=settings.bargo_page_limit,
        )

        trades = list(client.iter_trades(query))
        store = DuckDBStore(settings.duckdb_path)
        run_id = store.start_ingestion_run(source="bargo_congress_trades", query=query.to_params(page=0))
        try:
            store.upsert_raw_trades(trades, run_id=run_id)
            store.finish_ingestion_run(run_id=run_id, row_count=len(trades))
        except Exception:
            store.finish_ingestion_run(run_id=run_id, row_count=0, status="failed")
            raise
        return run_id

    ingest_latest_trades()


bargo_congress_trades_ingestion()
