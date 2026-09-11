from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb


class DuckDBStore:
    def __init__(self, database_path: str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.database_path))

    def _ensure_schema(self) -> None:
        with self._connect() as con:
            con.execute("create schema if not exists raw")
            con.execute("create schema if not exists audit")
            con.execute(
                """
                create table if not exists raw.bargo_congress_trades (
                    row_hash varchar primary key,
                    ingestion_run_id varchar not null,
                    ingested_at timestamp not null,
                    payload json not null
                )
                """
            )
            con.execute(
                """
                create table if not exists audit.ingestion_runs (
                    run_id varchar primary key,
                    source varchar not null,
                    query json not null,
                    started_at timestamp not null,
                    finished_at timestamp,
                    status varchar not null,
                    row_count integer
                )
                """
            )

    def start_ingestion_run(self, source: str, query: dict[str, Any]) -> str:
        run_id = str(uuid.uuid4())
        with self._connect() as con:
            con.execute(
                """
                insert into audit.ingestion_runs
                (run_id, source, query, started_at, status)
                values (?, ?, ?::json, ?, ?)
                """,
                [run_id, source, json.dumps(query), datetime.now(UTC), "running"],
            )
        return run_id

    def finish_ingestion_run(self, run_id: str, row_count: int, status: str = "success") -> None:
        with self._connect() as con:
            con.execute(
                """
                update audit.ingestion_runs
                set finished_at = ?, status = ?, row_count = ?
                where run_id = ?
                """,
                [datetime.now(UTC), status, row_count, run_id],
            )

    def upsert_raw_trades(self, trades: list[dict[str, Any]], run_id: str) -> None:
        if not trades:
            return

        rows = []
        ingested_at = datetime.now(UTC)
        for trade in trades:
            normalized_payload = json.dumps(trade, sort_keys=True)
            row_hash = str(uuid.uuid5(uuid.NAMESPACE_URL, normalized_payload))
            rows.append((row_hash, run_id, ingested_at, normalized_payload))

        with self._connect() as con:
            con.executemany(
                """
                insert or replace into raw.bargo_congress_trades
                (row_hash, ingestion_run_id, ingested_at, payload)
                values (?, ?, ?, ?::json)
                """,
                rows,
            )

    def fetch_flagged_trades(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as con:
            table_exists = con.execute(
                """
                select count(*)
                from information_schema.tables
                where table_schema = 'main' and table_name = 'mart_flagged_trades'
                """
            ).fetchone()[0]
            if not table_exists:
                return []
            result = con.execute(
                "select * from mart_flagged_trades order by anomaly_score desc limit ?",
                [limit],
            ).fetchdf()
        return result.to_dict(orient="records")
