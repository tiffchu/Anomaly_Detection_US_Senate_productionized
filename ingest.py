from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests
from requests import HTTPError


DEFAULT_BASE_URL = "https://www.bargo.ai/free-apis/congress/v1"
TRADE_ID_FIELDS = (
    "member",
    "member_slug",
    "ticker",
    "type",
    "amount_low",
    "amount_high",
    "transaction_date",
    "disclosure_date",
)


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def fetch_page(
    base_url: str,
    api_key: str | None,
    params: dict[str, Any],
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    headers = {"X-Api-Key": api_key} if api_key else {}
    response = requests.get(
        f"{base_url.rstrip('/')}/trades",
        params=params,
        headers=headers,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def build_trade_id(trade: dict[str, Any]) -> str:
    key = "|".join(str(trade.get(field) or "") for field in TRADE_ID_FIELDS)
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def with_trade_ids(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for trade in trades:
        record = dict(trade)
        record["trade_id"] = build_trade_id(record)
        records.append(record)
    return records


def fetch_trades(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    load_dotenv()
    api_key = os.getenv("BARGO_API_KEY")
    base_url = os.getenv("BARGO_BASE_URL", DEFAULT_BASE_URL)

    all_trades: list[dict[str, Any]] = []
    raw_pages: list[dict[str, Any]] = []

    for page in range(args.start_page, args.start_page + args.pages):
        params: dict[str, Any] = {"limit": args.limit, "page": page}
        for name in ("ticker", "member", "chamber", "type", "from_date", "to_date"):
            value = getattr(args, name)
            if value:
                api_name = {"from_date": "from", "to_date": "to"}.get(name, name)
                params[api_name] = value

        try:
            payload = fetch_page(base_url, api_key, params)
        except HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 429:
                if all_trades:
                    print(
                        f"page={page} status=rate_limited rows=0; "
                        f"using {len(all_trades)} rows already fetched"
                    )
                    break
                if args.allow_stale_on_rate_limit:
                    print(
                        "page=0 status=rate_limited rows=0; "
                        "preserving existing DuckDB raw_trades data"
                    )
                    break
            raise
        trades = payload.get("trades", [])
        raw_pages.append({"request_params": params, "response": payload})
        all_trades.extend(trades)

        print(
            f"page={page} status=ok rows={len(trades)} "
            f"envelope_keys={sorted(payload.keys())}"
        )
        if trades:
            print(f"sample_fields={sorted(trades[0].keys())}")

        if len(trades) < args.limit:
            break

    return all_trades, raw_pages


def save_outputs(
    trades: list[dict[str, Any]],
    raw_pages: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    trades_json_path = output_dir / "trades.json"
    timestamped_raw_path = output_dir / f"bargo_raw_response_{timestamp}.json"
    csv_path = output_dir / f"bargo_trades_{timestamp}.csv"

    records = with_trade_ids(trades)

    trades_json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    timestamped_raw_path.write_text(
        json.dumps(
            {
                "fetched_at_utc": timestamp,
                "page_count": len(raw_pages),
                "trade_count": len(records),
                "pages": raw_pages,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(records).to_csv(csv_path, index=False)
    return trades_json_path, timestamped_raw_path, csv_path


def ensure_raw_trades_table(con: duckdb.DuckDBPyConnection) -> None:
    con.sql(
        """
        create table if not exists raw_trades (
            trade_id varchar primary key,
            member varchar,
            member_slug varchar,
            chamber varchar,
            state varchar,
            ticker varchar,
            asset varchar,
            type varchar,
            amount_low bigint,
            amount_high bigint,
            amount_range varchar,
            transaction_date date,
            disclosure_date date,
            est_price double,
            recent_price double,
            recent_price_date date,
            perf_pct double,
            realized_return_pct json,
            outcome varchar,
            filing_portal varchar
        )
        """
    )


def load_duckdb(json_path: Path, duckdb_path: Path) -> None:
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(json_path.read_text(encoding="utf-8"))
    con = duckdb.connect(str(duckdb_path))
    try:
        raw_table_exists = con.sql(
            """
            select count(*) > 0
            from information_schema.tables
            where table_schema = 'main'
              and table_name = 'raw_trades'
            """
        ).fetchone()[0]
        if raw_table_exists:
            has_trade_id = con.sql(
                """
                select count(*) > 0
                from information_schema.columns
                where table_schema = 'main'
                  and table_name = 'raw_trades'
                  and column_name = 'trade_id'
                """
            ).fetchone()[0]
            if not has_trade_id:
                con.sql("drop view if exists stg_trades")
                con.sql("drop table raw_trades")

        ensure_raw_trades_table(con)

        con.sql("drop table if exists raw_trades_stage")
        if records:
            con.sql(
                f"""
                create table raw_trades_stage as
                select *
                from read_json_auto('{json_path.as_posix()}')
                """
            )
            con.sql(
                """
                insert or replace into raw_trades by name
                select
                    trade_id,
                    member,
                    member_slug,
                    chamber,
                    state,
                    ticker,
                    asset,
                    type,
                    amount_low,
                    amount_high,
                    amount_range,
                    transaction_date,
                    disclosure_date,
                    est_price,
                    recent_price,
                    recent_price_date,
                    perf_pct,
                    realized_return_pct,
                    outcome,
                    filing_portal
                from raw_trades_stage
                """
            )
        else:
            print("No trades returned; raw_trades table shape is preserved.")

        print("\nDuckDB exploration")
        print("------------------")
        print(con.sql("select count(*) as row_count from raw_trades").df().to_string(index=False))
        print(
            con.sql(
                """
                select
                    count(*) as rows,
                    count(distinct trade_id) as distinct_trade_ids,
                    count(distinct member || '|' || ticker || '|' || transaction_date || '|' || type || '|' || amount_range) as approx_distinct_rows,
                    sum(case when ticker is null then 1 else 0 end) as null_ticker_rows,
                    sum(case when transaction_date is null then 1 else 0 end) as null_transaction_date_rows,
                    sum(case when disclosure_date is null then 1 else 0 end) as null_disclosure_date_rows
                from raw_trades
                """
            )
            .df()
            .to_string(index=False)
        )
        print(
            con.sql(
                """
                select chamber, type, count(*) as rows
                from raw_trades
                group by chamber, type
                order by rows desc
                """
            )
            .df()
            .to_string(index=False)
        )
        print(
            con.sql(
                """
                describe raw_trades
                """
            )
            .df()
            .to_string(index=False)
        )
        con.sql("drop table if exists raw_trades_stage")
    finally:
        con.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Bargo REST ingest and DuckDB exploration script."
    )
    parser.add_argument("--limit", type=int, default=25, help="Rows per page.")
    parser.add_argument("--pages", type=int, default=1, help="Maximum pages to pull.")
    parser.add_argument("--start-page", type=int, default=0, help="First page to request.")
    parser.add_argument("--ticker", help="Optional ticker filter, e.g. NVDA.")
    parser.add_argument("--member", help="Optional member substring filter.")
    parser.add_argument("--chamber", choices=["house", "senate"], help="Optional chamber filter.")
    parser.add_argument("--type", help="Optional transaction type filter.")
    parser.add_argument("--from-date", help="Earliest transaction date, YYYY-MM-DD.")
    parser.add_argument("--to-date", help="Latest transaction date, YYYY-MM-DD.")
    parser.add_argument("--output-dir", default="data/raw", help="Raw output directory.")
    parser.add_argument("--duckdb-path", default="congress.duckdb", help="DuckDB database path.")
    parser.add_argument("--skip-duckdb", action="store_true", help="Only fetch and save files.")
    parser.add_argument(
        "--allow-stale-on-rate-limit",
        action="store_true",
        help="If Bargo returns 429 before any rows are fetched, keep existing DuckDB data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trades, raw_pages = fetch_trades(args)
    trades_json_path, raw_response_path, csv_path = save_outputs(
        trades,
        raw_pages,
        Path(args.output_dir),
    )

    print("\nSaved outputs")
    print("-------------")
    print(f"trades_json={trades_json_path}")
    print(f"raw_response_json={raw_response_path}")
    print(f"csv={csv_path}")

    if not args.skip_duckdb:
        load_duckdb(trades_json_path, Path(args.duckdb_path))


if __name__ == "__main__":
    main()
