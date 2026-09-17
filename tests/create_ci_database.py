from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def create_database(fixture_path: Path, database_path: Path) -> None:
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Fixture file does not exist: {fixture_path}")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.unlink(missing_ok=True)

    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            create table raw_trades as
            select *
            from read_json_auto(?, format = 'array')
            """,
            [str(fixture_path.resolve())],
        )
        row_count = connection.execute("select count(*) from raw_trades").fetchone()[0]

    print(f"Created {database_path} with {row_count} fixture trades.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the DuckDB database used by CI.")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/raw_trades.json"),
    )
    parser.add_argument("--database", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_database(args.fixture, args.database)
