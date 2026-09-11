from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


PROJECT_HOME = "/usr/local/airflow"
DUCKDB_PATH = f"{PROJECT_HOME}/data/congress.duckdb"


with DAG(
    dag_id="bargo_congress_trades_ingestion",
    description="Pull Bargo Congress trades, build dbt models, then score anomalies.",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={
        "owner": "data-eng",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["bargo", "duckdb", "dbt"],
) as dag:
    ingest_bargo_trades = BashOperator(
        task_id="ingest_bargo_trades",
        cwd=PROJECT_HOME,
        bash_command=(
            "python ingest.py "
            "--limit 100 "
            "--pages 10 "
            "--output-dir data/raw "
            "--allow-stale-on-rate-limit "
            f"--duckdb-path {DUCKDB_PATH}"
        ),
        env={
            "PYTHONPATH": PROJECT_HOME,
            "DUCKDB_PATH": DUCKDB_PATH,
            "DBT_DUCKDB_PATH": DUCKDB_PATH,
        },
        append_env=True,
    )

    run_dbt = BashOperator(
        task_id="run_dbt",
        cwd=f"{PROJECT_HOME}/dbt_project",
        bash_command="dbt build --profiles-dir .",
        env={
            "DBT_DUCKDB_PATH": DUCKDB_PATH,
        },
        append_env=True,
    )

    score_anomalies = BashOperator(
        task_id="score_anomalies",
        cwd=PROJECT_HOME,
        bash_command=f"python -m ml.score_trades --duckdb-path {DUCKDB_PATH}",
        env={
            "PYTHONPATH": PROJECT_HOME,
            "DUCKDB_PATH": DUCKDB_PATH,
        },
        append_env=True,
    )

    ingest_bargo_trades >> run_dbt >> score_anomalies
