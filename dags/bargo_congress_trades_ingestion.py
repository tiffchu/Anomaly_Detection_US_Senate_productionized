from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_HOME = "/usr/local/airflow"
DATA_HOME = f"{PROJECT_HOME}/include/data"
DUCKDB_PATH = f"{DATA_HOME}/congress.duckdb"
MLFLOW_TRACKING_URI = f"sqlite:///{DATA_HOME}/mlflow.db"
MODEL_NAME = "congress-trade-anomaly-detector"
MODEL_ALIAS = "production"


with DAG(
    dag_id="bargo_congress_trades_ingestion",
    description="Pull Bargo trades, build dbt models, then run production scoring.",
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
            f"--output-dir {DATA_HOME}/raw "
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

    score_with_production_model = BashOperator(
        task_id="score_with_production_model",
        cwd=PROJECT_HOME,
        bash_command=(
            "python -m ml.score_production "
            f"--duckdb-path {DUCKDB_PATH} "
            f"--tracking-uri {MLFLOW_TRACKING_URI} "
            f"--registry-uri {MLFLOW_TRACKING_URI} "
            f"--model-name {MODEL_NAME} "
            f"--model-alias {MODEL_ALIAS}"
        ),
        env={
            "PYTHONPATH": PROJECT_HOME,
            "DUCKDB_PATH": DUCKDB_PATH,
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "MLFLOW_REGISTRY_URI": MLFLOW_TRACKING_URI,
            "MLFLOW_MODEL_NAME": MODEL_NAME,
            "MLFLOW_MODEL_ALIAS": MODEL_ALIAS,
            "GIT_PYTHON_REFRESH": "quiet",
        },
        append_env=True,
    )

    ingest_bargo_trades >> run_dbt >> score_with_production_model
