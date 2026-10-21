from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_HOME = "/usr/local/airflow"
DATA_HOME = f"{PROJECT_HOME}/include/data"
DUCKDB_PATH = f"{DATA_HOME}/congress.duckdb"
MLFLOW_TRACKING_URI = f"sqlite:///{DATA_HOME}/mlflow.db"
MLFLOW_ARTIFACT_ROOT = f"file://{DATA_HOME}/mlartifacts"


with DAG(
    dag_id="bargo_anomaly_model_training",
    description="Train and log a weekly candidate model without automatic promotion.",
    start_date=datetime(2026, 1, 1),
    schedule="0 4 * * 0",
    catchup=False,
    default_args={
        "owner": "ml-eng",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["bargo", "mlflow", "training"],
) as dag:
    train_candidate_model = BashOperator(
        task_id="train_candidate_model",
        cwd=PROJECT_HOME,
        bash_command=(
            "python -m ml.train "
            f"--duckdb-path {DUCKDB_PATH} "
            f"--tracking-uri {MLFLOW_TRACKING_URI} "
            f"--artifact-root {MLFLOW_ARTIFACT_ROOT}"
        ),
        env={
            "PYTHONPATH": PROJECT_HOME,
            "DUCKDB_PATH": DUCKDB_PATH,
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "MLFLOW_ARTIFACT_ROOT": MLFLOW_ARTIFACT_ROOT,
            "MLFLOW_EXPERIMENT_NAME": "senate-trade-anomaly-detection",
            "GIT_PYTHON_REFRESH": "quiet",
        },
        append_env=True,
    )
