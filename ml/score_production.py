from __future__ import annotations

import argparse
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import mlflow
import pandas as pd

from ml.model_training import default_tracking_uri
from ml.promote_model import DEFAULT_MODEL_NAME

DEFAULT_MODEL_ALIAS = "production"
PREDICTION_COLUMNS = [
    "cluster_label",
    "cluster_anomaly_score",
    "reconstruction_error",
    "anomaly_score",
    "is_anomaly",
]


def ensure_scoring_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.sql(
        """
        create table if not exists model_scores (
            trade_id varchar primary key,
            model_name varchar,
            model_version varchar,
            model_alias varchar,
            source_run_id varchar,
            scored_at timestamp,
            cluster_label integer,
            cluster_anomaly_score double,
            reconstruction_error double,
            anomaly_score double,
            is_anomaly boolean
        )
        """
    )
    for column_name in ["model_name", "model_alias", "source_run_id"]:
        connection.sql(
            f"alter table model_scores add column if not exists {column_name} varchar"
        )

    connection.sql(
        """
        create table if not exists model_score_runs (
            scoring_run_id varchar,
            model_name varchar,
            model_version varchar,
            model_alias varchar,
            source_run_id varchar,
            scored_at timestamp,
            scored_rows integer,
            anomaly_rows integer
        )
        """
    )
    for column_name in [
        "scoring_run_id",
        "model_name",
        "model_alias",
        "source_run_id",
    ]:
        connection.sql(
            f"alter table model_score_runs add column if not exists {column_name} varchar"
        )


def load_features_to_score(
    connection: duckdb.DuckDBPyConnection,
    model_name: str,
    model_version: str,
) -> pd.DataFrame:
    return connection.execute(
        """
        select f.*
        from mart_trade_features f
        left join model_scores s using (trade_id)
        where s.trade_id is null
           or coalesce(s.model_name, '') <> ?
           or coalesce(s.model_version, '') <> ?
        order by f.trade_id
        """,
        [model_name, model_version],
    ).fetchdf()


def score_features(
    model: Any,
    features: pd.DataFrame,
    model_name: str,
    model_version: str,
    model_alias: str,
    source_run_id: str,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()

    predictions = model.predict(features)
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)

    missing_columns = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"Production model did not return columns: {missing_columns}")
    if len(predictions) != len(features):
        raise ValueError("Production model returned a different number of rows than it received.")

    scores = predictions[PREDICTION_COLUMNS].reset_index(drop=True).copy()
    scores.insert(0, "scored_at", datetime.now(UTC))
    scores.insert(0, "source_run_id", source_run_id)
    scores.insert(0, "model_alias", model_alias)
    scores.insert(0, "model_version", model_version)
    scores.insert(0, "model_name", model_name)
    scores.insert(0, "trade_id", features["trade_id"].reset_index(drop=True))
    return scores


def write_scores(
    connection: duckdb.DuckDBPyConnection,
    scores: pd.DataFrame,
    model_name: str,
    model_version: str,
    model_alias: str,
    source_run_id: str,
) -> None:
    if not scores.empty:
        connection.register("production_scores", scores)
        connection.sql(
            "insert or replace into model_scores by name select * from production_scores"
        )

    connection.execute(
        """
        insert into model_score_runs (
            scoring_run_id,
            model_name,
            model_version,
            model_alias,
            source_run_id,
            scored_at,
            scored_rows,
            anomaly_rows
        ) values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            str(uuid.uuid4()),
            model_name,
            model_version,
            model_alias,
            source_run_id,
            datetime.now(UTC),
            int(len(scores)),
            int(scores["is_anomaly"].sum()) if not scores.empty else 0,
        ],
    )

    connection.sql(
        """
        create or replace table mart_flagged_trades as
        select
            f.*,
            s.model_name,
            s.model_version,
            s.model_alias,
            s.source_run_id,
            s.scored_at,
            s.cluster_label,
            s.cluster_anomaly_score,
            s.reconstruction_error,
            s.anomaly_score,
            s.is_anomaly
        from mart_trade_features f
        join model_scores s using (trade_id)
        where s.is_anomaly
        order by s.anomaly_score desc, f.disclosure_date desc
        """
    )


def score_production_model(
    duckdb_path: Path,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    model_alias: str = DEFAULT_MODEL_ALIAS,
) -> tuple[int, int, str, str]:
    resolved_tracking_uri = (
        tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or default_tracking_uri(duckdb_path)
    )
    resolved_registry_uri = (
        registry_uri or os.getenv("MLFLOW_REGISTRY_URI") or resolved_tracking_uri
    )
    mlflow.set_tracking_uri(resolved_tracking_uri)
    mlflow.set_registry_uri(resolved_registry_uri)

    client = mlflow.MlflowClient()
    registered_model = client.get_model_version_by_alias(model_name, model_alias)
    model_version = str(registered_model.version)
    source_run_id = registered_model.run_id
    model = mlflow.pyfunc.load_model(registered_model.source)

    with duckdb.connect(str(duckdb_path)) as connection:
        ensure_scoring_tables(connection)
        features = load_features_to_score(connection, model_name, model_version)
        scores = score_features(
            model,
            features,
            model_name=model_name,
            model_version=model_version,
            model_alias=model_alias,
            source_run_id=source_run_id,
        )
        write_scores(
            connection,
            scores,
            model_name=model_name,
            model_version=model_version,
            model_alias=model_alias,
            source_run_id=source_run_id,
        )

    anomaly_rows = int(scores["is_anomaly"].sum()) if not scores.empty else 0
    return len(scores), anomaly_rows, model_version, source_run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score new trades with the MLflow production model."
    )
    parser.add_argument("--duckdb-path", default="congress.duckdb")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--registry-uri", default=None)
    parser.add_argument(
        "--model-name",
        default=os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME),
    )
    parser.add_argument(
        "--model-alias",
        default=os.getenv("MLFLOW_MODEL_ALIAS", DEFAULT_MODEL_ALIAS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scored_rows, anomaly_rows, model_version, source_run_id = score_production_model(
        Path(args.duckdb_path),
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
        model_name=args.model_name,
        model_alias=args.model_alias,
    )
    print(
        f"Scored {scored_rows} rows with production model version {model_version}; "
        f"flagged {anomaly_rows} anomalies; source run {source_run_id}."
    )


if __name__ == "__main__":
    main()
