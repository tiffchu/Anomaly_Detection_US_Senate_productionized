from __future__ import annotations

import duckdb
import pandas as pd

from ml.model_training import train_candidate
from ml.score_production import (
    ensure_scoring_tables,
    load_features_to_score,
    score_features,
    write_scores,
)


class FakeProductionModel:
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        anomaly_score = features["amount_high"] / 10_000
        return pd.DataFrame(
            {
                "cluster_label": [0, 1],
                "cluster_anomaly_score": [0.25, 2.5],
                "reconstruction_error": [0.1, 1.2],
                "anomaly_score": anomaly_score,
                "is_anomaly": anomaly_score >= 2,
            }
        )


def create_feature_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.sql(
        """
        create table mart_trade_features as
        select *
        from (
            values
                ('trade-1', 5000.0, date '2026-01-05'),
                ('trade-2', 50000.0, date '2026-01-06')
        ) as trades(trade_id, amount_high, disclosure_date)
        """
    )


def test_scoring_is_idempotent_and_rescores_for_a_new_model_version() -> None:
    connection = duckdb.connect(":memory:")
    create_feature_table(connection)
    ensure_scoring_tables(connection)

    features = load_features_to_score(connection, "detector", "1")
    scores = score_features(
        FakeProductionModel(),
        features,
        model_name="detector",
        model_version="1",
        model_alias="production",
        source_run_id="run-1",
    )
    write_scores(
        connection,
        scores,
        model_name="detector",
        model_version="1",
        model_alias="production",
        source_run_id="run-1",
    )

    assert len(scores) == 2
    assert connection.sql("select count(*) from mart_flagged_trades").fetchone()[0] == 1
    assert load_features_to_score(connection, "detector", "1").empty
    assert len(load_features_to_score(connection, "detector", "2")) == 2


def test_scoring_rejects_an_incompatible_registered_model() -> None:
    class IncompatibleModel:
        def predict(self, features: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"score": [0.1] * len(features)})

    features = pd.DataFrame({"trade_id": ["trade-1"], "amount_high": [5000.0]})

    try:
        score_features(
            IncompatibleModel(),
            features,
            model_name="detector",
            model_version="1",
            model_alias="production",
            source_run_id="run-1",
        )
    except ValueError as error:
        assert "did not return columns" in str(error)
    else:
        raise AssertionError("Expected incompatible model output to be rejected.")


def test_candidate_training_does_not_write_production_scores(
    tmp_path, monkeypatch
) -> None:
    database_path = tmp_path / "training.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.sql(
            """
            create table mart_trade_features as
            select *
            from (
                values
                    ('trade-1', 5000.0, 'senate', 'purchase'),
                    ('trade-2', 15000.0, 'senate', 'sale'),
                    ('trade-3', 50000.0, 'house', 'purchase')
            ) as trades(trade_id, amount_high, chamber, transaction_type)
            """
        )

    monkeypatch.setattr(
        "ml.model_training.log_mlflow_run",
        lambda *args, **kwargs: "candidate-run",
    )
    training_rows, _, run_id = train_candidate(
        database_path,
        tracking_uri="sqlite:///unused.db",
        artifact_root="file:///unused-artifacts",
    )

    with duckdb.connect(str(database_path), read_only=True) as connection:
        tables = {
            row[0]
            for row in connection.sql(
                "select table_name from information_schema.tables"
            ).fetchall()
        }

    assert training_rows == 3
    assert run_id == "candidate-run"
    assert "model_scores" not in tables
