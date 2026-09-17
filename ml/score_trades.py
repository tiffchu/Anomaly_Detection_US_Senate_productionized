from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import mlflow
import numpy as np
import pandas as pd

from ml.anomaly_model import FittedAnomalyModel


MODEL_VERSION = "bargo_native_cluster_pca_v2"
DEFAULT_EXPERIMENT_NAME = "senate-trade-anomaly-detection"
NUMERIC_COLUMNS = [
    "amount_low",
    "amount_high",
    "disclosure_lag_days",
    "estimated_trade_price",
    "recent_price",
    "performance_pct",
    "signed_amount_high",
    "member_avg_amount_high",
    "member_running_net_amount",
    "member_trade_count",
    "ticker_trade_count",
]
CATEGORICAL_COLUMNS = ["chamber", "state", "transaction_type", "outcome"]


def prepare_features(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    numeric = features.reindex(columns=NUMERIC_COLUMNS).copy()
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    medians = numeric.median(axis=0).fillna(0)
    numeric = numeric.fillna(medians)
    q1 = numeric.quantile(0.25, axis=0)
    q3 = numeric.quantile(0.75, axis=0)
    iqr = (q3 - q1).replace(0, 1)
    numeric = (numeric - medians) / iqr

    categorical = features.reindex(columns=CATEGORICAL_COLUMNS).fillna("unknown").astype(str)
    categorical = pd.get_dummies(categorical, columns=CATEGORICAL_COLUMNS, dummy_na=False)

    matrix = pd.concat([numeric, categorical], axis=1).astype(float)
    scaling = {
        "medians": {column: float(value) for column, value in medians.items()},
        "q1": {column: float(value) for column, value in q1.items()},
        "iqr": {column: float(value) for column, value in iqr.items()},
    }
    return matrix, scaling


def deterministic_kmeans(
    matrix: np.ndarray,
    n_clusters: int = 2,
    max_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(matrix) < n_clusters:
        labels = np.zeros(len(matrix), dtype=int)
        centers = matrix[:1] if len(matrix) else np.empty((0, matrix.shape[1]))
        distances = np.zeros(len(matrix))
        return labels, centers, distances

    norms = np.linalg.norm(matrix, axis=1)
    initial_indices = [int(np.argmin(norms)), int(np.argmax(norms))]
    centers = matrix[initial_indices].copy()

    labels = np.zeros(len(matrix), dtype=int)
    for _ in range(max_iter):
        all_distances = np.linalg.norm(matrix[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(all_distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(n_clusters):
            cluster_points = matrix[labels == cluster_id]
            if len(cluster_points):
                centers[cluster_id] = cluster_points.mean(axis=0)

    final_distances = np.linalg.norm(matrix - centers[labels], axis=1)
    return labels, centers, final_distances


def fit_pca(
    matrix: np.ndarray,
    max_components: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0) if len(matrix) else np.zeros(matrix.shape[1])
    if len(matrix) < 2 or matrix.shape[1] < 2:
        return mean, np.empty((0, matrix.shape[1])), np.zeros(len(matrix))

    centered = matrix - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = max(1, min(max_components, vt.shape[0] - 1, matrix.shape[1] - 1))
    components = vt[:n_components]
    reconstructed = centered @ components.T @ components
    errors = np.mean(np.square(centered - reconstructed), axis=1)
    return mean, components, errors


def fit_and_score(
    features: pd.DataFrame,
    n_clusters: int = 2,
    max_kmeans_iterations: int = 50,
    max_pca_components: int = 6,
    cluster_weight: float = 0.45,
    reconstruction_weight: float = 0.55,
    reconstruction_percentile: float = 90,
    combined_percentile: float = 95,
) -> tuple[pd.DataFrame, FittedAnomalyModel | None]:
    if features.empty:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "model_version",
                "scored_at",
                "cluster_label",
                "cluster_anomaly_score",
                "reconstruction_error",
                "anomaly_score",
                "is_anomaly",
            ]
        ), None

    feature_matrix, scaling = prepare_features(features)
    matrix = feature_matrix.to_numpy(dtype=float)
    labels, centers, cluster_distances = deterministic_kmeans(
        matrix, n_clusters=n_clusters, max_iter=max_kmeans_iterations
    )
    pca_mean, pca_components, reconstruction_errors = fit_pca(
        matrix, max_components=max_pca_components
    )

    cluster_counts = pd.Series(labels).value_counts()
    smallest_cluster = int(cluster_counts.sort_values().index[0])
    cluster_anomaly = labels == smallest_cluster
    cluster_scale = float(np.percentile(cluster_distances, 95) or 1)
    reconstruction_scale = float(np.percentile(reconstruction_errors, 95) or 1)
    anomaly_score = (
        cluster_weight * cluster_distances / cluster_scale
        + reconstruction_weight * reconstruction_errors / reconstruction_scale
    )
    reconstruction_threshold = float(
        np.percentile(reconstruction_errors, reconstruction_percentile)
    )
    combined_threshold = float(np.percentile(anomaly_score, combined_percentile))
    is_anomaly = (cluster_anomaly & (reconstruction_errors >= reconstruction_threshold)) | (
        anomaly_score >= combined_threshold
    )

    model = FittedAnomalyModel(
        model_version=MODEL_VERSION,
        feature_columns=list(feature_matrix.columns),
        numeric_columns=NUMERIC_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        numeric_medians=scaling["medians"],
        numeric_q1=scaling["q1"],
        numeric_iqr=scaling["iqr"],
        cluster_centers=centers,
        pca_mean=pca_mean,
        pca_components=pca_components,
        cluster_score_scale=cluster_scale,
        reconstruction_score_scale=reconstruction_scale,
        reconstruction_threshold=reconstruction_threshold,
        combined_threshold=combined_threshold,
        smallest_cluster=smallest_cluster,
        cluster_weight=cluster_weight,
        reconstruction_weight=reconstruction_weight,
    )
    scores = pd.DataFrame(
        {
            "trade_id": features["trade_id"].values,
            "model_version": MODEL_VERSION,
            "scored_at": datetime.now(UTC),
            "cluster_label": labels.astype(int),
            "cluster_anomaly_score": cluster_distances,
            "reconstruction_error": reconstruction_errors,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
        }
    )
    return scores, model


def log_mlflow_run(
    model: FittedAnomalyModel | None,
    scores: pd.DataFrame,
    tracking_uri: str,
    artifact_root: str,
    experiment_name: str,
) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=artifact_root)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=MODEL_VERSION) as active_run:
        mlflow.log_params(
            {
                "model_version": MODEL_VERSION,
                "model_type": "deterministic_kmeans_pca_reconstruction",
                "n_clusters": 2,
                "max_kmeans_iterations": 50,
                "max_pca_components": 6,
                "cluster_weight": 0.45,
                "reconstruction_weight": 0.55,
                "reconstruction_percentile": 90,
                "combined_percentile": 95,
                "numeric_feature_count": len(NUMERIC_COLUMNS),
                "categorical_feature_count": len(CATEGORICAL_COLUMNS),
            }
        )

        anomaly_rows = int(scores["is_anomaly"].sum()) if not scores.empty else 0
        metrics = {
            "scored_rows": float(len(scores)),
            "anomaly_rows": float(anomaly_rows),
            "anomaly_rate": float(anomaly_rows / len(scores)) if len(scores) else 0.0,
        }
        if not scores.empty:
            metrics.update(
                {
                    "mean_anomaly_score": float(scores["anomaly_score"].mean()),
                    "max_anomaly_score": float(scores["anomaly_score"].max()),
                    "mean_reconstruction_error": float(
                        scores["reconstruction_error"].mean()
                    ),
                    "max_reconstruction_error": float(
                        scores["reconstruction_error"].max()
                    ),
                }
            )
        mlflow.log_metrics(metrics)

        if model is not None:
            metadata = asdict(model)
            for key in ["cluster_centers", "pca_mean", "pca_components"]:
                metadata[key] = metadata[key].tolist()
            mlflow.log_dict(metadata, "model/model_metadata.json")

            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = Path(temp_dir) / "anomaly_model.pkl"
                with model_path.open("wb") as model_file:
                    pickle.dump(model, model_file)
                mlflow.log_artifact(str(model_path), artifact_path="model")

        return active_run.info.run_id


def write_scores(
    con: duckdb.DuckDBPyConnection,
    scores: pd.DataFrame,
    mlflow_run_id: str,
) -> None:
    con.sql(
        """
        create table if not exists model_scores (
            trade_id varchar primary key,
            model_version varchar,
            scored_at timestamp,
            cluster_label integer,
            cluster_anomaly_score double,
            reconstruction_error double,
            anomaly_score double,
            is_anomaly boolean
        )
        """
    )
    con.sql(
        """
        create table if not exists model_score_runs (
            model_version varchar,
            scored_at timestamp,
            scored_rows integer,
            anomaly_rows integer,
            mlflow_run_id varchar
        )
        """
    )
    con.sql("alter table model_score_runs add column if not exists mlflow_run_id varchar")

    if not scores.empty:
        con.register("scores_df", scores)
        con.sql("insert or replace into model_scores by name select * from scores_df")

    con.execute(
        """
        insert into model_score_runs (
            model_version, scored_at, scored_rows, anomaly_rows, mlflow_run_id
        ) values (?, ?, ?, ?, ?)
        """,
        [
            MODEL_VERSION,
            datetime.now(UTC),
            int(len(scores)),
            int(scores["is_anomaly"].sum()) if not scores.empty else 0,
            mlflow_run_id,
        ],
    )

    con.sql(
        """
        create or replace table mart_flagged_trades as
        select
            f.*,
            s.model_version,
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


def default_tracking_uri(duckdb_path: Path) -> str:
    database_path = duckdb_path.resolve().parent / "mlflow.db"
    return f"sqlite:///{database_path.as_posix()}"


def default_artifact_root(duckdb_path: Path) -> str:
    return (duckdb_path.resolve().parent / "mlartifacts").as_uri()


def run(
    duckdb_path: Path,
    tracking_uri: str | None = None,
    artifact_root: str | None = None,
    experiment_name: str | None = None,
) -> tuple[int, int, str]:
    resolved_tracking_uri = (
        tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or default_tracking_uri(duckdb_path)
    )
    resolved_experiment_name = (
        experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT_NAME
    )
    resolved_artifact_root = (
        artifact_root
        or os.getenv("MLFLOW_ARTIFACT_ROOT")
        or default_artifact_root(duckdb_path)
    )

    con = duckdb.connect(str(duckdb_path))
    try:
        features = con.sql("select * from mart_trade_features").df()
        scores, model = fit_and_score(features)
        mlflow_run_id = log_mlflow_run(
            model,
            scores,
            tracking_uri=resolved_tracking_uri,
            artifact_root=resolved_artifact_root,
            experiment_name=resolved_experiment_name,
        )
        write_scores(con, scores, mlflow_run_id)
        anomaly_count = int(scores["is_anomaly"].sum()) if not scores.empty else 0
        return len(scores), anomaly_count, mlflow_run_id
    finally:
        con.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and score Bargo-native trade features, with MLflow tracking."
    )
    parser.add_argument("--duckdb-path", default="congress.duckdb", help="DuckDB database path.")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI or a local mlruns directory.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name. Defaults to MLFLOW_EXPERIMENT_NAME.",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Artifact URI. Defaults to MLFLOW_ARTIFACT_ROOT or a local mlartifacts directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scored_rows, anomaly_rows, mlflow_run_id = run(
        Path(args.duckdb_path),
        tracking_uri=args.tracking_uri,
        artifact_root=args.artifact_root,
        experiment_name=args.experiment_name,
    )
    print(
        f"Scored {scored_rows} rows with {MODEL_VERSION}; "
        f"flagged {anomaly_rows} anomalies; MLflow run {mlflow_run_id}."
    )


if __name__ == "__main__":
    main()
