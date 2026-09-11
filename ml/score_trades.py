from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


MODEL_VERSION = "bargo_native_cluster_pca_v1"
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
CATEGORICAL_COLUMNS = [
    "chamber",
    "state",
    "transaction_type",
    "outcome",
]


def robust_scale(values: pd.DataFrame) -> pd.DataFrame:
    medians = values.median(axis=0)
    q1 = values.quantile(0.25, axis=0)
    q3 = values.quantile(0.75, axis=0)
    iqr = (q3 - q1).replace(0, 1)
    return (values - medians) / iqr


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    numeric = features.reindex(columns=NUMERIC_COLUMNS).copy()
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median()).fillna(0)
    numeric = robust_scale(numeric)

    categorical = features.reindex(columns=CATEGORICAL_COLUMNS).fillna("unknown").astype(str)
    categorical = pd.get_dummies(categorical, columns=CATEGORICAL_COLUMNS, dummy_na=False)

    matrix = pd.concat([numeric, categorical], axis=1)
    return matrix.astype(float)


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
    initial_indices = [
        int(np.argmin(norms)),
        int(np.argmax(norms)),
    ]
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


def pca_reconstruction_error(matrix: np.ndarray, max_components: int = 6) -> np.ndarray:
    if len(matrix) < 2 or matrix.shape[1] < 2:
        return np.zeros(len(matrix))

    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = max(1, min(max_components, vt.shape[0] - 1, matrix.shape[1] - 1))
    components = vt[:n_components]
    reconstructed = centered @ components.T @ components
    return np.mean(np.square(centered - reconstructed), axis=1)


def score(features: pd.DataFrame) -> pd.DataFrame:
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
        )

    feature_matrix = prepare_features(features)
    matrix = feature_matrix.to_numpy(dtype=float)

    labels, _, cluster_distances = deterministic_kmeans(matrix)
    reconstruction_errors = pca_reconstruction_error(matrix)

    cluster_counts = pd.Series(labels).value_counts()
    smallest_cluster = int(cluster_counts.sort_values().index[0])
    cluster_anomaly = labels == smallest_cluster

    cluster_scale = np.percentile(cluster_distances, 95) or 1
    reconstruction_scale = np.percentile(reconstruction_errors, 95) or 1
    normalized_cluster_score = cluster_distances / cluster_scale
    normalized_reconstruction_score = reconstruction_errors / reconstruction_scale
    anomaly_score = 0.45 * normalized_cluster_score + 0.55 * normalized_reconstruction_score

    reconstruction_threshold = np.percentile(reconstruction_errors, 90)
    combined_threshold = np.percentile(anomaly_score, 95)
    is_anomaly = (cluster_anomaly & (reconstruction_errors >= reconstruction_threshold)) | (
        anomaly_score >= combined_threshold
    )

    return pd.DataFrame(
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


def write_scores(con: duckdb.DuckDBPyConnection, scores: pd.DataFrame) -> None:
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
            anomaly_rows integer
        )
        """
    )

    if not scores.empty:
        con.register("scores_df", scores)
        con.sql("insert or replace into model_scores by name select * from scores_df")

    con.execute(
        """
        insert into model_score_runs
        values (?, ?, ?, ?)
        """,
        [
            MODEL_VERSION,
            datetime.now(UTC),
            int(len(scores)),
            int(scores["is_anomaly"].sum()) if not scores.empty else 0,
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


def run(duckdb_path: Path) -> tuple[int, int]:
    con = duckdb.connect(str(duckdb_path))
    try:
        features = con.sql("select * from mart_trade_features").df()
        scores = score(features)
        write_scores(con, scores)
        anomaly_count = int(scores["is_anomaly"].sum()) if not scores.empty else 0
        return len(scores), anomaly_count
    finally:
        con.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Bargo-native trade features for anomalies.")
    parser.add_argument("--duckdb-path", default="congress.duckdb", help="DuckDB database path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scored_rows, anomaly_rows = run(Path(args.duckdb_path))
    print(
        f"Scored {scored_rows} rows with {MODEL_VERSION}; "
        f"flagged {anomaly_rows} anomalies."
    )


if __name__ == "__main__":
    main()
