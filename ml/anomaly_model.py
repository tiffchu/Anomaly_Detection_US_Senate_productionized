from __future__ import annotations

from dataclasses import dataclass

import mlflow.pyfunc
import numpy as np
import pandas as pd


@dataclass
class FittedAnomalyModel:
    model_version: str
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_medians: dict[str, float]
    numeric_q1: dict[str, float]
    numeric_iqr: dict[str, float]
    cluster_centers: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    cluster_score_scale: float
    reconstruction_score_scale: float
    reconstruction_threshold: float
    combined_threshold: float
    smallest_cluster: int
    cluster_weight: float
    reconstruction_weight: float

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        numeric = features.reindex(columns=self.numeric_columns).copy()
        numeric = numeric.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(pd.Series(self.numeric_medians)).fillna(0)
        numeric = (numeric - pd.Series(self.numeric_medians)) / pd.Series(self.numeric_iqr)

        categorical = (
            features.reindex(columns=self.categorical_columns)
            .fillna("unknown")
            .astype(str)
        )
        categorical = pd.get_dummies(
            categorical,
            columns=self.categorical_columns,
            dummy_na=False,
        )
        matrix_frame = pd.concat([numeric, categorical], axis=1)
        matrix_frame = matrix_frame.reindex(columns=self.feature_columns, fill_value=0)
        matrix = matrix_frame.to_numpy(dtype=float)

        if not len(matrix):
            return pd.DataFrame(
                columns=[
                    "cluster_label",
                    "cluster_anomaly_score",
                    "reconstruction_error",
                    "anomaly_score",
                    "is_anomaly",
                ]
            )

        all_distances = np.linalg.norm(
            matrix[:, None, :] - self.cluster_centers[None, :, :],
            axis=2,
        )
        labels = np.argmin(all_distances, axis=1)
        cluster_distances = all_distances[np.arange(len(matrix)), labels]

        centered = matrix - self.pca_mean
        if len(self.pca_components):
            reconstructed = centered @ self.pca_components.T @ self.pca_components
            reconstruction_errors = np.mean(
                np.square(centered - reconstructed),
                axis=1,
            )
        else:
            reconstruction_errors = np.zeros(len(matrix))

        anomaly_score = (
            self.cluster_weight * cluster_distances / self.cluster_score_scale
            + self.reconstruction_weight
            * reconstruction_errors
            / self.reconstruction_score_scale
        )
        is_anomaly = (
            (labels == self.smallest_cluster)
            & (reconstruction_errors >= self.reconstruction_threshold)
        ) | (anomaly_score >= self.combined_threshold)

        return pd.DataFrame(
            {
                "cluster_label": labels.astype(int),
                "cluster_anomaly_score": cluster_distances,
                "reconstruction_error": reconstruction_errors,
                "anomaly_score": anomaly_score,
                "is_anomaly": is_anomaly,
            },
            index=features.index,
        )


class BargoAnomalyPythonModel(mlflow.pyfunc.PythonModel):
    def __init__(self, fitted_model: FittedAnomalyModel) -> None:
        self.fitted_model = fitted_model

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        del context, params
        return self.fitted_model.predict(model_input)
