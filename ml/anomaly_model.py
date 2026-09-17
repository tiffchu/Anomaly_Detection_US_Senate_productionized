from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
