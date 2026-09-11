from __future__ import annotations

import mlflow
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import get_settings


def train_baseline_model(features: pd.DataFrame) -> IsolationForest:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    model = IsolationForest(contamination=0.05, random_state=42)
    with mlflow.start_run(run_name="baseline-isolation-forest"):
        model.fit(features)
        mlflow.log_params({"model_type": "IsolationForest", "contamination": 0.05})
        mlflow.sklearn.log_model(model, artifact_path="model")
    return model


def replace_with_existing_notebook_model() -> None:
    """Marker for porting the original autoencoder + clustering code into a module."""
    raise NotImplementedError(
        "Port the notebook's autoencoder and clustering code here after the core pipeline runs."
    )
