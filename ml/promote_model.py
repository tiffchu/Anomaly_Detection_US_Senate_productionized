from __future__ import annotations

import argparse
import os

import mlflow
from mlflow import MlflowClient


DEFAULT_MODEL_NAME = "congress-trade-anomaly-detector"


def promote_run(
    run_id: str,
    tracking_uri: str,
    model_name: str = DEFAULT_MODEL_NAME,
    alias: str = "production",
) -> int:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient()

    run = client.get_run(run_id)
    model_uri = run.data.tags.get("registry_model_uri")
    if not model_uri:
        raise ValueError(
            f"Run {run_id} has no registry_model_uri tag. "
            "Re-run training with the registry-compatible logger first."
        )

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            "source_run_id": run_id,
            "model_version": run.data.params.get("model_version", "unknown"),
        },
    )
    version = int(model_version.version)

    client.set_registered_model_alias(model_name, alias, version)
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(model_name, str(version), "deployment_status", "production")
    print(
        f"Promoted MLflow run {run_id} to {model_name} version {version}; "
        f"stage=Production alias={alias}."
    )
    return version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register an MLflow run and promote it to production."
    )
    parser.add_argument("--run-id", required=True, help="MLflow training run ID to promote.")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
        required=os.getenv("MLFLOW_TRACKING_URI") is None,
        help="MLflow tracking and registry URI.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--alias", default="production")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    promote_run(
        run_id=args.run_id,
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        alias=args.alias,
    )


if __name__ == "__main__":
    main()
