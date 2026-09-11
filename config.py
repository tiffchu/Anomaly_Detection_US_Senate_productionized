from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bargo_api_key: str | None = Field(default=None, alias="BARGO_API_KEY")
    bargo_base_url: str = Field(
        default="https://www.bargo.ai/free-apis/congress/v1",
        alias="BARGO_BASE_URL",
    )
    bargo_page_limit: int = Field(default=100, alias="BARGO_PAGE_LIMIT")

    warehouse_backend: str = Field(default="duckdb", alias="WAREHOUSE_BACKEND")
    duckdb_path: str = Field(default="./data/warehouse/senate_trades.duckdb", alias="DUCKDB_PATH")

    gcp_project_id: str | None = Field(default=None, alias="GCP_PROJECT_ID")
    bigquery_dataset: str = Field(default="senate_trades", alias="BIGQUERY_DATASET")

    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(
        default="senate-trade-anomaly-detection",
        alias="MLFLOW_EXPERIMENT_NAME",
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()
