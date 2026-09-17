# US Senate Trading Anomaly Pipeline

Production template for the original notebook project: scheduled congressional-trade ingestion, warehouse layers, dbt transforms, anomaly scoring, MLflow tracking, FastAPI serving, dashboard-ready marts, Docker, and CI.

The first implementation target is intentionally boring and defensible:

1. Airflow pulls Bargo's Congress Trades REST API on a schedule.
2. Raw JSON lands in DuckDB for local development, with BigQuery as the planned cloud warehouse.
3. dbt cleans raw rows into staging and dashboard marts.
4. The existing anomaly detection workflow is wrapped as a repeatable scoring job.
5. FastAPI exposes health, latest flagged trades, and on-demand scoring.

## Architecture

```text
Bargo REST API
    |
    v
Airflow DAG: bargo_congress_trades_ingestion
    |
    v
DuckDB raw tables / BigQuery raw dataset
    |
    v
dbt staging + mart models
    |
    v
ML training/scoring + MLflow
    |
    v
FastAPI + dashboard extracts
```

Use Bargo's REST API for scheduled ingestion because it is deterministic and easy to audit. Use the Bargo MCP server separately for ad hoc research, diagnostics, and natural-language exploration.

## Repository Layout

```text
ingestion/               Bargo REST client and ingestion CLI
dbt_project/             dbt project for raw, staging, and mart transforms
dags/                    Airflow DAGs
ml/                      Training and scoring code
api/                     FastAPI app
storage/                 Warehouse adapters
Dockerfile               Astro Runtime image for local Airflow
airflow_settings.yaml    Local-only Astro/Airflow settings
docker/                  Dockerfiles for API and Airflow
tests/                   Unit tests
.github/workflows/       CI checks
```

## What You Need

- Python 3.11+
- Docker Desktop
- Astro CLI for local Airflow development
- A Bargo API key for higher limits and MCP access
- Optional: Google Cloud project, BigQuery dataset, and service account JSON
- Optional: Tableau or Power BI Desktop for dashboarding

## Quick Start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Update `.env` with your Bargo key and storage choice. For local development, keep `WAREHOUSE_BACKEND=duckdb`.

Run a one-off ingestion:

```powershell
python -m ingestion.cli ingest-bargo --from-date 2026-01-01 --to-date 2026-09-09
```

Before wiring ingestion into Airflow, validate the source with the standalone script:

```powershell
python ingest.py --limit 10 --pages 2
```

This writes `data/raw/trades.json`, a timestamped raw response envelope, a CSV extract, and `congress.duckdb`. The script also creates a `raw_trades` table with DuckDB's `read_json_auto` and prints basic SQL checks for row counts, nulls, duplicate candidates, transaction types, and inferred column types.

The ingest is idempotent: it computes a stable `trade_id`, stages each pull, then `insert or replace`s into DuckDB using `trade_id` as the primary key. Re-running the same pull refreshes rows instead of duplicating them.

Run dbt:

```powershell
cd dbt_project
dbt debug --profiles-dir .
dbt run --profiles-dir .
dbt test --profiles-dir .
```

Or run the model plus tests together:

```powershell
dbt build --profiles-dir .
```

Run Airflow locally with Astro:

```powershell
astro dev init
astro dev start
```

This repo already includes the files Astro expects (`Dockerfile`, `dags/`, `requirements.txt`, `packages.txt`, and `airflow_settings.yaml`). If `astro dev init` says the folder is not empty, skip it and run `astro dev start` from the repo root.

The daily DAG is `bargo_congress_trades_ingestion`:

1. `ingest_bargo_trades` calls `python ingest.py` for the scheduled date window.
2. `run_dbt` runs `dbt build --profiles-dir .`.
3. `score_anomalies` trains the clustering/PCA anomaly model, logs its parameters,
   metrics, and serialized fitted state to MLflow, then writes scores to DuckDB.

For Astro, keep `BARGO_API_KEY` in the root `.env` file. The DAG writes to
`/usr/local/airflow/include/data/congress.duckdb`; Astro mounts `include`, so the database
and MLflow history survive container rebuilds. The DAG sets `DBT_DUCKDB_PATH` so dbt reads
the same file.

MLflow run metadata is stored in `include/data/mlflow.db` and artifacts in
`include/data/mlartifacts`, so no hosted MLflow server or additional secret is required for
local development. To inspect the runs outside Airflow:

```powershell
mlflow ui --backend-store-uri sqlite:///include/data/mlflow.db --port 5000
```

Then open `http://localhost:5000`. For a remote tracking server later, set
`MLFLOW_TRACKING_URI` and any credentials required by that server.

Promote a verified run to the local model registry:

```powershell
python -m ml.promote_model `
  --run-id <MLFLOW_RUN_ID> `
  --tracking-uri sqlite:///include/data/mlflow.db
```

The command creates a version of `congress-trade-anomaly-detector`, assigns the modern
`production` alias, and also marks the version with MLflow's legacy `Production` stage.

Start the production-model API:

```powershell
docker compose up --build -d
```

Compose starts the MLflow tracking/registry server on `http://localhost:5000` and the API on
`http://localhost:8000`. The API waits for MLflow to become healthy, then loads
`congress-trade-anomaly-detector@production` once during startup. Airflow remains managed by
Astro and is intentionally not duplicated in this Compose stack.

Score one Bargo-native feature record:

```powershell
curl.exe -X POST http://localhost:8000/score `
  -H "Content-Type: application/json" `
  -d '{"amount_low":1001,"amount_high":15000,"disclosure_lag_days":30,"signed_amount_high":15000,"member_avg_amount_high":15000,"member_running_net_amount":15000,"member_trade_count":1,"ticker_trade_count":1,"chamber":"senate","state":"CA","transaction_type":"purchase","outcome":"gain"}'
```

The response includes the anomaly score and flag plus the registry model version and source
run ID used for inference. Changing the registry's `production` alias takes effect the next
time the API starts.

## Pull Request CI

The GitHub Actions workflow runs Python linting and a full `dbt build` for every pull
request. It creates a temporary DuckDB database from the committed fixture in
`tests/fixtures/raw_trades.json`, so CI never calls Bargo and does not require API secrets.

After a pull request is merged into `main`, a separate workflow builds
`docker/api.Dockerfile` and publishes the FastAPI image to GHCR as both `latest` and an
immutable commit-SHA tag. It authenticates with the repository's built-in `GITHUB_TOKEN`;
no Docker Hub credentials or additional GitHub secrets are needed.

Run the same checks locally from the repository root:

```powershell
python -m pip install -r requirements-ci.txt
ruff check .
python tests/create_ci_database.py --database .tmp/ci.duckdb
$env:DBT_DUCKDB_PATH = (Resolve-Path .tmp/ci.duckdb)
dbt build --project-dir dbt_project --profiles-dir dbt_project
```

## Dashboard Direction

For this project, start with Tableau Public or Power BI connected to the mart table/export `mart_flagged_trades`.

Recommended first dashboard:

- KPI row: total trades, flagged trades, flag rate, latest disclosure date
- Time series: flagged trades over transaction and disclosure dates
- Breakdown: member, ticker, chamber, transaction type
- Detail table: flagged trade, anomaly score, model version, source filing URL

## Suggested Sequence

1. Finish the core path locally: Airflow -> DuckDB -> dbt -> anomaly scoring.
2. Move raw/staging/mart tables to BigQuery when local semantics are stable.
3. Add MLflow experiment tracking around the existing autoencoder and clustering code.
4. Add FastAPI model endpoints and Docker images.
5. Add dashboard refresh and CI/CD hardening.

## Data Notes

Bargo's free Congress Trades API currently exposes recent disclosed STOCK Act trades via REST, including fields such as member, chamber, ticker, transaction type, amount range, transaction date, disclosure date, estimated trade price, recent price, and performance percentage. Disclosures can lag transactions, so dashboard language should describe results as "disclosed trades" rather than real-time trades.

built with a lot of help from chatgpt <3
