FROM astrocrpublic.azurecr.io/runtime:3.3-7

ENV PYTHONPATH=/usr/local/airflow

COPY requirements.txt ./
COPY ingest.py config.py ./
COPY ingestion ./ingestion
COPY storage ./storage
COPY dbt_project ./dbt_project
COPY api ./api
COPY ml ./ml
