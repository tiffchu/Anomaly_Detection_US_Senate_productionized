FROM apache/airflow:2.9.3-python3.11

USER root
RUN mkdir -p /opt/airflow/data/warehouse && chown -R airflow:0 /opt/airflow/data
USER airflow

COPY pyproject.toml README.md /opt/airflow/project/
COPY config.py /opt/airflow/project/
COPY ingestion /opt/airflow/project/ingestion
COPY storage /opt/airflow/project/storage
COPY ml /opt/airflow/project/ml
COPY api /opt/airflow/project/api
RUN pip install --no-cache-dir -e "/opt/airflow/project[dbt]"

ENV PYTHONPATH=/opt/airflow/project
