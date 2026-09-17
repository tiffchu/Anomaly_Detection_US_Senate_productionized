FROM python:3.14-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY pyproject.toml README.md ./
COPY config.py ./
COPY ingestion ./ingestion
COPY storage ./storage
COPY ml ./ml
COPY api ./api
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
