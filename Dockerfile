FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib \
    METADATA_PATH=/app/models/metadata.json \
    LOG_DIR=/app/logs

WORKDIR /app

# The serving image only needs the runtime libraries, not MLflow / notebooks.
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src ./src
COPY api ./api
COPY models ./models

RUN useradd --create-home appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
