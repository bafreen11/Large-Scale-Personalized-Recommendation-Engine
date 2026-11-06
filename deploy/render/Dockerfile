# Render web service for the recommendation API (Docker runtime, free plan).
# Build context is the repository root:  Render dashboard -> New -> Blueprint -> render.yaml.
#
# src/ comes straight from the repository; the models and the slim training data come from
# deploy/render/artifacts/, staged by scripts/stage_render_artifacts.py.
# No Redis / Triton here: the API falls back to an in-memory cache and in-process models.
FROM python:3.11-slim

WORKDIR /app

COPY deploy/huggingface/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# src/common/config.py resolves ROOT from src/, so models/ and data/ sit next to it under /app
COPY src/ ./src/
COPY deploy/render/artifacts/models/ ./models/
COPY deploy/render/artifacts/data/ ./data/

ENV PYTHONPATH=/app TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1

EXPOSE 8080

# Render passes the port in $PORT; 8080 is the local default. One worker: the free plan has 512 MB.
CMD exec uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
