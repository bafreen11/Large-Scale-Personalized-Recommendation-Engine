# Hugging Face Space (Docker SDK) for the recommendation API.
# Built from the folder that scripts/build_hf_space.py assembles: src/, models/, data/training/ (slim).
# No Redis / Triton here: the API falls back to an in-memory cache and in-process models.
FROM python:3.11-slim

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

ENV PYTHONPATH=/app TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "7860"]
