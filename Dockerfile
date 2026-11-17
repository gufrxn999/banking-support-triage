# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Generate bytecode at build time is pointless in a container; keep logs
# unbuffered so `docker logs` shows the request line immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    TRIAGE_DATA_DIR=/app/data \
    TRIAGE_ARTIFACTS_DIR=/app/artifacts \
    TRIAGE_REPORTS_DIR=/app/reports

WORKDIR /app

# Install dependencies from the lock before copying source: the expensive layer
# then survives application changes, and the image uses the same versions CI
# rebuilds the reports with.
COPY pyproject.toml README.md LICENSE requirements.lock ./
RUN pip install -r requirements.lock

COPY src ./src
COPY streamlit_app.py ./
RUN pip install . --no-deps

# Curated content is committed. The model is not: it is built here so the image
# is self-contained and the artifact cannot drift from the code that serves it.
COPY data/faq.jsonl data/probes.jsonl ./data/
RUN python -m banking_triage.data \
    && python -m banking_triage.train \
    && python -m banking_triage.calibrate

RUN useradd --create-home --uid 10001 triage \
    && chown -R triage:triage /app
USER triage

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2)"

# Shell form so a platform-assigned PORT (Render, Railway, Fly) is honored;
# 8000 remains the local/compose default.
CMD ["sh", "-c", "exec uvicorn banking_triage.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
