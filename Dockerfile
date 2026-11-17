# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Generate bytecode at build time is pointless in a container; keep logs
# unbuffered so `docker logs` shows the request line immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TRIAGE_DATA_DIR=/app/data \
    TRIAGE_ARTIFACTS_DIR=/app/artifacts \
    TRIAGE_REPORTS_DIR=/app/reports

WORKDIR /app

# Install before copying the rest of the source: dependency layers change far
# less often than application code.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY streamlit_app.py ./
RUN pip install ".[demo]"

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
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

CMD ["uvicorn", "banking_triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
