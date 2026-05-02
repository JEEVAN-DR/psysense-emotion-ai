# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-prod.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements-prod.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime
LABEL maintainer="Hitan K <hitank2004@gmail.com>"
LABEL org.opencontainers.image.title="PsySense Emotion AI API"
LABEL org.opencontainers.image.description="Production FastAPI service for multi-label emotion classification"
LABEL org.opencontainers.image.source="https://github.com/Hitan547/psysense-emotion-ai"

WORKDIR /app
COPY --from=builder /install /usr/local
COPY config.py .
COPY inference.py .
COPY api/ api/
COPY model/ model/

RUN mkdir -p /app/.cache /app/.config /app/.local && \
    chmod -R 777 /app/.cache /app/.config /app/.local

RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --home /app appuser && \
    chown -R appuser:appgroup /app

USER appuser

ENV HOME=/app
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV MPLCONFIGDIR=/app/.cache/matplotlib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
