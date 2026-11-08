# Stage 1 installs dependencies. Only the finished environment moves to stage 2.
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
# Dependencies come before source, so code edits reuse this expensive cached layer.
COPY requirements-serve.txt .
RUN /opt/venv/bin/pip install -r requirements-serve.txt

FROM python:3.12-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    FLASK_DEBUG=0 \
    PORT=5000 \
    MODEL_PATH=/models/model.keras
RUN groupadd --system --gid 10001 appuser \
    && useradd --system --uid 10001 --gid appuser --home-dir /app --no-create-home --shell /usr/sbin/nologin appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY visionguard/ ./visionguard/
COPY webapp/ ./webapp/
COPY model_api/ ./model_api/
COPY docker/healthcheck.py ./docker/healthcheck.py
USER appuser
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD ["python", "docker/healthcheck.py"]
CMD ["flask", "--app", "webapp:create_app", "run", "--host=0.0.0.0", "--port=5000", "--no-debug"]
