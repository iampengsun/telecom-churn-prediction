FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/tmp/uv-cache PATH="/app/.venv/bin:$PATH"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.9.18
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra lightgbm \
    && rm -rf /tmp/uv-cache \
    && useradd --uid 10001 --create-home churn \
    && mkdir -p /app/artifacts /app/data /app/reports \
    && chown -R churn:churn /app
USER churn
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"
CMD ["churn", "serve", "--host", "0.0.0.0", "--port", "8000"]
