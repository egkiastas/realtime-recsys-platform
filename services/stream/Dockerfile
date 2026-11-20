FROM python:3.12.16-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra stream
COPY recsys recsys
COPY config.toml ./
COPY infra/feast infra/feast
COPY services/stream services/stream
CMD ["python", "-m", "services.stream.main"]
