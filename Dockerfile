FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

WORKDIR /app

# Install Python and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/

ENV UV_PYTHON=python3.10 \
    UV_SYSTEM_PYTHON=1

# Copy dependency manifest and lockfile first (layer cache optimisation).
COPY pyproject.toml uv.lock ./

# Install all runtime dependencies from the lockfile.
RUN uv sync --frozen --no-install-project --no-dev --extra cuda

# Copy source and install the project package.
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN uv sync --frozen --no-dev --extra cuda

VOLUME ["/app/configs", "/app/checkpoints", "/app/data"]

ENTRYPOINT ["uv", "run", "python3.10", "scripts/train.py"]
CMD ["--config", "configs/server.yaml"]
